"""
DIAYN (Diversity Is All You Need)
----------------------------------

Original Paper: https://arxiv.org/pdf/1802.06070

Original Code: Code: https://github.com/alirezakazemipour/DIAYN-PyTorch/tree/main

DIAYN is an unsupervised skill discovery method that learns a
diverse set of behaviors without using external rewards.

Core Idea:
- Introduce a latent skill variable z.
- Train a skill-conditioned policy:
      π(a | s, z)
- Encourage skills to visit distinct states.

Objective:
    max  I(z ; s)

Mutual information between skill z and visited state s is
maximized while keeping the policy high-entropy.

Variational Formulation:
- Introduce a discriminator q_φ(z | s).
- Intrinsic reward:
      r(s, z) = log q_φ(z | s) - log p(z)

The policy is rewarded when the discriminator can correctly
infer which skill generated the state.

Training:
- Skills z sampled from a fixed prior (typically uniform).
- Off-policy RL (commonly SAC) optimizes the policy.
- Discriminator trained via cross-entropy classification.

Key Behaviour:
- Skills become distinguishable in state space.
- No task reward required.
- Encourages diverse, steady-state behaviors.

Differences from Dynamics-Aware Methods (e.g., DADS):
- DIAYN maximizes I(z; s).
- Does not explicitly model environment dynamics.
- Diversity is state-based rather than transition-based.

Advantages:
- Simple and scalable.
- Produces reusable skill primitives.
- Useful for pretraining and hierarchical RL.

DIAYN = unsupervised skill learning via
        mutual information between skill and state.
"""

import logging
import os
from dataclasses import replace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.memory.memory_sampler as memory_sampler
from cares_reinforcement_learning.algorithm.algorithm import SARLAlgorithm
from cares_reinforcement_learning.algorithm.policy import SAC
from cares_reinforcement_learning.memory.memory_buffer import SARLMemoryBuffer
from cares_reinforcement_learning.networks.DIAYN import Discriminator
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import SARLObservation
from cares_reinforcement_learning.util.configurations import DIAYNConfig


class DIAYN(SARLAlgorithm[np.ndarray]):
    def __init__(
        self,
        skills_agent: SAC,
        discriminator_network: Discriminator,
        config: DIAYNConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="usd", config=config, device=device)

        self.skills_agent = skills_agent
        self.discriminator_net = discriminator_network.to(device)

        self.num_skills = config.num_skills

        self.p_z = np.full(
            self.num_skills, 1.0 / self.num_skills, dtype=np.float32
        )  # (K,)

        self.z = np.random.choice(self.num_skills, p=self.p_z)

        self.cross_ent_loss = torch.nn.CrossEntropyLoss()

        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator_net.parameters(), lr=config.discriminator_lr
        )

    def set_skill(self, skill: int, evaluation: bool = False) -> None:
        if skill < 0 or skill >= self.num_skills:
            raise ValueError(f"Skill index {skill} is out of bounds.")

        self.z = skill

    def _concat_state_latent(self, state: np.ndarray) -> np.ndarray:
        z_one_hot = np.zeros(self.num_skills)
        z_one_hot[self.z] = 1
        return np.concatenate([state, z_one_hot])

    def act(
        self, observation: SARLObservation, evaluation: bool = False
    ) -> ActionSample[np.ndarray]:

        observation = replace(
            observation,
            vector_state=self._concat_state_latent(observation.vector_state),
        )

        action_sample = self.skills_agent.act(observation, evaluation)
        action_sample.extras["skill"] = self.z
        return action_sample

    def _calculate_value(self, state: SARLObservation, action: np.ndarray) -> float:  # type: ignore[override]
        state = replace(
            state, vector_state=self._concat_state_latent(state.vector_state)
        )

        return self.skills_agent._calculate_value(state, action)

    def train(
        self,
        memory_buffer: SARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:

        info: dict[str, Any] = {}

        if len(memory_buffer) < self.batch_size:
            return {}

        (
            observation_tensor,
            actions_tensor,
            rewards_tensor,
            next_observation_tensor,
            dones_tensor,
            weights_tensor,
            train_data,
            indices,
        ) = memory_sampler.sample(
            memory=memory_buffer,
            batch_size=self.batch_size,
            device=self.device,
            use_per_buffer=0,  # DIAYN does not use PER
        )

        batch_size = len(observation_tensor.vector_state_tensor)

        skills = [extra["skill"] for extra in train_data]
        zs_tensor = torch.tensor(skills, dtype=torch.long, device=self.device)

        # Concatenate zs (skills) as one-hot to states
        # pylint: disable-next=not-callable
        zs_one_hot = F.one_hot(zs_tensor, num_classes=self.num_skills).to(
            observation_tensor.vector_state_tensor.dtype
        )

        states_z_tensor = torch.cat(
            [observation_tensor.vector_state_tensor, zs_one_hot], dim=1
        )

        next_states_z_tensor = torch.cat(
            [next_observation_tensor.vector_state_tensor, zs_one_hot], dim=1
        )

        # Derive rewards from the discriminator
        p_z = torch.as_tensor(self.p_z, dtype=torch.float32, device=self.device)  # (K,)

        # Choose state vs next_state; keeping your current next_state choice:
        logits = self.discriminator_net(
            observation_tensor.vector_state_tensor
        )  # (B, K)

        z_idx = zs_tensor.unsqueeze(1)  # (B, 1)
        logq = F.log_softmax(logits, dim=-1)  # (B, K)
        logq_z = logq.gather(1, z_idx)  # (B, 1)

        p_z_batch = p_z[zs_tensor].unsqueeze(1)  # (B, 1)
        rewards_tensor = logq_z.detach() - torch.log(p_z_batch + 1e-6)  # (B, 1)

        # Reshape to batch_size x whatever
        dones_tensor = dones_tensor.reshape(batch_size, 1)
        weights_tensor = weights_tensor.reshape(batch_size, 1)

        # ---- DIAYN diagnostics (CSV-friendly scalars) ----
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)  # (B, K)
            pred_z = probs.argmax(dim=-1)  # (B,)

            disc_acc = (pred_z == zs_tensor).float().mean()
            disc_entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
            mean_logq_z = logq_z.mean()

            r = rewards_tensor.squeeze(1)
            r_mean = r.mean()
            r_std = r.std(unbiased=False)
            r_p10 = torch.quantile(r, 0.10)
            r_p50 = torch.quantile(r, 0.50)
            r_p90 = torch.quantile(r, 0.90)

            z_unique = zs_tensor.unique().numel()
        # -----------------------------------------------

        observation_z_tensor = replace(
            observation_tensor,
            vector_state_tensor=states_z_tensor,
        )
        next_observation_z_tensor = replace(
            next_observation_tensor,
            vector_state_tensor=next_states_z_tensor,
        )

        agent_info, _ = self.skills_agent.update_from_batch(
            observation_tensor=observation_z_tensor,
            actions_tensor=actions_tensor,
            rewards_tensor=rewards_tensor,
            next_observation_tensor=next_observation_z_tensor,
            dones_tensor=dones_tensor,
            weights_tensor=weights_tensor,
        )
        info |= agent_info

        # Add DIAYN diagnostics to info dict (numbers only)
        info.update(
            {
                "disc_acc": float(disc_acc.item()),
                "disc_entropy": float(disc_entropy.item()),
                "mean_logq_z": float(mean_logq_z.item()),
                "r_mean": float(r_mean.item()),
                "r_std": float(r_std.item()),
                "r_p10": float(r_p10.item()),
                "r_p50": float(r_p50.item()),
                "r_p90": float(r_p90.item()),
                "z_unique": int(z_unique),
            }
        )

        # Update the Discriminator
        discriminator_loss = self.cross_ent_loss(logits, zs_tensor)

        info["discriminator_loss"] = discriminator_loss.item()

        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        return info

    def episode_done(self):
        self.z = np.random.choice(self.num_skills, p=self.p_z)

        return super().episode_done()

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        # Save skills agent
        self.skills_agent.save_models(filepath, f"{filename}_skill_agent")

        # Save DIAYN-specific state in a single checkpoint
        checkpoint = {
            "discriminator_state_dict": self.discriminator_net.state_dict(),
            "discriminator_optimizer_state_dict": self.discriminator_optimizer.state_dict(),
            "z": self.z,
        }
        torch.save(checkpoint, f"{filepath}/{filename}_diayn.pth")
        logging.info("DIAYN models and state have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        self.skills_agent.load_models(filepath, f"{filename}_skill_agent")

        checkpoint = torch.load(f"{filepath}/{filename}_diayn.pth")

        self.discriminator_net.load_state_dict(checkpoint["discriminator_state_dict"])
        self.discriminator_optimizer.load_state_dict(
            checkpoint["discriminator_optimizer_state_dict"]
        )

        self.z = checkpoint.get("z", self.z)
        logging.info("DIAYN models and state have been loaded...")
