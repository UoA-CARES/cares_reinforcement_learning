"""
DADS DYNAMICS-AWARE DISCOVERY OF SKILLS https://arxiv.org/pdf/1907.01657

Code: https://github.com/google-research/dads
"""

import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from cares_reinforcement_learning.algorithm.algorithm import VectorAlgorithm
from cares_reinforcement_learning.algorithm.policy import SAC
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.networks.DADS import SkillDynamicsModel
from cares_reinforcement_learning.util.configurations import DADSConfig


class DADS(VectorAlgorithm):
    def __init__(
        self,
        skills_agent: SAC,
        discriminator_network: SkillDynamicsModel,
        config: DADSConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="usd", config=config, device=device)

        self.skills_agent = skills_agent
        self.discriminator_net = discriminator_network.to(device)

        self.num_skills = config.num_skills

        p_z = np.full(self.num_skills, 1 / self.num_skills)
        self.z = np.random.choice(self.num_skills, p=p_z)

        self.z_experience_index = []
        self.z_experience_index.append(self.z)

        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator_net.parameters(), lr=config.discriminator_lr
        )

    def set_skill(self, skill: int, evaluation: bool = False) -> None:
        if skill < 0 or skill >= self.num_skills:
            raise ValueError(f"Skill index {skill} is out of bounds.")

        self.z = skill

        if not evaluation:
            self.z_experience_index.append(self.z)

    def _concat_state_latent(self, state: np.ndarray) -> np.ndarray:
        z_one_hot = np.zeros(self.num_skills)
        z_one_hot[self.z] = 1
        return np.concatenate([state, z_one_hot])

    def select_action_from_policy(
        self, state: np.ndarray, evaluation: bool = False
    ) -> np.ndarray:
        state = self._concat_state_latent(state)

        if not evaluation:
            self.z_experience_index.append(self.z)

        return self.skills_agent.select_action_from_policy(state, evaluation)

    def _calculate_value(self, state: np.ndarray, action: np.ndarray) -> float:  # type: ignore[override]
        state = self._concat_state_latent(state)

        return self.skills_agent._calculate_value(state, action)

    def train_policy(
        self, memory: MemoryBuffer, batch_size: int, training_step: int
    ) -> dict[str, Any]:
        if len(memory) < batch_size:
            return {}

        experiences = memory.sample_uniform(batch_size)
        states, actions, _, next_states, dones, indices = experiences
        weights = [1.0] * batch_size

        batch_size = len(states)

        zs = np.array(self.z_experience_index)[indices]

        zs_tensor = torch.LongTensor(zs).unsqueeze(-1).to(self.device)

        # Concatenate zs (skills) as one-hot to states
        zs_one_hot = np.eye(self.num_skills)[zs]
        states_zs = np.concatenate([states, zs_one_hot], axis=1)
        next_states_zs = np.concatenate([next_states, zs_one_hot], axis=1)

        # Convert into tensor
        states_tensor = torch.FloatTensor(np.asarray(states)).to(self.device)
        states_zs_tensor = torch.FloatTensor(np.asarray(states_zs)).to(self.device)

        actions_tensor = torch.FloatTensor(np.asarray(actions)).to(self.device)

        next_states_tensor = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        next_states_zs_tensor = torch.FloatTensor(np.asarray(next_states_zs)).to(
            self.device
        )

        dones_tensor = torch.LongTensor(np.asarray(dones)).to(self.device)
        weights_tensor = torch.FloatTensor(np.asarray(weights)).to(self.device)

        # One-hot encode skill
        # pylint: disable-next=not-callable
        z_onehot = F.one_hot(zs_tensor.squeeze(-1), self.num_skills).float()

        # Predict next-state distribution
        mean, logvar = self.discriminator_net(states_tensor, z_onehot)

        # Compute Gaussian log-likelihood (sum over state dims)
        epsilon = 1e-6
        log_likelihood = -0.5 * (
            ((next_states_tensor - mean) ** 2) / (logvar.exp() + epsilon)
            + logvar
            + np.log(2 * np.pi)
        ).sum(dim=-1)

        # Use as intrinsic reward (detached)
        rewards_tensor = log_likelihood.detach().unsqueeze(-1)

        # Reshape to batch_size x whatever
        # rewards_tensor = rewards_tensor.reshape(batch_size, 1)
        dones_tensor = dones_tensor.reshape(batch_size, 1)
        weights_tensor = weights_tensor.reshape(batch_size, 1)
        rewards_tensor = rewards_tensor.reshape(batch_size, 1)

        info = self.skills_agent.update_networks(
            memory,
            indices,
            states_zs_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_zs_tensor,
            dones_tensor,
            weights_tensor,
        )

        # Update the Discriminator
        discriminator_loss = -log_likelihood.mean()

        info["discriminator_loss"] = discriminator_loss.item()

        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        return info

    def episode_done(self):
        p_z = np.full(self.num_skills, 1 / self.num_skills)
        self.z = np.random.choice(self.num_skills, p=p_z)

        return super().episode_done()

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        self.skills_agent.save_models(filepath, filename)
        torch.save(
            self.discriminator_net.state_dict(),
            f"{filepath}/{filename}_discriminator.pht",
        )
        logging.info("models has been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        self.skills_agent.load_models(filepath, filename)
        self.discriminator_net.load_state_dict(
            torch.load(f"{filepath}/{filename}_discriminator.pht")
        )
        logging.info("models has been loaded...")
