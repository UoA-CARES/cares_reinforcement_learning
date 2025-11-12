"""
DIAYN (Diversity Is All You Need) implementation: https://arxiv.org/pdf/1802.06070

Code: https://github.com/alirezakazemipour/DIAYN-PyTorch/tree/main
"""

import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from cares_reinforcement_learning.algorithm.algorithm import VectorAlgorithm
from cares_reinforcement_learning.algorithm.policy import SAC
from cares_reinforcement_learning.networks.DIAYN import Discriminator
from cares_reinforcement_learning.util.configurations import DIAYNConfig
from cares_reinforcement_learning.util.training_context import (
    ActionContext,
    TrainingContext,
)


class DIAYN(VectorAlgorithm):
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

        p_z = np.full(self.num_skills, 1 / self.num_skills)
        self.p_z = np.tile(p_z, self.batch_size).reshape(
            self.batch_size, self.num_skills
        )

        self.z = np.random.choice(self.num_skills, p=p_z)

        self.z_experience_index = []
        self.z_experience_index.append(self.z)

        self.cross_ent_loss = torch.nn.CrossEntropyLoss()

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

    def select_action_from_policy(self, action_context: ActionContext) -> np.ndarray:

        state = action_context.state
        evaluation = action_context.evaluation

        assert isinstance(state, np.ndarray)

        action_context.state = self._concat_state_latent(state)

        if not evaluation:
            self.z_experience_index.append(self.z)

        return self.skills_agent.select_action_from_policy(action_context)

    def _calculate_value(self, state: np.ndarray, action: np.ndarray) -> float:  # type: ignore[override]
        state = self._concat_state_latent(state)

        return self.skills_agent._calculate_value(state, action)

    def train_policy(self, training_context: TrainingContext) -> dict[str, Any]:
        memory = training_context.memory
        batch_size = training_context.batch_size

        if len(memory) < batch_size:
            return {}

        experiences = memory.sample_uniform(batch_size)
        states, actions, _, next_states, dones, indices = experiences
        weights = [1.0] * batch_size

        batch_size = len(states)

        zs = np.array(self.z_experience_index)[indices]

        zs_tensor = torch.tensor(zs, dtype=torch.long, device=self.device).unsqueeze(-1)

        # Concatenate zs (skills) as one-hot to states
        zs_one_hot = np.eye(self.num_skills)[zs]
        states_zs = np.concatenate([states, zs_one_hot], axis=1)
        next_states_zs = np.concatenate([next_states, zs_one_hot], axis=1)

        # Convert into tensor using training utilities
        states_tensor = torch.tensor(
            np.asarray(states), dtype=torch.float32, device=self.device
        )
        states_zs_tensor = torch.tensor(
            np.asarray(states_zs), dtype=torch.float32, device=self.device
        )

        actions_tensor = torch.tensor(
            np.asarray(actions), dtype=torch.float32, device=self.device
        )

        next_states_tensor = torch.tensor(
            np.asarray(next_states), dtype=torch.float32, device=self.device
        )
        next_states_zs_tensor = torch.tensor(
            np.asarray(next_states_zs), dtype=torch.float32, device=self.device
        )

        dones_tensor = torch.tensor(
            np.asarray(dones), dtype=torch.long, device=self.device
        )
        weights_tensor = torch.tensor(
            np.asarray(weights), dtype=torch.float32, device=self.device
        )

        # Dervive rewards from the discriminator
        p_z = torch.tensor(self.p_z, dtype=torch.float32, device=self.device)

        logits = self.discriminator_net(next_states_tensor)
        p_z = p_z.gather(-1, zs_tensor)
        logq_z_ns = F.log_softmax(logits, dim=-1)
        rewards_tensor = logq_z_ns.gather(-1, zs_tensor).detach() - torch.log(
            p_z + 1e-6
        )

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
        logits = self.discriminator_net(states_tensor)
        discriminator_loss = self.cross_ent_loss(logits, zs_tensor.squeeze(-1))

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

        # Save skills agent
        self.skills_agent.save_models(filepath, f"{filename}_skill_agent")

        # Save DIAYN-specific state in a single checkpoint
        checkpoint = {
            "discriminator_state_dict": self.discriminator_net.state_dict(),
            "discriminator_optimizer_state_dict": self.discriminator_optimizer.state_dict(),
            "z": self.z,
            "z_experience_index": self.z_experience_index,
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
        self.z_experience_index = checkpoint.get(
            "z_experience_index", self.z_experience_index
        )
        logging.info("DIAYN models and state have been loaded...")
