"""
DADS DYNAMICS-AWARE DISCOVERY OF SKILLS https://arxiv.org/pdf/1907.01657

Code: https://github.com/google-research/dads
"""

import logging
import math
import os
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from cares_reinforcement_learning.algorithm.algorithm import Algorithm
from cares_reinforcement_learning.algorithm.policy import SAC
from cares_reinforcement_learning.memory import memory_sampler
from cares_reinforcement_learning.memory.memory_buffer import SARLMemoryBuffer
from cares_reinforcement_learning.networks.DADS import SkillDynamicsModel
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import SARLObservation
from cares_reinforcement_learning.util.configurations import DADSConfig


class DADS(Algorithm[SARLObservation, np.ndarray, SARLMemoryBuffer]):
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

        self.z_dim = config.z_dim
        self.z = np.random.randn(self.z_dim).astype(np.float32)  # z ~ N(0, I)

        rng = np.random.default_rng(100)
        self.eval_z_radius = getattr(config, "eval_z_radius", 2.0)
        self.eval_z_bank = rng.standard_normal(
            size=(self.num_skills, self.z_dim)
        ).astype(np.float32)

        norms = np.linalg.norm(self.eval_z_bank, axis=1, keepdims=True) + 1e-8
        self.eval_z_bank = (self.eval_z_bank / norms) * self.eval_z_radius

        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator_net.parameters(), lr=config.discriminator_lr
        )

    def set_skill(self, skill: int, evaluation: bool = False) -> None:
        if skill < 0 or skill >= self.num_skills:
            raise ValueError(f"Skill index {skill} is out of bounds.")

        self.z = self.eval_z_bank[skill]

    def _concat_state_latent(self, state: np.ndarray) -> np.ndarray:
        return np.concatenate([state, self.z])

    def select_action_from_policy(
        self, observation: SARLObservation, evaluation: bool = False
    ) -> ActionSample[np.ndarray]:

        observation = replace(
            observation,
            vector_state=self._concat_state_latent(observation.vector_state),
        )

        action_sample = self.skills_agent.select_action_from_policy(
            observation, evaluation
        )
        action_sample.extras["z"] = self.z.copy()
        return action_sample

    def _calculate_value(self, state: SARLObservation, action: np.ndarray) -> float:  # type: ignore[override]
        state = replace(
            state, vector_state=self._concat_state_latent(state.vector_state)
        )

        return self.skills_agent._calculate_value(state, action)

    def _gaussian_log_likelihood(
        self,
        x: torch.Tensor,  # [B, D]
        mean: torch.Tensor,  # [B, D]
        logvar: torch.Tensor,  # [B, D]  (log variance)
    ) -> torch.Tensor:
        """
        Stable log N(x | mean, exp(logvar)).
        Returns shape [B] (sum over feature dim).
        """
        # Clamp log-variance for numerical stability
        inv_var = torch.exp(-logvar)

        # log-likelihood per-dimension then sum over feature dim
        log_likelihood = -0.5 * (
            (x - mean) ** 2 * inv_var + logvar + math.log(2.0 * math.pi)
        )
        return log_likelihood.sum(dim=-1)  # [B]

    def train_policy(
        self,
        memory_buffer: SARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:

        info: dict[str, Any] = {}

        if len(memory_buffer) < self.batch_size:
            return info

        (
            observation_tensor,
            actions_tensor,
            rewards_tensor,
            next_observation_tensor,
            dones_tensor,
            weights_tensor,
            extras,
            indices,
        ) = memory_sampler.sample(
            memory=memory_buffer,
            batch_size=self.batch_size,
            device=self.device,
            use_per_buffer=0,  # DADS does not use PER
        )

        z_list = [extra["z"] for extra in extras]
        z_tensor = torch.tensor(
            np.asarray(z_list), dtype=torch.float32, device=self.device
        )  # (B, z_dim)

        states_z_tensor = torch.cat(
            [observation_tensor.vector_state_tensor, z_tensor], dim=1
        )
        next_states_z_tensor = torch.cat(
            [next_observation_tensor.vector_state_tensor, z_tensor], dim=1
        )

        # Predict next-state distribution
        mean, logvar = self.discriminator_net(
            observation_tensor.vector_state_tensor, z_tensor
        )

        use_delta_s = True
        target = (
            (
                next_observation_tensor.vector_state_tensor
                - observation_tensor.vector_state_tensor
            )
            if use_delta_s
            else next_observation_tensor.vector_state_tensor
        )

        # Compute Gaussian log-likelihood (sum over state dims)
        log_q_true = self._gaussian_log_likelihood(target, mean, logvar)

        # -----------------------------
        # TF-faithful marginal estimate
        # current + per-transition alternates (excluding current)
        num_skills = 16
        batch_size, state_dim = observation_tensor.vector_state_tensor.shape

        z_alt = torch.randn(
            batch_size, num_skills, self.z_dim, device=self.device
        )  # (N, M, dz)

        states_rep = (
            observation_tensor.vector_state_tensor.unsqueeze(1)
            .expand(batch_size, num_skills, state_dim)
            .reshape(batch_size * num_skills, state_dim)
        )
        target_rep = (
            target.unsqueeze(1)
            .expand(batch_size, num_skills, state_dim)
            .reshape(batch_size * num_skills, state_dim)
        )
        z_alt_rep = z_alt.reshape(batch_size * num_skills, self.z_dim)  # (N*M, dz)

        mean_alt, logvar_alt = self.discriminator_net(states_rep, z_alt_rep)
        log_q_alt = self._gaussian_log_likelihood(
            target_rep, mean_alt, logvar_alt
        ).view(batch_size, num_skills)

        log_q_all = torch.cat([log_q_true.unsqueeze(1), log_q_alt], dim=1)  # (N, M+1)
        log_marginal = torch.logsumexp(log_q_all, dim=1) - math.log(num_skills + 1)
        rewards_tensor = (log_q_true - log_marginal).detach().unsqueeze(1)  # (N, 1)

        with torch.no_grad():
            # reward stats
            r = rewards_tensor.squeeze(-1)  # [N]
            info["reward_mean"] = r.mean().item()
            info["reward_std"] = r.std(unbiased=False).item()
            info["reward_min"] = r.min().item()
            info["reward_max"] = r.max().item()

            # target (Δs) magnitude
            abs_target = target.abs()
            info["target_abs_mean"] = abs_target.mean().item()
            info["target_abs_max"] = abs_target.max().item()

            # log_q stats
            info["logq_true_mean"] = log_q_true.mean().item()
            info["logq_true_std"] = log_q_true.std(unbiased=False).item()
            info["log_marginal_mean"] = log_marginal.mean().item()

            # variance / logvar stats (current skill)
            info["disc_logvar_cur_min"] = logvar.min().item()
            info["disc_logvar_cur_mean"] = logvar.mean().item()
            info["disc_logvar_cur_max"] = logvar.max().item()

            # variance / logvar stats (alternate skills)
            info["disc_logvar_alt_min"] = logvar_alt.min().item()
            info["disc_logvar_alt_mean"] = logvar_alt.mean().item()
            info["disc_logvar_alt_max"] = logvar_alt.max().item()

        # Reshape to batch_size x whatever
        # rewards_tensor = rewards_tensor.reshape(batch_size, 1)
        dones_tensor = dones_tensor.reshape(batch_size, 1)
        weights_tensor = weights_tensor.reshape(batch_size, 1)
        rewards_tensor = rewards_tensor.reshape(batch_size, 1)

        info |= self.skills_agent.update_networks(
            memory_buffer,
            indices,
            states_z_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_z_tensor,
            dones_tensor,
            weights_tensor,
        )

        # Update the Discriminator
        discriminator_loss = -log_q_true.mean()

        info["discriminator_loss"] = discriminator_loss.item()

        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        return info

    def episode_done(self):
        self.z = np.random.randn(self.z_dim).astype(np.float32)

        return super().episode_done()

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        # Save skills agent and discriminator
        self.skills_agent.save_models(filepath, f"{filename}_skill_agent")

        checkpoint = {
            "discriminator": self.discriminator_net.state_dict(),
            "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
        }
        torch.save(checkpoint, f"{filepath}/{filename}_dads.pth")
        logging.info("models, optimisers, and DADS state have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        self.skills_agent.load_models(filepath, f"{filename}_skill_agent")

        checkpoint = torch.load(f"{filepath}/{filename}_dads.pth")
        self.discriminator_net.load_state_dict(checkpoint["discriminator"])
        self.discriminator_optimizer.load_state_dict(
            checkpoint["discriminator_optimizer"]
        )
        logging.info("models, optimisers, and DADS state have been loaded...")
