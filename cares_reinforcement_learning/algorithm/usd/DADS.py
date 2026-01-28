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
import torch.nn.functional as F

from cares_reinforcement_learning.algorithm.algorithm import Algorithm
from cares_reinforcement_learning.algorithm.policy import SAC
from cares_reinforcement_learning.memory.memory_buffer import SARLMemoryBuffer
from cares_reinforcement_learning.networks.DADS import SkillDynamicsModel
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import SARLObservation
from cares_reinforcement_learning.util.configurations import DADSConfig


class DADS(Algorithm[SARLObservation, SARLMemoryBuffer]):
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
        self, observation: SARLObservation, evaluation: bool = False
    ) -> np.ndarray:

        observation = replace(
            observation,
            vector_state=self._concat_state_latent(observation.vector_state),
        )

        if not evaluation:
            self.z_experience_index.append(self.z)

        return self.skills_agent.select_action_from_policy(observation, evaluation)

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

        sample = memory_buffer.sample_uniform(self.batch_size)
        batch_size = len(sample.experiences)

        weights = [1.0] * batch_size

        zs = np.array(self.z_experience_index)[sample.indices]

        zs_tensor = torch.tensor(zs, dtype=torch.long, device=self.device).unsqueeze(-1)

        # Concatenate zs (skills) as one-hot to states
        zs_one_hot = np.eye(self.num_skills)[zs]

        states_tensor = np.stack(
            [experience.observation.vector_state for experience in sample.experiences]
        )
        states_zs = np.concatenate([states_tensor, zs_one_hot], axis=1)

        next_states_tensor = np.stack(
            [
                experience.next_observation.vector_state
                for experience in sample.experiences
            ]
        )
        next_states_zs = np.concatenate([next_states_tensor, zs_one_hot], axis=1)

        # Convert into tensor using training utilities
        states_tensor = torch.tensor(
            np.asarray(states_tensor), dtype=torch.float32, device=self.device
        )
        states_zs_tensor = torch.tensor(
            np.asarray(states_zs), dtype=torch.float32, device=self.device
        )

        actions_tensor = torch.tensor(
            np.asarray([experience.action for experience in sample.experiences]),
            dtype=torch.float32,
            device=self.device,
        )

        next_states_tensor = torch.tensor(
            np.asarray(next_states_tensor), dtype=torch.float32, device=self.device
        )
        next_states_zs_tensor = torch.tensor(
            np.asarray(next_states_zs), dtype=torch.float32, device=self.device
        )

        dones_tensor = torch.tensor(
            np.asarray([experience.done for experience in sample.experiences]),
            dtype=torch.long,
            device=self.device,
        )
        weights_tensor = torch.tensor(
            np.asarray(weights), dtype=torch.float32, device=self.device
        )

        # One-hot encode skill
        # pylint: disable-next=not-callable
        z_onehot = F.one_hot(zs_tensor.squeeze(-1), self.num_skills).float()

        # Predict next-state distribution
        mean, logvar = self.discriminator_net(states_tensor, z_onehot)

        use_delta_s = True
        target = (
            (next_states_tensor - states_tensor) if use_delta_s else next_states_tensor
        )

        # Compute Gaussian log-likelihood (sum over state dims)
        log_q_true = self._gaussian_log_likelihood(target, mean, logvar)

        # -----------------------------
        # TF-faithful marginal estimate
        # current + per-transition alternates (excluding current)
        # -----------------------------
        num_alt_skills = min(self.num_skills - 1, 16)  # TF: num_reps
        if num_alt_skills < 1:
            raise ValueError(
                "DADS requires num_skills >= 2 to form alternate-skill marginal."
            )

        num_transitions, state_dim = states_tensor.shape
        num_skills = self.num_skills

        # current skill indices per transition: [N]
        current_skill = zs_tensor.squeeze(-1)  # long, shape [N]

        # Sample per-transition alternate skills uniformly from {0..K-1}\{current_skill}
        # Vectorized "skip" trick:
        # 1) sample offsets in [0, K-2]
        alt_offsets = torch.randint(
            low=0,
            high=num_skills - 1,
            size=(num_transitions, num_alt_skills),
            device=self.device,
        )  # [N, M]

        # 2) map offsets to actual skill ids in [0, K-1], skipping current_skill
        alt_skill_ids = (
            alt_offsets + (alt_offsets >= current_skill.unsqueeze(1)).long()
        )  # [N, M]

        # One-hot: [N, M, K]
        # pylint: disable-next=not-callable
        alt_skill_onehot = F.one_hot(alt_skill_ids, num_skills).float()

        # Repeat states/targets across the M alternates: [N*M, D]
        states_rep = (
            states_tensor.unsqueeze(1)
            .expand(num_transitions, num_alt_skills, state_dim)
            .reshape(num_transitions * num_alt_skills, state_dim)
        )
        target_rep = (
            target.unsqueeze(1)
            .expand(num_transitions, num_alt_skills, state_dim)
            .reshape(num_transitions * num_alt_skills, state_dim)
        )

        # Flatten skills: [N*M, K]
        alt_skills_rep = alt_skill_onehot.reshape(
            num_transitions * num_alt_skills, num_skills
        )

        # Evaluate log q for alternates: log q(Δs | s, z_alt)
        mean_alt, logvar_alt = self.discriminator_net(states_rep, alt_skills_rep)
        log_q_alt = self._gaussian_log_likelihood(
            target_rep, mean_alt, logvar_alt
        ).view(
            num_transitions, num_alt_skills
        )  # [N, M]

        # TF-style denominator includes current-skill likelihood explicitly:
        # log( (q_z + sum q_alt) / (M+1) ) computed stably in log-space
        log_q_all = torch.cat([log_q_true.unsqueeze(1), log_q_alt], dim=1)  # [N, M+1]
        log_marginal = torch.logsumexp(log_q_all, dim=1) - math.log(
            num_alt_skills + 1
        )  # [N]

        # Intrinsic reward (detach before SAC update)
        rewards_tensor = (log_q_true - log_marginal).detach().unsqueeze(-1)  # [N, 1]

        with torch.no_grad():
            # reward stats
            r = rewards_tensor.squeeze(-1)  # [N]
            info["dads_reward_mean"] = r.mean().item()
            info["dads_reward_std"] = r.std(unbiased=False).item()
            info["dads_reward_min"] = r.min().item()
            info["dads_reward_max"] = r.max().item()

            # target (Δs) magnitude
            abs_target = target.abs()
            info["dads_target_abs_mean"] = abs_target.mean().item()
            info["dads_target_abs_max"] = abs_target.max().item()

            # log_q stats
            info["dads_logq_true_mean"] = log_q_true.mean().item()
            info["dads_logq_true_std"] = log_q_true.std(unbiased=False).item()
            info["dads_log_marginal_mean"] = log_marginal.mean().item()

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
            np.asarray(sample.indices),
            states_zs_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_zs_tensor,
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
        p_z = np.full(self.num_skills, 1 / self.num_skills)
        self.z = np.random.choice(self.num_skills, p=p_z)

        return super().episode_done()

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        # Save skills agent and discriminator
        self.skills_agent.save_models(filepath, f"{filename}_skill_agent")

        checkpoint = {
            "discriminator": self.discriminator_net.state_dict(),
            "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
            "z_experience_index": self.z_experience_index,
            "z": self.z,
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
        self.z_experience_index = checkpoint.get(
            "z_experience_index", self.z_experience_index
        )
        self.z = checkpoint.get("z", self.z)
        logging.info("models, optimisers, and DADS state have been loaded...")
