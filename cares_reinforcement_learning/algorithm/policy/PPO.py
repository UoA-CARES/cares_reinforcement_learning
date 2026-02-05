"""
PPO (Proximal Policy Optimization) implementation notes
--------------------------------------------------------------
Original Paper: https://arxiv.org/abs/1707.06347

This implementation follows the clipped-surrogate PPO formulation with
Generalized Advantage Estimation (GAE), minibatch SGD, and optional KL-based
early stopping.

Rollout collection:
- Experience is collected strictly on-policy using the current stochastic policy.
- For each step, the sampled action, its log-probability under the behavior
  policy, and the critic value V(s) are stored.

Advantage estimation:
- Advantages are computed using Generalized Advantage Estimation (GAE),
  bootstrapped from a single final value for truncated rollouts.
- Returns for critic updates are computed as advantage + value.
- Advantages are normalized across the batch for stability.

Policy and value updates:
- The actor is optimized using the PPO clipped surrogate objective with an
  optional entropy bonus.
- The critic is trained by regression onto the computed returns.
- Updates are performed using multiple epochs of minibatch SGD over the
  same on-policy rollout.
- Gradient norm clipping is applied to improve numerical stability.

KL control:
- An approximate KL divergence between the old and updated policy is monitored.
- If the KL exceeds a configured threshold, further policy updates for the
  current iteration are stopped early, providing an additional trust-region
  constraint beyond clipping.
"""

import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

import cares_reinforcement_learning.memory.memory_sampler as memory_sampler
from cares_reinforcement_learning.algorithm.algorithm import Algorithm
from cares_reinforcement_learning.memory.memory_buffer import SARLMemoryBuffer
from cares_reinforcement_learning.networks.PPO import Actor, Critic
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import SARLObservation
from cares_reinforcement_learning.util.configurations import PPOConfig
from cares_reinforcement_learning.types.action import ActionSample


class PPO(Algorithm[SARLObservation, np.ndarray, SARLMemoryBuffer]):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: PPOConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        self.actor_net = actor_network.to(device)
        self.critic_net = critic_network.to(device)

        self.gamma = config.gamma
        self.action_num = self.actor_net.num_actions
        self.device = device

        self.gae_lambda = config.gae_lambda
        self.minibatch_size = config.minibatch_size

        self.gae_lambda = config.gae_lambda
        self.entropy_coef = config.entropy_coef
        self.target_kl = config.target_kl

        self.max_grad_norm = config.max_grad_norm

        init_log_std = torch.log(torch.sqrt(torch.tensor(0.5, device=self.device)))
        self.log_std = torch.nn.Parameter(init_log_std.repeat(self.action_num))

        self.actor_net_optimiser = torch.optim.Adam(
            list(self.actor_net.parameters()) + [self.log_std], lr=config.actor_lr
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=config.critic_lr
        )

        self.updates_per_iteration = config.updates_per_iteration
        self.eps_clip = config.eps_clip

    def _dist(self, mean: torch.Tensor) -> MultivariateNormal:
        # mean: [B, act_dim] or [act_dim]
        std = self.log_std.exp()  # [act_dim]
        cov = torch.diag(std * std)  # [act_dim, act_dim] (broadcasts across batch)
        return MultivariateNormal(mean, covariance_matrix=cov)

    def _calculate_log_prob(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        self.actor_net.eval()
        with torch.no_grad():
            mean = self.actor_net(state)
            dist = self._dist(mean)
            log_prob = dist.log_prob(action)

        self.actor_net.train()
        return log_prob

    def act(
        self, observation: SARLObservation, evaluation: bool = False
    ) -> ActionSample[np.ndarray]:
        self.actor_net.eval()
        self.critic_net.eval()
        state = observation.vector_state

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            state_tensor = state_tensor.unsqueeze(0)  # add batch dimension

            mean = self.actor_net(state_tensor)
            dist = self._dist(mean)

            sample = mean if evaluation else dist.sample()

            # Clamp the action you will actually execute
            sample = sample.clamp(-1.0, 1.0)

            log_prob = dist.log_prob(sample)  # store π_old(a|s) for PPO ratio

            value = self.critic_net(state_tensor).squeeze(-1)  # shape [1]

            action = sample.squeeze(0).cpu().numpy()

        self.actor_net.train()
        self.critic_net.train()

        return ActionSample(
            action=action,
            source="policy",
            extras={"log_prob": float(log_prob.item()), "value": float(value.item())},
        )

    def _calculate_value(self, state: SARLObservation, action: np.ndarray) -> float:  # type: ignore[override]
        state_tensor = torch.tensor(
            state.vector_state, dtype=torch.float32, device=self.device
        )
        state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            value = self.critic_net(state_tensor)

        return value[0].item()

    def _calculate_gae(
        self,
        rewards: torch.Tensor,  # [N] or [N,1]
        dones: torch.Tensor,  # [N] or [N,1] (1.0 if done else 0.0)
        values: torch.Tensor,  # [N]
        last_value: torch.Tensor,  # scalar tensor, V(s_T) for bootstrap
        gae_lambda: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rewards = rewards.view(-1)
        dones = dones.view(-1)
        values = values.view(-1)

        batch_size = rewards.shape[0]
        advantages = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        gae = 0.0
        next_value = last_value

        for t in reversed(range(batch_size)):
            mask = 1.0 - dones[t]  # 0 if terminal else 1
            delta = rewards[t] + self.gamma * mask * next_value - values[t]
            gae = delta + self.gamma * gae_lambda * mask * gae
            advantages[t] = gae
            next_value = values[t]

        returns = advantages + values
        return advantages, returns

    def train_policy(
        self,
        memory_buffer: SARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:
        # pylint: disable-next=unused-argument

        sample = memory_buffer.flush()
        batch_size = len(sample.experiences)

        # Convert to tensors using helper method (no next_states needed for PPO, so pass dummy data)
        (
            observation_tensor,
            actions_tensor,
            rewards_tensor,
            next_observation_tensor,
            dones_tensor,
            _,  # weights not needed
            _,
        ) = memory_sampler.sample_to_tensors(sample, self.device)

        states = observation_tensor.vector_state_tensor  # shape [B, obs_dim]

        # Old log_probs + values stored at action time
        old_log_probs = [
            experience.train_data["log_prob"] for experience in sample.experiences
        ]
        old_log_probs_tensor = torch.tensor(
            np.asarray(old_log_probs), dtype=torch.float32, device=self.device
        )

        old_values = [
            experience.train_data["value"] for experience in sample.experiences
        ]
        old_values_tensor = torch.tensor(
            np.asarray(old_values), dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            last_next_state = next_observation_tensor.vector_state_tensor[-1].unsqueeze(
                0
            )  # [1, obs_dim]
            last_value = self.critic_net(last_next_state).view(-1)[0]  # scalar

        # GAE: compute advantages using GAE, which uses critic values to bootstrap:
        advantages, returns = self._calculate_gae(
            rewards=rewards_tensor,
            dones=dones_tensor,
            values=old_values_tensor,
            last_value=last_value,
            gae_lambda=self.gae_lambda,
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        mb_size = min(self.minibatch_size, batch_size)

        actions_tensor = actions_tensor.view(-1, self.action_num)  # [N, act_dim]
        old_log_probs_tensor = old_log_probs_tensor.view(-1)  # [N]
        advantages = advantages.view(-1)  # [N]
        returns = returns.view(-1)  # [N]

        for _ in range(self.updates_per_iteration):
            idx = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, mb_size):
                mb = idx[start : start + mb_size]

                states_mb = states[mb]
                actions_mb = actions_tensor[mb]
                old_logp_mb = old_log_probs_tensor[mb].detach()
                advantages_mb = advantages[mb]
                returns_mb = returns[mb]

                # ---- Actor ----
                mean = self.actor_net(states_mb)
                dist = self._dist(mean)
                curr_log_probs = dist.log_prob(actions_mb)

                ratios = torch.exp(curr_log_probs - old_logp_mb)
                unclipped_objective = ratios * advantages_mb
                clipped_ratio = torch.clamp(
                    ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip
                )
                clipped_objective = clipped_ratio * advantages_mb

                policy_objective = torch.min(unclipped_objective, clipped_objective)

                actor_loss = -policy_objective.mean()

                entropy = dist.entropy().mean()
                actor_loss = actor_loss - self.entropy_coef * entropy

                self.actor_net_optimiser.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor_net.parameters(), self.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_([self.log_std], self.max_grad_norm)
                self.actor_net_optimiser.step()

                # ---- Critic ----
                v = self.critic_net(states_mb).view(-1)
                critic_loss = F.mse_loss(v, returns_mb)

                self.critic_net_optimiser.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic_net.parameters(), self.max_grad_norm
                )
                self.critic_net_optimiser.step()

                # ---- Optional KL early stopping ----
                if self.target_kl is not None:
                    with torch.no_grad():
                        approx_kl = (old_logp_mb - curr_log_probs).mean()
                    if approx_kl > self.target_kl:
                        break

        info: dict[str, Any] = {}
        info["critic_loss"] = critic_loss.item()
        info["actor_loss"] = actor_loss.item()

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        checkpoint = {
            "actor": self.actor_net.state_dict(),
            "critic": self.critic_net.state_dict(),
            "log_std": self.log_std.data.detach().cpu(),
            "actor_optimizer": self.actor_net_optimiser.state_dict(),
            "critic_optimizer": self.critic_net_optimiser.state_dict(),
        }
        torch.save(checkpoint, f"{filepath}/{filename}_checkpoint.pth")
        logging.info("models and optimisers have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        checkpoint = torch.load(f"{filepath}/{filename}_checkpoint.pth")
        self.actor_net.load_state_dict(checkpoint["actor"])
        self.critic_net.load_state_dict(checkpoint["critic"])

        self.log_std.data.copy_(checkpoint["log_std"].to(self.device))

        self.actor_net_optimiser.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_net_optimiser.load_state_dict(checkpoint["critic_optimizer"])
        logging.info("models and optimisers have been loaded...")
