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
from torch.distributions import Normal

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
        self.min_log_std = config.log_std_bounds[0]
        self.max_log_std = config.log_std_bounds[1]

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

    def _dist(self, mean: torch.Tensor) -> Normal:
        # mean: [B, act_dim] or [act_dim]
        std = self.log_std.exp()  # [act_dim]
        return Normal(mean, std)

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

            sample = mean if evaluation else dist.rsample()
            # store π_old(a|s) for PPO ratio
            log_prob = dist.log_prob(sample).sum(dim=-1)

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

        with torch.no_grad():
            info_adv_mean = advantages.mean()
            info_adv_std = advantages.std()
            info_returns_mean = returns.mean()
            info_returns_std = returns.std()

            # Critic fit quality (explained variance) on the whole batch
            v_pred_all = self.critic_net(states).view(-1)
            y_all = returns.view(-1)
            explained_var = 1.0 - torch.var(y_all - v_pred_all) / (
                torch.var(y_all) + 1e-8
            )

            # Exploration stats
            log_std_mean = self.log_std.mean()
            log_std_min = self.log_std.min()
            log_std_max = self.log_std.max()

        mb_size = min(self.minibatch_size, batch_size)

        actions_tensor = actions_tensor.view(-1, self.action_num)  # [N, act_dim]
        old_log_probs_tensor = old_log_probs_tensor.view(-1)  # [N]
        advantages = advantages.view(-1)  # [N]
        returns = returns.view(-1)  # [N]

        kl_early_stop = False
        for _ in range(self.updates_per_iteration):
            idx = torch.randperm(batch_size, device=self.device)

            sum_kl = 0.0
            sum_entropy = 0.0
            sum_clip_frac = 0.0
            sum_ratio_mean = 0.0
            sum_ratio_std = 0.0
            num_mbs = 0

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
                curr_log_probs = dist.log_prob(actions_mb).sum(dim=-1)

                assert curr_log_probs.shape == old_logp_mb.shape

                # ---- Optional KL early stopping ----
                if self.target_kl is not None:
                    with torch.no_grad():
                        approx_kl = (old_logp_mb - curr_log_probs).mean()
                        sum_kl += float(approx_kl.item())
                    if approx_kl > self.target_kl:
                        kl_early_stop = True
                        break

                num_mbs += 1

                ratios = torch.exp(curr_log_probs - old_logp_mb)

                unclipped_objective = ratios * advantages_mb
                clipped_ratio = torch.clamp(
                    ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip
                )
                clipped_objective = clipped_ratio * advantages_mb

                policy_objective = torch.min(unclipped_objective, clipped_objective)

                actor_loss = -policy_objective.mean()

                entropy = dist.entropy().sum(dim=-1).mean()
                actor_loss = actor_loss - self.entropy_coef * entropy

                self.actor_net_optimiser.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor_net.parameters()) + [self.log_std],
                    self.max_grad_norm,
                )
                self.actor_net_optimiser.step()

                with torch.no_grad():
                    self.log_std.clamp_(self.min_log_std, self.max_log_std)

                # ---- Critic ----
                v = self.critic_net(states_mb).view(-1)
                critic_loss = F.mse_loss(v, returns_mb)

                self.critic_net_optimiser.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic_net.parameters(), self.max_grad_norm
                )
                self.critic_net_optimiser.step()

                with torch.no_grad():
                    clip_frac = (
                        (
                            (ratios > 1.0 + self.eps_clip)
                            | (ratios < 1.0 - self.eps_clip)
                        )
                        .float()
                        .mean()
                    )
                    sum_clip_frac += float(clip_frac.item())
                    sum_ratio_mean += float(ratios.mean().item())
                    sum_ratio_std += float(ratios.std(unbiased=False).item())
                    sum_entropy += float(entropy.item())

            if kl_early_stop:
                break

        info: dict[str, Any] = {}
        info["critic_loss"] = float(critic_loss.item())
        info["actor_loss"] = float(actor_loss.item())

        # Batch-level stats
        info["adv_mean"] = float(info_adv_mean.item())
        info["adv_std"] = float(info_adv_std.item())
        info["returns_mean"] = float(info_returns_mean.item())
        info["returns_std"] = float(info_returns_std.item())
        info["explained_variance"] = float(explained_var.item())

        # Exploration
        info["log_std_mean"] = float(log_std_mean.item())
        info["log_std_min"] = float(log_std_min.item())
        info["log_std_max"] = float(log_std_max.item())

        # Update health (averaged over minibatches)
        if num_mbs > 0:
            info["entropy"] = sum_entropy / num_mbs
            info["clip_frac"] = sum_clip_frac / num_mbs
            info["ratio_mean"] = sum_ratio_mean / num_mbs
            info["ratio_std"] = sum_ratio_std / num_mbs

        # KL (only if enabled)
        if self.target_kl is not None and num_mbs > 0:
            info["approx_kl"] = sum_kl / num_mbs

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
