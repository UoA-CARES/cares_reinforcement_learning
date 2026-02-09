"""
PPO (Proximal Policy Optimization) implementation notes
--------------------------------------------------------------
Original Paper: https://arxiv.org/abs/1707.06347

This implementation follows the clipped-surrogate PPO formulation with
Generalized Advantage Estimation (GAE), minibatch SGD, and optional KL-based
early stopping, adapted for bounded continuous control.

Policy parameterization (bounded actions):
- The actor represents a Gaussian policy in *pre-squash* space:
    u ~ Normal(mean(s), std)
- Actions are obtained via a tanh squashing function:
    a = tanh(u), ensuring actions lie in [-1, 1].
- Log-probabilities are computed with the correct change-of-variables
  correction for the tanh transformation.
- The rollout buffer stores the *squashed actions* a, along with their
  corrected log-probabilities and critic values.
- During training, pre-squash actions are reconstructed via atanh(a).

Rollout collection:
- Experience is collected strictly on-policy using the current stochastic policy.
- For each step, the following are stored:
    - the squashed action a in [-1, 1],
    - the corrected log-probability log π(a | s),
    - the critic value estimate V(s).
- No environment-side action clipping is relied upon; boundedness is enforced
  directly by the policy.

Advantage estimation:
- Advantages are computed using Generalized Advantage Estimation (GAE):
    δ_t = r_t + γ (1 - done_t) V(s_{t+1}) - V(s_t)
- A single bootstrap value V(s_T) is used for truncated rollouts.
- If the final transition in the rollout is terminal, the bootstrap value
  is masked out (set to zero).
- Returns for critic updates are computed as:
    return_t = advantage_t + V(s_t)
- Advantages are normalized across the batch for numerical stability.

Policy and value updates:
- The actor is optimized using the PPO clipped surrogate objective.
- An optional entropy bonus (computed from the base Gaussian) encourages
  exploration.
- The critic is trained by regression onto the computed returns.
- Updates are performed using multiple epochs of minibatch SGD over the same
  on-policy rollout.
- Gradient norm clipping is applied to both actor (including log_std) and
  critic parameters.

Exploration noise (learnable log_std):
- The policy maintains a learnable log standard deviation parameter (log_std)
  for each action dimension.
- This parameter controls the scale of exploration noise in pre-squash space
  and is optimized jointly with the actor network parameters.
- log_std is constrained to lie within configurable bounds to prevent excessive
  exploration or premature collapse to near-deterministic behavior.
- During action sampling, the Gaussian standard deviation is obtained via
  exp(log_std), and gradients are propagated through this parameter.
- After each policy update, log_std is projected back into its valid range
  to ensure numerical stability and consistent exploration behavior.

KL control:
- An approximate KL divergence between the old and updated policy is monitored
  using a PPO-style second-order approximation.
- If the KL exceeds a configured threshold, further minibatch and epoch updates
  for the current rollout are stopped early, providing an additional trust-region
  constraint beyond ratio clipping.

Notes:
- This implementation assumes all actions are normalized to [-1, 1] and that
  the environment wrapper preserves this convention.
- The tanh-squashed Gaussian formulation ensures consistency between the
  executed actions and the likelihoods used for policy optimization, which is
  particularly important for bounded robotics control tasks.
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

        # Initialize log_std as a learnable parameter, starting at max_log_std for high initial exploration
        init_log_std = torch.full(
            (self.action_num,), self.max_log_std, device=self.device
        )
        self.log_std = torch.nn.Parameter(init_log_std)

        self.actor_net_optimiser = torch.optim.Adam(
            list(self.actor_net.parameters()) + [self.log_std], lr=config.actor_lr
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=config.critic_lr
        )

        self.updates_per_iteration = config.updates_per_iteration
        self.eps_clip = config.eps_clip

    def _dist(self, mean: torch.Tensor) -> Normal:
        log_std = self.log_std.clamp(self.min_log_std, self.max_log_std)
        # mean: [B, act_dim] or [act_dim]
        std = log_std.exp()  # [act_dim]
        return Normal(mean, std)

    def _atanh(self, a: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # stable inverse tanh
        a = torch.clamp(a, -1.0 + eps, 1.0 - eps)
        return 0.5 * (torch.log1p(a) - torch.log1p(-a))

    def _squash(self, u: torch.Tensor) -> torch.Tensor:
        return torch.tanh(u)

    def _squashed_log_prob(
        self, dist: Normal, u: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Computes log π(a|s) where u ~ Normal(mean, std) and a = tanh(u).
        Returns shape [B]
        """
        a = torch.tanh(u)
        logp_u = dist.log_prob(u).sum(dim=-1)
        # change-of-variables: log |det da/du| = sum log(1 - tanh(u)^2)
        log_det = torch.log(1.0 - a * a + eps).sum(dim=-1)
        return logp_u - log_det

    def act(
        self,
        observation: SARLObservation,
        evaluation: bool = False,
        calculate_value: bool = True,
    ) -> ActionSample[np.ndarray]:
        self.actor_net.eval()
        self.critic_net.eval()
        state = observation.vector_state

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            state_tensor = state_tensor.unsqueeze(0)  # add batch dimension

            mean = self.actor_net(state_tensor)
            dist = self._dist(mean)

            u = mean if evaluation else dist.rsample()

            action_t = self._squash(u)  # in [-1, 1]
            log_prob = self._squashed_log_prob(dist, u)  # consistent log π(a|s)

            value = (
                self.critic_net(state_tensor).squeeze(-1)
                if calculate_value
                else torch.tensor(0.0, device=self.device)
            )

            action = action_t.squeeze(0).cpu().numpy()

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

        gae = torch.zeros((), dtype=torch.float32, device=self.device)
        next_value = last_value

        for t in reversed(range(batch_size)):
            mask = 1.0 - dones[t]  # 0 if terminal else 1
            delta = rewards[t] + self.gamma * mask * next_value - values[t]
            gae = delta + self.gamma * gae_lambda * mask * gae
            advantages[t] = gae
            next_value = values[t]

        returns = advantages + values
        return advantages, returns

    def update_actor_minibatch(
        self,
        states_mb: torch.Tensor,  # [mb, obs_dim]
        actions_mb: torch.Tensor,  # [mb, act_dim] (squashed)
        old_logp_mb: torch.Tensor,  # [mb]
        advantages_mb: torch.Tensor,  # [mb]
    ) -> tuple[bool, dict[str, float]]:

        mean = self.actor_net(states_mb)
        dist = self._dist(mean)
        u_mb = self._atanh(actions_mb)  # invert tanh
        curr_log_probs = self._squashed_log_prob(dist, u_mb)

        log_ratio = curr_log_probs - old_logp_mb
        ratios = torch.exp(curr_log_probs - old_logp_mb)

        # ---- Optional KL early stopping ----
        if self.target_kl is not None:
            with torch.no_grad():
                approx_kl = (ratios - 1 - log_ratio).mean()
                if approx_kl > self.target_kl:
                    return True, {"approx_kl": float(approx_kl.item())}

        unclipped_objective = ratios * advantages_mb
        clipped_ratio = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip)
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

        # ---- Debug stats: saturation, pre-tanh magnitude, log-ratio stats ----
        with torch.no_grad():
            # action saturation rate
            clip_frac = (
                ((ratios > 1.0 + self.eps_clip) | (ratios < 1.0 - self.eps_clip))
                .float()
                .mean()
            )
            sat_rate = (actions_mb.abs() > 0.99).float().mean()
            u_abs_mean = u_mb.abs().mean()
            u_abs_max = u_mb.abs().max()

            log_ratio_mean = log_ratio.mean()
            log_ratio_std = log_ratio.std(unbiased=False)
            log_ratio_max_abs = log_ratio.abs().max()

        info = {
            "actor_loss": float(actor_loss.item()),
            "entropy": float(entropy.item()),
            "approx_kl": float(approx_kl.item()) if self.target_kl is not None else 0.0,
            "clip_frac": float(clip_frac.item()),
            "ratio_mean": float(ratios.mean().item()),
            "ratio_std": float(ratios.std(unbiased=False).item()),
            "action_sat_rate": float(sat_rate.item()),
            "u_abs_mean": float(u_abs_mean.item()),
            "u_abs_max": float(u_abs_max.item()),
            "log_ratio_mean": float(log_ratio_mean.item()),
            "log_ratio_std": float(log_ratio_std.item()),
            "log_ratio_max_abs": float(log_ratio_max_abs.item()),
        }

        return False, info

    def update_critic_minibatch(
        self,
        states_mb: torch.Tensor,  # [mb, obs_dim]
        returns_mb: torch.Tensor,  # [mb]
    ) -> dict[str, float]:

        v = self.critic_net(states_mb).view(-1)
        critic_loss = F.mse_loss(v, returns_mb)

        self.critic_net_optimiser.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
        self.critic_net_optimiser.step()

        return {"critic_loss": float(critic_loss.item())}

    def train_policy(
        self,
        memory_buffer: SARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:
        # pylint: disable-next=unused-argument

        sample = memory_buffer.flush()
        batch_size = len(sample.experiences)

        if batch_size == 0:
            return {}

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

        # Compute last value for GAE bootstrap (V(s_T) for truncated rollouts, 0 for terminal)
        # Boot Strap next_value for the final step in the buffer - zero out if it is a terminal state,
        # otherwise use critic estimate for V(s_T)
        with torch.no_grad():
            last_next_state = next_observation_tensor.vector_state_tensor[-1].unsqueeze(
                0
            )  # [1, obs_dim]
            last_value = self.critic_net(last_next_state).view(-1)[0]  # scalar
            last_done = dones_tensor.reshape(-1)[-1].bool()  # True if terminal
            last_value = last_value * (~last_done).to(last_value.dtype)

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
            log_std_clamped = self.log_std.clamp(self.min_log_std, self.max_log_std)
            log_std_mean = log_std_clamped.mean()
            log_std_min = log_std_clamped.min()
            log_std_max = log_std_clamped.max()

        mb_size = min(self.minibatch_size, batch_size)

        actions_tensor = actions_tensor.view(-1, self.action_num)  # [N, act_dim]
        old_log_probs_tensor = old_log_probs_tensor.view(-1)  # [N]
        advantages = advantages.view(-1)  # [N]
        returns = returns.view(-1)  # [N]

        kl_early_stop = False
        # ---- Debug accumulators (across all minibatches/epochs) ----
        sum_sat_rate = 0.0
        sum_u_abs_mean = 0.0
        sum_u_abs_max = 0.0

        sum_log_ratio_mean = 0.0
        sum_log_ratio_std = 0.0
        sum_log_ratio_max_abs = 0.0

        sum_kl = 0.0
        max_kl_seen = 0.0
        sum_entropy = 0.0
        sum_clip_frac = 0.0
        sum_ratio_mean = 0.0
        sum_ratio_std = 0.0

        sum_critic_loss = 0.0
        sum_actor_loss = 0.0

        num_mbs = 0

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
                kl_early_stop, actor_info = self.update_actor_minibatch(
                    states_mb, actions_mb, old_logp_mb, advantages_mb
                )

                if kl_early_stop:
                    break

                # ---- Debug stats: saturation, pre-tanh magnitude, log-ratio stats ----
                sum_sat_rate += actor_info["action_sat_rate"]
                sum_u_abs_mean += actor_info["u_abs_mean"]
                sum_u_abs_max += actor_info["u_abs_max"]

                sum_log_ratio_mean += actor_info["log_ratio_mean"]
                sum_log_ratio_std += actor_info["log_ratio_std"]
                sum_log_ratio_max_abs += actor_info["log_ratio_max_abs"]

                sum_clip_frac += actor_info["clip_frac"]
                sum_ratio_mean += actor_info["ratio_mean"]
                sum_ratio_std += actor_info["ratio_std"]
                sum_entropy += actor_info["entropy"]

                sum_actor_loss += actor_info["actor_loss"]

                num_mbs += 1

                # ---- Critic ----
                critic_info = self.update_critic_minibatch(states_mb, returns_mb)
                sum_critic_loss += critic_info["critic_loss"]

            if kl_early_stop:
                break

        info: dict[str, Any] = {}
        info["critic_loss"] = sum_critic_loss / num_mbs if num_mbs > 0 else 0.0
        info["actor_loss"] = sum_actor_loss / num_mbs if num_mbs > 0 else 0.0

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

            info["action_sat_rate"] = sum_sat_rate / num_mbs
            info["u_abs_mean"] = sum_u_abs_mean / num_mbs
            info["u_abs_max"] = sum_u_abs_max / num_mbs

            info["log_ratio_mean"] = sum_log_ratio_mean / num_mbs
            info["log_ratio_std"] = sum_log_ratio_std / num_mbs
            info["log_ratio_max_abs"] = sum_log_ratio_max_abs / num_mbs

        # KL (only if enabled)
        if self.target_kl is not None and num_mbs > 0:
            info["approx_kl"] = sum_kl / num_mbs
            info["kl_early_stop"] = int(kl_early_stop)
            info["max_kl_seen"] = max_kl_seen

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
