"""
LA3P (Loss-Adjusted Approximate Actor Prioritized Experience Replay)
----------------------------------------------------------------------

Original Paper: https://arxiv.org/abs/2209.00532

Original Code: https://github.com/h-yamani/RD-PER-baselines

LA3P adapts Prioritized Experience Replay (PER) to
off-policy actor-critic algorithms in continuous control.

Core Insight:
- Large TD-error transitions imply large Q-estimation error.
- Training the actor on high TD-error samples can cause
  the approximate policy gradient to diverge from the
  true gradient under the optimal Q-function.
- Therefore, actor and critic should not be trained on
  the same prioritized distribution.

Key Modifications to PER:

1) Inverse Sampling for the Actor
   - Critic: prioritized sampling (high TD error).
   - Actor: inverse-prioritized sampling (low TD error).
   - Ensures actor is trained on transitions where the
     critic has reliable estimates.

2) Shared Uniform Fraction (λ)
   - A fraction λ of each batch is sampled uniformly.
   - Both actor and critic update on these shared transitions.
   - Prevents instability from completely disjoint updates
     (actor-critic theory requirement).

3) Loss Corrections
   - Critic uses Huber loss (κ = 1) with LAP-style clipping:
         p_i = max(|δ_i|^α, 1)
   - Uniform portion uses PAL (Prioritized Approximate Loss),
     which matches the expected gradient of prioritized sampling.
   - Prevents outlier bias from MSE + PER interaction.

Training Structure (per update):
- λ·N transitions sampled uniformly:
      Critic updated with PAL
      Actor updated normally
- (1−λ)·N transitions:
      Critic updated with prioritized sampling (Huber)
      Actor updated with inverse-prioritized sampling
- Priorities updated after critic steps.

Rationale:
- PER works well for critics.
- PER harms actor learning in continuous control.
- LA3P decouples actor/critic sampling while preserving
  theoretical consistency.

LA3P = PER + Inverse Actor Sampling + Uniform Sharing + Loss Adjustment.
"""

from typing import Any

import numpy as np
import torch

import cares_reinforcement_learning.algorithm.lossess as loss
import cares_reinforcement_learning.memory.memory_sampler as memory_sampler
import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.algorithm.configurations import LA3PTD3Config
from cares_reinforcement_learning.algorithm.policy import TD3
from cares_reinforcement_learning.memory.memory_buffer import Sample, SARLMemoryBuffer
from cares_reinforcement_learning.networks.LA3PTD3 import Actor, Critic
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.experience import SingleAgentExperience


class LA3PTD3(TD3):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: LA3PTD3Config,
        device: torch.device,
    ):
        super().__init__(actor_network, critic_network, config, device)

        self.prioritized_fraction = config.prioritized_fraction

    def _update_target_network(self) -> None:
        # Update target network params
        self.soft_update_params(self.critic_net, self.target_critic_net, self.tau)
        self.soft_update_params(self.actor_net, self.target_actor_net, self.tau)

    # pylint: disable-next=arguments-differ, arguments-renamed
    def _update_critic(  # type: ignore[override]
        self,
        sample: Sample[SingleAgentExperience],
        uniform_sampling: bool,
    ) -> tuple[dict[str, Any], np.ndarray]:
        info: dict[str, Any] = {}

        # Convert into tensors using helper method
        (
            observation_tensor,
            actions_tensor,
            rewards_tensor,
            next_observation_tensor,
            dones_tensor,
            _,
            _,
        ) = memory_sampler.sample_to_tensors(sample, self.device)

        with torch.no_grad():
            next_actions = self.target_actor_net(
                next_observation_tensor.vector_state_tensor
            )

            target_noise = self.policy_noise * torch.randn_like(next_actions)
            target_noise = torch.clamp(
                target_noise, -self.policy_noise_clip, self.policy_noise_clip
            )
            next_actions = next_actions + target_noise
            next_actions = torch.clamp(next_actions, min=-1, max=1)

            target_q_values_one, target_q_values_two = self.target_critic_net(
                next_observation_tensor.vector_state_tensor, next_actions
            )

            target_q_values = torch.minimum(target_q_values_one, target_q_values_two)

            q_target = (
                rewards_tensor + self.gamma * (1 - dones_tensor) * target_q_values
            )

        q_values_one, q_values_two = self.critic_net(
            observation_tensor.vector_state_tensor, actions_tensor
        )

        td_error_one = (q_values_one - q_target).abs()
        td_error_two = (q_values_two - q_target).abs()

        if uniform_sampling:
            critic_loss_one = loss.prioritized_approximate_loss(
                td_error_one, self.min_priority, self.per_alpha
            )
            critic_loss_two = loss.prioritized_approximate_loss(
                td_error_two, self.min_priority, self.per_alpha
            )
            critic_loss_total = critic_loss_one + critic_loss_two

            critic_loss_total /= (
                torch.max(td_error_one, td_error_two)
                .clamp(min=self.min_priority)
                .pow(self.per_alpha)
                .mean()
                .detach()
            )
        else:
            critic_loss_one = loss.calculate_huber_loss(
                td_error_one, self.min_priority, use_quadratic_smoothing=False
            )
            critic_loss_two = loss.calculate_huber_loss(
                td_error_two, self.min_priority, use_quadratic_smoothing=False
            )
            critic_loss_total = critic_loss_one + critic_loss_two

        # Update the Critic
        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        priorities = (
            torch.max(td_error_one, td_error_two)
            .clamp(self.min_priority)
            .pow(self.per_alpha)
            .cpu()
            .data.numpy()
            .flatten()
        )

        with torch.no_grad():
            # --- TD3-style smoothing diagnostics ---
            # Noise diagnostics
            # What it tells you:
            # - target_noise_abs_mean: effective smoothing magnitude.
            # - target_noise_clip_frac high early: noise often clipped (clip too small or noise too large).
            target_noise_abs_mean = target_noise.abs().mean().item()
            target_noise_clip_frac = (
                (target_noise.abs() >= self.policy_noise_clip).float().mean().item()
            )
            info["target_noise_abs_mean"] = float(target_noise_abs_mean)
            info["target_noise_clip_frac"] = float(target_noise_clip_frac)

            # --- Twin critic disagreement (stability/uncertainty) ---
            # If this grows over training, critics are diverging / becoming inconsistent.
            info["q1_mean"] = q_values_one.mean().item()
            info["q2_mean"] = q_values_two.mean().item()
            info["q_twin_gap_abs_mean"] = (
                (q_values_one - q_values_two).abs().mean().item()
            )

            # --- Target critics disagreement (target stability) ---
            # Large/unstable gap here often means target critics are drifting or policy is visiting OOD actions.
            info["target_q1_mean"] = target_q_values_one.mean().item()
            info["target_q2_mean"] = target_q_values_two.mean().item()
            info["target_q_twin_gap_abs_mean"] = (
                (target_q_values_one - target_q_values_two).abs().mean().item()
            )

            # --- Bellman target scale (reward scaling / discount sanity) ---
            # If q_target drifts upward without reward improvement, suspect reward_scale, gamma, or instability.
            info["q_target_mean"] = q_target.mean().item()
            info["q_target_std"] = q_target.std().item()

            # --- TD error diagnostics (Bellman fit quality) ---
            # td_abs_mean down over time is healthy; persistent growth/spikes often indicate critic instability.
            td1 = q_values_one - q_target  # signed
            td2 = q_values_two - q_target  # signed

            info["td1_mean"] = td1.mean().item()
            info["td1_std"] = td1.std().item()
            info["td1_abs_mean"] = td1.abs().mean().item()

            info["td2_mean"] = td2.mean().item()
            info["td2_std"] = td2.std().item()
            info["td2_abs_mean"] = td2.abs().mean().item()

            # --- Losses (optimization progress; less diagnostic than TD/twin gaps) ---
            info["critic_loss_one"] = critic_loss_one.item()
            info["critic_loss_two"] = critic_loss_two.item()
            info["critic_loss_total"] = critic_loss_total.item()

        return info, priorities

    def train(
        self,
        memory_buffer: SARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:
        self.learn_counter += 1

        uniform_batch_size = int(self.batch_size * (1 - self.prioritized_fraction))
        priority_batch_size = int(self.batch_size * self.prioritized_fraction)

        policy_update = self.learn_counter % self.policy_update_freq == 0

        ######################### UNIFORM SAMPLING #########################
        uniform_buffer = memory_buffer.sample_uniform(uniform_batch_size)

        info_uniform: dict[str, Any] = {}

        critic_info, priorities = self._update_critic(
            uniform_buffer, uniform_sampling=True
        )
        info_uniform |= critic_info

        memory_buffer.update_priorities(np.asarray(uniform_buffer.indices), priorities)

        if policy_update:
            weights = np.array([1.0] * len(uniform_buffer.experiences))
            weights_tensor = torch.tensor(
                weights, dtype=torch.float32, device=self.device
            )
            observation_tensor = memory_sampler.observation_to_tensors(
                [experience.observation for experience in uniform_buffer.experiences],
                device=self.device,
            )

            actor_info = self._update_actor(
                observation_tensor.vector_state_tensor, weights_tensor
            )
            info_uniform |= actor_info

            self._update_target_network()

        ######################### CRITIC PRIORITIZED SAMPLING #########################
        priority_sample = memory_buffer.sample_priority(
            priority_batch_size,
            sampling_strategy=self.per_sampling_strategy,
            weight_normalisation=self.per_weight_normalisation,
        )

        info_priority: dict[str, Any] = {}

        critic_info, priorities = self._update_critic(
            priority_sample,
            uniform_sampling=False,
        )
        info_priority |= critic_info

        memory_buffer.update_priorities(np.asarray(priority_sample.indices), priorities)

        ######################### ACTOR PRIORITIZED SAMPLING #########################
        if policy_update:
            inverse_sample = memory_buffer.sample_inverse_priority(priority_batch_size)

            weights = np.array([1.0] * len(inverse_sample.experiences))
            weights_tensor = torch.tensor(
                weights, dtype=torch.float32, device=self.device
            )
            observation_tensor = memory_sampler.observation_to_tensors(
                [experience.observation for experience in inverse_sample.experiences],
                device=self.device,
            )

            actor_info = self._update_actor(
                observation_tensor.vector_state_tensor, weights_tensor
            )
            info_priority |= actor_info

            self._update_target_network()

        info = {"uniform": info_uniform, "priority": info_priority}
        return info
