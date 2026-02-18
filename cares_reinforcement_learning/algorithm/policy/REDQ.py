"""
REDQ (Randomized Ensembled Double Q-Learning)
----------------------------------------------

Original Paper: https://arxiv.org/pdf/2101.05982.pdf

REDQ is an off-policy actor-critic algorithm designed to
improve sample efficiency in continuous control by using
a large ensemble of Q-networks.

Core Problem:
- Standard SAC / TD3 use two critics (clipped double Q).
- Larger ensembles improve bias reduction but increase cost.
- Frequent updates improve sample efficiency but can
  amplify overestimation bias.

Core Idea:
- Maintain N Q-networks (N >> 2).
- For each target computation, randomly select M critics
  (M < N) and take the minimum over the subset.
- Perform multiple gradient updates per environment step.

Critic Target:
    Sample M critics from N
    y = r + γ ( min_j Q_target_j(s', a') - α log π(a'|s') )

This stochastic subset minimization:
- Reduces overestimation bias
- Preserves diversity across critics
- Avoids always taking the global minimum

Actor Update:
- Same as SAC:
      maximize E[ min_j Q_j(s, π(s)) - α log π ]
- Typically uses full ensemble mean or subset min.

Update-to-Data Ratio (UTD):
- REDQ increases the number of gradient steps per
  environment interaction.
- High UTD improves sample efficiency without
  requiring model-based components.

Key Behaviour:
- Ensemble size N improves bias reduction.
- Subsample size M controls conservativeness.
- High UTD enables fast learning from limited data.

Advantages:
- Strong sample efficiency in continuous control.
- Simple extension of SAC.
- No model learning required.

REDQ = SAC + Large Q-ensemble + Randomized subset
        minimization + High update-to-data ratio.
"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.memory.memory_sampler as memory_sampler
import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.algorithm.policy import SAC
from cares_reinforcement_learning.memory.memory_buffer import SARLMemoryBuffer
from cares_reinforcement_learning.networks.REDQ import Actor, Critic
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import SARLObservation
from cares_reinforcement_learning.util.configurations import REDQConfig


class REDQ(SAC):
    critic_net: Critic
    target_critic_net: Critic

    def __init__(
        self,
        actor_network: Actor,
        ensemble_critic: Critic,
        config: REDQConfig,
        device: torch.device,
    ):
        super().__init__(
            actor_network=actor_network,
            critic_network=ensemble_critic,
            config=config,
            device=device,
        )

        self.num_sample_critics = config.num_sample_critics
        self.ensemble_size = config.ensemble_size

        self.lr_ensemble_critic = config.critic_lr
        self.ensemble_critic_optimizers = [
            torch.optim.Adam(
                critic_net.parameters(),
                lr=self.lr_ensemble_critic,
                **config.critic_lr_params,
            )
            for critic_net in self.critic_net.critics
        ]

    def _calculate_value(self, state: SARLObservation, action: np.ndarray) -> float:  # type: ignore[override]
        state_tensor = torch.FloatTensor(state.vector_state).to(self.device)
        state_tensor = state_tensor.unsqueeze(0)

        action_tensor = torch.FloatTensor(action).to(self.device)
        action_tensor = action_tensor.unsqueeze(0)

        with torch.no_grad():
            with hlp.evaluating(self.critic_net):
                q_values = self.critic_net(state_tensor, action_tensor)
                q_value = q_values.mean()

        return q_value.item()

    # pylint: disable-next=arguments-differ, arguments-renamed
    def _update_critic(  # type: ignore[override]
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {}
        # replace=False so that not picking the same idx twice
        idx = np.random.choice(
            self.ensemble_size, self.num_sample_critics, replace=False
        )

        with torch.no_grad():
            with hlp.evaluating(self.actor_net):
                next_actions, next_log_pi, _ = self.actor_net(next_states)

            target_q_values_one = self.target_critic_net.critics[idx[0]](
                next_states, next_actions
            )

            target_q_values_two = self.target_critic_net.critics[idx[1]](
                next_states, next_actions
            )

            target_q_values = (
                torch.minimum(target_q_values_one, target_q_values_two)
                - self.alpha * next_log_pi
            )

            q_target = rewards + self.gamma * (1 - dones) * target_q_values

        critic_loss_totals: list[float] = []
        critic_td_abs_means: list[float] = []
        critic_q_means: list[float] = []

        # For ensemble diagnostics (store per-critic outputs on this batch)
        q_set: list[torch.Tensor] = []

        for critic_net, critic_net_optimiser in zip(
            self.critic_net.critics, self.ensemble_critic_optimizers
        ):
            q_values = critic_net(states, actions)

            q_set.append(q_values)

            td = q_values - q_target  # signed TD error
            critic_td_abs_means.append(td.abs().mean().item())
            critic_q_means.append(q_values.mean().item())

            critic_loss = 0.5 * F.mse_loss(q_values, q_target)

            critic_net_optimiser.zero_grad()
            critic_loss.backward()
            critic_net_optimiser.step()

            critic_loss_totals.append(critic_loss.item())

        with torch.no_grad():
            # Which target critics were sampled (for debugging + reproducibility)
            info["idx0"] = int(idx[0])
            info["idx1"] = int(idx[1])

            # --- Target-side diagnostics (s', pi(s')) ---
            info["target_q1_mean"] = target_q_values_one.mean().item()
            info["target_q2_mean"] = target_q_values_two.mean().item()
            info["target_min_q_mean"] = target_q_values.mean().item()

            # Disagreement between the sampled target critics
            target_gap = (target_q_values_one - target_q_values_two).abs()
            info["target_q_gap_abs_mean"] = target_gap.mean().item()
            info["target_q_gap_abs_p95"] = target_gap.quantile(0.95).item()

            # --- Soft target decomposition (SAC-specific) ---
            # min_target_q_mean: the conservative bootstrap value from twin critics (pre-entropy)
            # entropy_term_mean: magnitude of entropy regularization in the target (alpha * log_pi is usually negative)
            # soft_target_value_mean: the exact term used inside the Bellman target before reward/discount
            min_target_q = torch.minimum(target_q_values_one, target_q_values_two)

            # alpha_log_pi is typically negative; entropy_bonus is typically positive
            alpha_log_pi = self.alpha * next_log_pi
            # this is what gets ADDED to minQ in the target
            entropy_bonus = -self.alpha * next_log_pi

            soft_target_value = min_target_q + entropy_bonus  # == minQ - alpha*log_pi

            info["target_min_q_mean"] = min_target_q.mean().item()
            info["alpha_log_pi_mean"] = alpha_log_pi.mean().item()
            info["entropy_bonus_mean"] = entropy_bonus.mean().item()
            info["soft_target_value_mean"] = soft_target_value.mean().item()

            # Bellman target scale
            info["q_target_mean"] = q_target.mean().item()
            info["q_target_std"] = q_target.std(unbiased=False).item()

            # --- Critic loss diagnostics ---
            info["critic_loss_total"] = float(np.mean(critic_loss_totals))
            info["critic_loss_totals"] = critic_loss_totals  # per-critic scalars

            # “Bad apple” detection across critics
            info["critic_loss_std_across_critics"] = float(np.std(critic_loss_totals))
            info["critic_td_abs_mean_across_critics"] = float(
                np.mean(critic_td_abs_means)
            )
            info["critic_td_abs_std_across_critics"] = float(
                np.std(critic_td_abs_means)
            )
            info["critic_q_mean_across_critics"] = float(np.mean(critic_q_means))
            info["critic_q_std_across_critics"] = float(np.std(critic_q_means))

            # --- Ensemble disagreement on replay batch (s,a) ---
            q_mat = torch.cat(q_set, dim=1)  # (B,E)
            info["current_ensemble_q_mean"] = q_mat.mean().item()
            info["current_ensemble_q_std_mean"] = (
                q_mat.std(dim=1, unbiased=False).mean().item()
            )
            # If this grows: see critic divergence / epistemic spread.

            # TD tail risk (more sensitive than mean)
            td_mat = q_mat - q_target  # broadcast (B,E) - (B,1)
            td_abs_max = td_mat.abs().max(dim=1).values  # (B,)
            info["td_abs_max_mean"] = td_abs_max.mean().item()
            info["td_abs_max_p95"] = td_abs_max.quantile(0.95).item()
            info["td_abs_max_max"] = td_abs_max.max().item()

        return info

    # pylint: disable-next=arguments-differ, arguments-renamed
    def _update_actor_alpha(  # type: ignore[override]
        self,
        states: torch.Tensor,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {}

        pi, log_pi, _ = self.actor_net(states)

        with hlp.evaluating(self.critic_net):
            q_values = self.critic_net(states, pi)
            q_mean = q_values.mean(dim=1)

        actor_loss = ((self.alpha * log_pi) - q_mean).mean()

        # ---------------------------------------------------------
        # Stochastic Policy Gradient Strength (∇a [α log π(a|s) − Q(s,a)])
        # ---------------------------------------------------------
        # Measures how steep the entropy-regularized critic objective is
        # w.r.t. the sampled policy actions.
        #
        # ~0 early  -> critic surface and entropy term nearly flat;
        #              actor receives weak learning signal.
        #
        # Very large -> critic or entropy term is very sharp around policy
        #               actions; can lead to unstable or overly aggressive
        #               actor updates.
        dq_da = torch.autograd.grad(
            outputs=actor_loss,
            inputs=pi,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]

        with torch.no_grad():
            info["dq_da_abs_mean"] = dq_da.abs().mean().item()
            info["dq_da_norm_mean"] = dq_da.norm(dim=1).mean().item()
            info["dq_da_norm_p95"] = dq_da.norm(dim=1).quantile(0.95).item()

        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        # update the temperature
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        with torch.no_grad():
            # --- Policy entropy diagnostics (exploration health) ---
            # log_pi more negative -> higher entropy (more stochastic). Less negative -> lower entropy (more deterministic).
            info["log_pi_mean"] = log_pi.mean().item()
            info["log_pi_std"] = log_pi.std().item()

            # --- Action magnitude/saturation (tanh policies) ---
            # High saturation fraction can indicate the policy is slamming bounds; may reduce effective gradients.
            info["pi_action_abs_mean"] = pi.abs().mean().item()
            info["pi_action_std"] = pi.std().item()
            info["pi_action_saturation_frac"] = (pi.abs() > 0.95).float().mean().item()

            # --- On-policy critic signal (REDQ uses ensemble mean) ---
            # REDQ actor uses mean over ensemble as value signal
            info["q_pi_mean"] = q_mean.mean().item()
            info["q_pi_std"] = q_mean.std(unbiased=False).item()

            # --- Ensemble disagreement at policy actions (REDQ analogue of twin-gap) ---
            # If this grows, critics disagree on current policy behaviour (instability / epistemic spread).
            q_std_across_critics = q_values.std(dim=1, unbiased=False)  # (B,)
            info["q_pi_ensemble_std_mean"] = q_std_across_critics.mean().item()
            info["q_pi_ensemble_std_p95"] = q_std_across_critics.quantile(0.95).item()

            # You can also track dominance extremes if you ever use weighted fusion later
            info["q_pi_ensemble_min_mean"] = q_values.min(dim=1).values.mean().item()
            info["q_pi_ensemble_max_mean"] = q_values.max(dim=1).values.mean().item()

            # --- Entropy gap (alpha tuning health) ---
            # entropy_gap ~ 0 means entropy matches target.
            # > 0: entropy too low -> alpha should increase; < 0: entropy too high -> alpha should decrease.
            entropy_gap = -(log_pi + self.target_entropy)
            info["entropy_gap_mean"] = entropy_gap.mean().item()

            # --- Losses and temperature ---
            info["actor_loss"] = actor_loss.item()
            info["alpha_loss"] = alpha_loss.item()
            info["alpha"] = self.alpha.item()
            info["log_alpha"] = self.log_alpha.item()

        return info

    def train(
        self,
        memory_buffer: SARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:
        self.learn_counter += 1

        (
            observation_tensor,
            actions_tensor,
            rewards_tensor,
            next_observation_tensor,
            dones_tensor,
            _,
            _,
            _,
        ) = memory_sampler.sample(
            memory=memory_buffer,
            batch_size=self.batch_size,
            device=self.device,
            use_per_buffer=0,  # REDQ uses uniform sampling
        )

        info: dict[str, Any] = {}

        # Update the Critics
        critic_info = self._update_critic(
            observation_tensor.vector_state_tensor,
            actions_tensor,
            rewards_tensor,
            next_observation_tensor.vector_state_tensor,
            dones_tensor,
        )
        info |= critic_info

        if self.learn_counter % self.policy_update_freq == 0:
            # Update the Actor
            actor_info = self._update_actor_alpha(
                observation_tensor.vector_state_tensor
            )
            info |= actor_info
            info["alpha"] = self.alpha.item()

        if self.learn_counter % self.target_update_freq == 0:
            # Update ensemble of target critics
            for critic_net, target_critic_net in zip(
                self.critic_net.critics, self.target_critic_net.critics
            ):
                hlp.soft_update_params(critic_net, target_critic_net, self.tau)

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        super().save_models(filepath, filename)
        # Save each ensemble critic optimizer in a single file
        ensemble_optim_state = {
            f"optimizer_{idx}": opt.state_dict()
            for idx, opt in enumerate(self.ensemble_critic_optimizers)
        }
        torch.save(
            ensemble_optim_state,
            f"{filepath}/{filename}_ensemble_critic_optimizers.pth",
        )

    def load_models(self, filepath: str, filename: str) -> None:
        super().load_models(filepath, filename)
        # Load each ensemble critic optimizer from the single file
        ensemble_optim_state = torch.load(
            f"{filepath}/{filename}_ensemble_critic_optimizers.pth"
        )
        for idx, opt in enumerate(self.ensemble_critic_optimizers):
            opt.load_state_dict(ensemble_optim_state[f"optimizer_{idx}"])
