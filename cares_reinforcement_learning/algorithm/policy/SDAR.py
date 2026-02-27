"""
SDAR (Spatially Decoupled Action Repetition)
---------------------------------------------

Original Paper: https://openreview.net/pdf?id=PDgZ3rvqHn

SDAR is a closed-loop action repetition framework for
continuous control that performs act-or-repeat selection
independently for each action dimension.

Core Problem:
- Standard RL selects actions at every timestep.
- Existing repetition methods treat the entire action vector
  as a whole when deciding to act or repeat.
- Different actuators often require different repetition
  frequencies, making joint repetition inflexible.

Core Idea:
- Decouple repetition decisions across action dimensions.
- For each dimension i, decide:
      repeat previous action  (b_i = 0)
      or generate new action  (b_i = 1)

Two-Stage Policy:

1) Selection Policy β(b | s, a_prev)
   - Outputs Bernoulli probabilities per dimension.
   - Produces repetition mask:
         b ∈ {0,1}^{|A|}

2) Action Policy π(â | s, a_prev, b)
   - Generates new actions only where b_i = 1.
   - Final action is mixed as:
         a = (1 - b) ⊙ a_prev + b ⊙ â

This guarantees exact repetition where selected.

Learning:
- Off-policy training with replay buffer.
- Twin Q-functions (clipped double-Q).
- Joint entropy-regularized objective for:
      • selection policy β
      • action policy π
- Separate temperature terms for β and π.

Key Behaviour:
- Higher action persistence without sacrificing agility.
- Improved balance between repetition and diversity.
- Reduced action fluctuation.
- Higher sample efficiency vs SAC, open-loop,
  and prior closed-loop repetition methods.

SDAR = closed-loop, per-dimension action repetition
        with joint entropy-regularized training.
"""

import copy
import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.memory.memory_sampler as memory_sampler
import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.algorithm.algorithm import SARLAlgorithm
from cares_reinforcement_learning.memory.memory_buffer import SARLMemoryBuffer
from cares_reinforcement_learning.networks.SDAR import Actor, Critic
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import SARLObservation
from cares_reinforcement_learning.algorithm.configurations import SDARConfig


class SDAR(SARLAlgorithm[np.ndarray]):
    actor_network: Actor
    critic_network: Critic

    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: SDARConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        # SAC-style initialization
        self.gamma = config.gamma
        self.tau = config.tau
        self.reward_scale = config.reward_scale

        # PER
        self.use_per_buffer = config.use_per_buffer
        self.per_sampling_strategy = config.per_sampling_strategy
        self.per_weight_normalisation = config.per_weight_normalisation
        self.per_alpha = config.per_alpha
        self.min_priority = config.min_priority

        self.learn_counter = 0
        self.policy_update_freq = config.policy_update_freq
        self.target_update_freq = config.target_update_freq

        # Networks
        self.actor_net = actor_network.to(self.device)
        self.critic_net = critic_network.to(self.device)
        self.target_critic_net = copy.deepcopy(self.critic_net).to(self.device)
        self.target_critic_net.eval()

        self.target_entropy = -self.actor_net.num_actions

        # Optimizers
        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=config.actor_lr, **config.actor_lr_params
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=config.critic_lr, **config.critic_lr_params
        )

        # Alpha (entropy regularization)
        alpha_init_temperature = 1.0
        self.log_alpha = torch.tensor(
            np.log(alpha_init_temperature), dtype=torch.float32, device=device
        )
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=config.alpha_lr, **config.alpha_lr_params
        )

        # SDAR-specific initialization
        self.prev_action_tensor = torch.zeros(
            (1, self.actor_net.num_actions), device=self.device
        )

        self.force_act = True

        self.target_beta = -0.5 * self.actor_net.num_actions

        # Beta (action regularization specific to SDAR)
        beta_init_temperature = 1.0
        self.log_beta = torch.tensor(
            np.log(beta_init_temperature), dtype=torch.float32, device=device
        )
        self.log_beta.requires_grad = True
        self.log_beta_optimizer = torch.optim.Adam(
            [self.log_beta], lr=config.beta_lr, **config.beta_lr_params
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @property
    def beta(self) -> torch.Tensor:
        return self.log_beta.exp()

    def episode_done(self):
        # Reset the previous action to the dummy action
        self.prev_action_tensor = torch.zeros(
            (1, self.actor_net.num_actions), device=self.device
        )
        self.force_act = True

    def act(
        self, observation: SARLObservation, evaluation: bool = False
    ) -> ActionSample[np.ndarray]:
        # note that when evaluating this algorithm we need to select mu as action
        self.actor_net.eval()

        state = observation.vector_state

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            state_tensor = state_tensor.unsqueeze(0)
            if evaluation:
                _, _, action, *_ = self.actor_net(
                    state_tensor, self.prev_action_tensor, force_act=self.force_act
                )
            else:
                action, _, *_ = self.actor_net(
                    state_tensor, self.prev_action_tensor, force_act=self.force_act
                )

            self.prev_action_tensor = action
            self.force_act = False

            action = action.cpu().data.numpy().flatten()
        self.actor_net.train()

        return ActionSample(action=action, source="policy")

    # pylint: disable-next=arguments-differ, arguments-renamed
    def _update_critic(  # type: ignore[override]
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[dict[str, Any], np.ndarray]:
        info: dict[str, Any] = {}
        with torch.no_grad():
            with hlp.evaluating(self.actor_net):
                next_actions, next_log_pi, *_ = self.actor_net(
                    next_states, actions, force_act=False
                )

            target_q_values_one, target_q_values_two = self.target_critic_net(
                next_states, next_actions
            )
            target_q_values = (
                torch.minimum(target_q_values_one, target_q_values_two)
                - self.alpha * next_log_pi
            )

            q_target = (
                rewards * self.reward_scale + self.gamma * (1 - dones) * target_q_values
            )

        q_values_one, q_values_two = self.critic_net(states, actions)

        td_error_one = (q_values_one - q_target).abs()
        td_error_two = (q_values_two - q_target).abs()

        critic_loss_one = F.mse_loss(q_values_one, q_target, reduction="none")
        critic_loss_one = (critic_loss_one * weights).mean()

        critic_loss_two = F.mse_loss(q_values_two, q_target, reduction="none")
        critic_loss_two = (critic_loss_two * weights).mean()

        critic_loss_total = critic_loss_one + critic_loss_two

        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        # Update the Priorities - PER only
        priorities = (
            torch.max(td_error_one, td_error_two)
            .clamp(self.min_priority)
            .pow(self.per_alpha)
            .cpu()
            .data.numpy()
            .flatten()
        )

        with torch.no_grad():
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

    # pylint: disable-next=arguments-differ, arguments-renamed
    def _update_actor_alpha(  # type: ignore[override]
        self,
        states: torch.Tensor,
        prev_actions: torch.Tensor,
        weights: torch.Tensor,  # pylint: disable=unused-argument
    ) -> dict[str, Any]:
        info: dict[str, Any] = {}

        (
            pi,
            log_pi,
            _,
            act_probs,
            binary_mask,
            log_beta,
        ) = self.actor_net(states, prev_actions, force_act=False)

        with hlp.evaluating(self.critic_net):
            qf_pi_one, qf_pi_two = self.critic_net(states, pi)

        min_qf_pi = torch.minimum(qf_pi_one, qf_pi_two)

        actor_loss = ((self.alpha * log_pi) + (self.beta * log_beta) - min_qf_pi).mean()

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

        # update the temperature (alpha)
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # === Update α_β (for β) ===
        beta_loss = -(self.log_beta * (log_beta + self.target_beta).detach()).mean()

        self.log_beta_optimizer.zero_grad()
        beta_loss.backward()
        self.log_beta_optimizer.step()

        with torch.no_grad():
            # --- SDAR specific diagnostics ---
            # act_probs: the Bernoulli probabilities for selecting new actions vs repeating old ones.
            # binary_mask: the actual sampled 0/1 mask for repetition vs new action.
            # log_beta: the log of the temperature for the β regularization term; should adapt to balance repetition vs new action selection.
            info["act_prob_mean"] = act_probs.mean().item()
            info["log_beta_mean"] = log_beta.mean().item()
            info["binary_mask_mean"] = binary_mask.mean().item()
            info["beta"] = self.beta.item()
            info["log_beta"] = log_beta.mean().item()

            # --- Policy entropy diagnostics (exploration health) ---
            # log_pi more negative -> higher entropy (more stochastic). Less negative -> lower entropy (more deterministic).
            info["log_pi_mean"] = log_pi.mean().item()
            info["log_pi_std"] = log_pi.std().item()

            # --- Action magnitude/saturation (tanh policies) ---
            # High saturation fraction can indicate the policy is slamming bounds; may reduce effective gradients.
            info["pi_action_abs_mean"] = pi.abs().mean().item()
            info["pi_action_std"] = pi.std().item()
            info["pi_action_saturation_frac"] = (pi.abs() > 0.95).float().mean().item()

            # --- On-policy critic signal ---
            # min_qf_pi_mean should generally increase as the policy improves (higher value actions under the policy).
            info["min_qf_pi_mean"] = min_qf_pi.mean().item()

            # --- Twin critics disagreement at policy actions (more relevant than replay actions) ---
            # Large gap here means critics disagree on what the current policy is doing (can destabilize actor updates).
            info["qf_pi_gap_abs_mean"] = (qf_pi_one - qf_pi_two).abs().mean().item()

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

        # Use training utilities for consecutive sampling and tensor conversion
        (
            _,  # states_t1_tensor (not used by SDAR)
            prev_actions_tensor,  # actions_t1_tensor (SDAR's prev_actions)
            _,  # rewards_t1_tensor (not used)
            _,  # next_states_t1_tensor (not used)
            _,  # dones_t1_tensor (not used)
            _,  # extras ignored
            observation_tensor,  # states_t2_tensor (SDAR's current states)
            actions_tensor,  # actions_t2_tensor (SDAR's current actions)
            rewards_tensor,  # rewards_t2_tensor (SDAR's current rewards)
            next_observation_tensor,  # next_states_t2_tensor (SDAR's next states)
            dones_tensor,  # dones_t2_tensor (SDAR's current dones)
            _,  # extras ignored
            _,  # indices (not used by SDAR)
        ) = memory_sampler.consecutive_sample(
            memory_buffer, self.batch_size, self.device
        )

        # Create weights tensor (SDAR doesn't use PER with consecutive sampling)
        batch_size = len(observation_tensor.vector_state_tensor)
        weights_tensor = torch.ones(
            batch_size, 1, dtype=torch.float32, device=self.device
        )

        info: dict[str, Any] = {}

        # Update the Critic
        critic_info, _ = self._update_critic(
            observation_tensor.vector_state_tensor,
            actions_tensor,
            rewards_tensor,
            next_observation_tensor.vector_state_tensor,
            dones_tensor,
            weights_tensor,
        )
        info |= critic_info

        if self.learn_counter % self.policy_update_freq == 0:
            # Update the Actor and Alpha
            actor_info = self._update_actor_alpha(
                observation_tensor.vector_state_tensor,
                prev_actions_tensor,
                weights_tensor,
            )
            info |= actor_info

        if self.learn_counter % self.target_update_freq == 0:
            self.soft_update_params(self.critic_net, self.target_critic_net, self.tau)

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        checkpoint = {
            "actor": self.actor_net.state_dict(),
            "critic": self.critic_net.state_dict(),
            "target_critic": self.target_critic_net.state_dict(),
            "actor_optimizer": self.actor_net_optimiser.state_dict(),
            "critic_optimizer": self.critic_net_optimiser.state_dict(),
            "log_alpha": float(self.log_alpha.detach().cpu().item()),
            "log_alpha_optimizer": self.log_alpha_optimizer.state_dict(),
            "log_beta": float(self.log_beta.detach().cpu().item()),
            "log_beta_optimizer": self.log_beta_optimizer.state_dict(),
            "target_beta": self.target_beta,
            "learn_counter": int(self.learn_counter),
        }
        torch.save(checkpoint, f"{filepath}/{filename}_checkpoint.pth")
        logging.info("models, optimisers, and training state have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        checkpoint = torch.load(f"{filepath}/{filename}_checkpoint.pth")

        self.actor_net.load_state_dict(checkpoint["actor"])
        self.critic_net.load_state_dict(checkpoint["critic"])

        self.target_critic_net.load_state_dict(checkpoint["target_critic"])
        self.actor_net_optimiser.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_net_optimiser.load_state_dict(checkpoint["critic_optimizer"])

        self.log_alpha.data = torch.tensor(
            checkpoint["log_alpha"], dtype=torch.float32, device=self.device
        )
        self.log_alpha_optimizer.load_state_dict(checkpoint["log_alpha_optimizer"])

        self.log_beta.data = torch.tensor(
            checkpoint["log_beta"], dtype=torch.float32, device=self.device
        )
        self.log_beta_optimizer.load_state_dict(checkpoint["log_beta_optimizer"])
        self.target_beta = checkpoint.get("target_beta", self.target_beta)

        self.learn_counter = checkpoint.get("learn_counter", 0)

        logging.info("models, optimisers, and training state have been loaded...")
