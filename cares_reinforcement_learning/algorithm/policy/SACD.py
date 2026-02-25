"""
SACD (Soft Actor-Critic for Discrete Action Settings)
------------------------------------------------------

Original Paper: https://arxiv.org/pdf/1910.07207
Original Code: https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/actor_critic_agents/SAC_Discrete.py

SACD adapts Soft Actor-Critic (SAC) to discrete action
spaces while preserving the maximum-entropy objective.

Core Idea:
- Maximize expected return + entropy bonus.
- Replace continuous policy with a categorical distribution.
- Compute expectations exactly over discrete actions
  (no reparameterization trick required).

Objective:
    J(π) = E[ r(s,a) + γ V(s') ]
    with entropy regularization:
    + α H(π(·|s))

Architecture Changes vs Continuous SAC:

1) Q-Network:
   - Outputs Q(s) ∈ R^{|A|}
   - One value per discrete action.

2) Policy:
   - Outputs π(a|s) via softmax.
   - Direct probability vector over actions.

Critic Target:
    V(s) = Σ_a π(a|s) [ Q(s,a) - α log π(a|s) ]

    y = r + γ V_target(s')

Twin Q-networks are used and the minimum is applied
for stability (clipped double Q-learning).

Actor Update:
    J_actor = E_s [ Σ_a π(a|s)
                    ( α log π(a|s) - Q(s,a) ) ]

The expectation over actions is computed exactly,
reducing variance compared to sampling-based updates.

Temperature Update:
    J(α) = E_s [ Σ_a π(a|s)
                 ( -α (log π(a|s) + H_target) ) ]

Key Behaviour:
- No action sampling required for expectation terms.
- Lower variance policy and temperature updates.
- Maintains entropy-regularized exploration.

Advantages:
- Extends SAC to discrete domains (e.g., Atari).
- Competitive sample efficiency without tuning.
- Simple modification of SAC structure.

SACD = SAC with categorical policy +
        exact expectation over discrete actions.
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
from cares_reinforcement_learning.networks.SACD import Actor, Critic
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import SARLObservation
from cares_reinforcement_learning.util.configurations import SACDConfig


class SACD(SARLAlgorithm[int]):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: SACDConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="discrete_policy", config=config, device=device)

        # this may be called policy_net in other implementations
        self.actor_net = actor_network.to(device)

        # this may be called soft_q_net in other implementations
        self.critic_net = critic_network.to(device)
        self.target_critic_net = copy.deepcopy(self.critic_net).to(device)
        self.target_critic_net.eval()  # never in training mode - helps with batch/drop out layers

        self.gamma = config.gamma
        self.tau = config.tau
        self.reward_scale = config.reward_scale

        self.learn_counter = 0
        self.policy_update_freq = config.policy_update_freq
        self.target_update_freq = config.target_update_freq

        self.action_num = self.actor_net.num_actions

        # For smaller action spaces, set the multiplier to lower values (probs should be a config option)
        self.target_entropy = (
            -np.log(1.0 / self.action_num) * config.target_entropy_multiplier
        )

        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=config.actor_lr
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=config.critic_lr
        )

        # Temperature (alpha) for the entropy loss
        # Set to initial alpha to 1.0 according to other baselines.
        init_temperature = 1.0
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=config.alpha_lr
        )

    def act(
        self, observation: SARLObservation, evaluation: bool = False
    ) -> ActionSample[int]:

        self.actor_net.eval()

        state = observation.vector_state

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            state_tensor = state_tensor.unsqueeze(0)
            if evaluation:
                _, _, action = self.actor_net(state_tensor)
                # action = np.argmax(action_probs)
            else:
                action, _, _ = self.actor_net(state_tensor)
                # action = np.random.choice(a=self.action_num, p=action_probs)
        self.actor_net.train()

        return ActionSample(action=action.item(), source="policy")

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {}

        with torch.no_grad():
            with hlp.evaluating(self.actor_net):
                _, (action_probs, log_actions_probs), _ = self.actor_net(next_states)

            qf1_next, qf2_next = self.target_critic_net(next_states)
            min_q_next = torch.minimum(qf1_next, qf2_next)

            # Soft value: expectation over discrete actions
            soft_value = (
                action_probs * (min_q_next - self.alpha * log_actions_probs)
            ).sum(dim=1, keepdim=True)

            next_q_value = (
                rewards * self.reward_scale + (1.0 - dones) * self.gamma * soft_value
            )

        q_values_one, q_values_two = self.critic_net(states)

        gathered_q_values_one = q_values_one.gather(1, actions.long().unsqueeze(-1))
        gathered_q_values_two = q_values_two.gather(1, actions.long().unsqueeze(-1))

        critic_loss_one = F.mse_loss(gathered_q_values_one, next_q_value)
        critic_loss_two = F.mse_loss(gathered_q_values_two, next_q_value)
        critic_loss_total = critic_loss_one + critic_loss_two

        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        with torch.no_grad():
            # --- Target decomposition ---
            info["target_min_q_mean"] = min_q_next.mean().item()
            info["entropy_bonus_mean"] = (-self.alpha * log_actions_probs).mean().item()
            info["soft_value_mean"] = soft_value.mean().item()

            # --- Bellman target scale ---
            info["q_target_mean"] = next_q_value.mean().item()
            info["q_target_std"] = next_q_value.std(unbiased=False).item()

            # --- Critic value scale ---
            info["q1_mean"] = q_values_one.mean().item()
            info["q2_mean"] = q_values_two.mean().item()
            info["q_twin_gap_abs_mean"] = (
                (q_values_one - q_values_two).abs().mean().item()
            )

            # --- TD error diagnostics ---
            td1 = gathered_q_values_one - next_q_value
            td2 = gathered_q_values_two - next_q_value

            td_abs = torch.maximum(td1.abs(), td2.abs()).squeeze(1)
            info["td_abs_mean"] = td_abs.mean().item()
            info["td_abs_p95"] = td_abs.quantile(0.95).item()
            info["td_abs_max"] = td_abs.max().item()

            # --- Loss ---
            info["critic_loss_one"] = critic_loss_one.item()
            info["critic_loss_two"] = critic_loss_two.item()
            info["critic_loss_total"] = critic_loss_total.item()

        return info

    def _update_actor_alpha(self, states: torch.Tensor) -> dict[str, Any]:
        info: dict[str, Any] = {}

        _, (action_probs, log_action_probs), _ = self.actor_net(states)

        with hlp.evaluating(self.critic_net):
            qf1_pi, qf2_pi = self.critic_net(states)

        min_qf_pi = torch.minimum(qf1_pi, qf2_pi)

        inside_term = self.alpha * log_action_probs - min_qf_pi
        actor_loss = (action_probs * inside_term).sum(dim=1).mean()

        expected_log_prob = torch.sum(log_action_probs * action_probs, dim=1)

        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        # update the temperature (alpha)
        alpha_loss = -(
            self.log_alpha * (expected_log_prob + self.target_entropy).detach()
        ).mean()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        with torch.no_grad():
            # --- Policy distribution health ---
            entropy = -(action_probs * log_action_probs).sum(dim=1)

            info["entropy_mean"] = entropy.mean().item()
            info["entropy_std"] = entropy.std(unbiased=False).item()

            # Action distribution sharpness
            max_prob = action_probs.max(dim=1).values
            info["max_action_prob_mean"] = max_prob.mean().item()
            info["max_action_prob_p95"] = max_prob.quantile(0.95).item()

            info["policy_prob_std_mean"] = action_probs.std(dim=1).mean().item()

            # --- Q signal to actor ---
            info["min_q_pi_mean"] = min_qf_pi.mean().item()
            info["min_q_pi_std"] = min_qf_pi.std(unbiased=False).item()

            # --- Entropy calibration ---
            entropy_gap = -(expected_log_prob + self.target_entropy)
            info["entropy_gap_mean"] = entropy_gap.mean().item()

            # --- Losses & temperature ---
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
            use_per_buffer=0,  # SACD uses uniform sampling
        )

        info = {}

        # Update the Critic
        critic_info = self._update_critic(
            observation_tensor.vector_state_tensor,
            actions_tensor,
            rewards_tensor,
            next_observation_tensor.vector_state_tensor,
            dones_tensor,
        )
        info.update(critic_info)

        if self.learn_counter % self.policy_update_freq == 0:
            # Update the Actor and Alpha
            actor_info = self._update_actor_alpha(
                observation_tensor.vector_state_tensor
            )
            info.update(actor_info)

        if self.learn_counter % self.target_update_freq == 0:
            hlp.soft_update_params(self.critic_net, self.target_critic_net, self.tau)

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
            # Save log_alpha as a float, not a numpy array
            "log_alpha": self.log_alpha.detach().cpu().item(),
            "log_alpha_optimizer": self.log_alpha_optimizer.state_dict(),
            "learn_counter": self.learn_counter,
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

        self.log_alpha.data = torch.tensor(checkpoint["log_alpha"]).to(self.device)
        self.log_alpha_optimizer.load_state_dict(checkpoint["log_alpha_optimizer"])

        self.learn_counter = checkpoint.get("learn_counter", 0)
        logging.info("models, optimisers, and training state have been loaded...")
