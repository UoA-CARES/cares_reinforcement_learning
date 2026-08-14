"""
SACD (Soft Actor-Critic for Discrete Action Settings)
------------------------------------------------------

Original Paper: https://arxiv.org/pdf/1910.07207
Original Code: https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/actor_critic_agents/SAC_Discrete.py

Entropy Penalty and Double Average Q-learning with Q-clip:
Original Paper: https://openreview.net/forum?id=EUF2R6VBeU
Original Code: https://github.com/coldsummerday/SD-SAC.git

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
- Regularises entropy updates to mitigate policy instability during training
- Double average Q-learning with Q-clip to reduce pessimistic bias in Q-value estimates.

Advantages:
- Extends SAC to discrete domains (e.g., Atari).
- Competitive sample efficiency without tuning.
- Simple modification of SAC structure.

SACD = SAC with categorical policy +
        exact expectation over discrete actions.
"""

from dataclasses import dataclass
from typing import Any, Mapping
from types import MappingProxyType
import copy

import numpy as np
import torch
import torch.nn.functional as F

from cares_reinforcement_learning.algorithm.policy import SAC
from cares_reinforcement_learning.algorithm.configurations import SACDConfig
from cares_reinforcement_learning.networks.SACD import Actor, Critic

from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import SARLObservation
from cares_reinforcement_learning.memory.memory_buffer import SARLMemoryBuffer
import cares_reinforcement_learning.memory.memory_sampler as memory_sampler
from cares_reinforcement_learning.networks import functional as fnc


@dataclass(frozen=True)
class CriticLossInfo:
    q_values_one: torch.Tensor
    q_values_two: torch.Tensor
    critic_loss_one: torch.Tensor
    critic_loss_two: torch.Tensor
    extra_info: Mapping[str, Any] = MappingProxyType({})

    @property
    def total_loss(self) -> torch.Tensor:
        return self.critic_loss_one + self.critic_loss_two

    @property
    def log_info(self) -> Mapping[str, Any]:
        info = {
            "critic_loss_one": self.critic_loss_one.item(),
            "critic_loss_two": self.critic_loss_two.item(),
            "critic_loss_total": self.total_loss.item(),
        }
        return info | dict(self.extra_info)


class SACD(SAC):
    """
    A discrete action version of Soft Actor-Critic (SAC) algorithm.

    Includes the following toggleable features/enhancements:
    - Reward scaling
    - Automatic entropy tuning
    - N-step returns
    - Clipped Q values in critic loss calculation
    - Average or minimum Q value target calculation
    - Actor update entropy penalty

    :param actor_network: The actor network to use for action selection
    :type actor_network: SACD.Actor
    :param critic_network: The critic network to use for Q-value estimation
    :type critic_network: SACD.Critic
    :param config: Configuration parameters for SACD
    :type config: SACDConfig
    :param device: The device to run the computations on
    :type device: torch.device
    """

    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: SACDConfig,
        device: torch.device,
    ):
        super().__init__(
            actor_network, critic_network, config, device, policy_type="discrete_policy"
        )
        # Override typing for actor and critic networks
        self.actor_net: Actor
        self.critic_net: Critic
        self.policy_type = "discrete_policy"

        # Override default SAC configs
        self.action_num = self.actor_net.num_actions
        self.max_entropy = np.log(self.action_num)
        self.target_entropy = self.max_entropy * config.target_entropy_multiplier
        self.n_step = config.n_step

        # Configure automatic entropy tuning
        self.auto_entropy_tuning = config.auto_entropy_tuning
        if not self.auto_entropy_tuning:
            self.log_alpha.requires_grad = False
            self.log_alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=config.alpha_lr
            )

        # Additional configs
        self.use_clipped_q = config.use_clipped_q
        self.use_average_q = config.use_average_q
        self.use_entropy_penalty = config.use_entropy_penalty

        if self.use_clipped_q:
            self.q_clip_epsilon = config.q_clip_epsilon
            self._get_critic_loss = self._get_clipped_critic_loss

        if self.use_average_q:
            self._get_min_q_target = self._get_avg_q_target

        if self.use_entropy_penalty:
            self.entropy_penalty_beta = config.entropy_penalty_beta

        self.entropy = None

        self.normalise_state = config.normalise_state

    def get_exploration_extras(self):
        return {"entropy": self.max_entropy}

    def _get_min_q_target(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Standard Q-target calculation using minimum of two Q-values.

        :param q1: Critic 1 Q-values
        :type q1: torch.Tensor
        :param q2: Critic 2 Q-values
        :type q2: torch.Tensor
        :return: Minimum of the two Q-values
        :rtype: Tensor
        """
        return torch.minimum(q1, q2)

    def _get_avg_q_target(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Calculates the average Q-target using the mean of two Q-values.

        :param q1: Critic 1 Q-values
        :param q2: Critic 2 Q-values
        :return: Average of the two Q-values
        :rtype: torch.Tensor
        """
        return torch.mean(torch.stack((q1, q2), dim=-1), dim=-1)

    def _get_state_action_q_values(
        self, state: torch.Tensor, actions: torch.Tensor, critic_network: Critic
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves Q-values for given states and actions from the specified critic network.

        :param state: Batch of states
        :type state: torch.Tensor
        :param actions: Batch of actions taken at those states
        :type actions: torch.Tensor
        :param critic_network: Critic network to use for Q-value estimation
        :type critic_network: Critic
        :return: Q-values corresponding to the states and actions
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        q_values_one, q_values_two = critic_network(state)
        return q_values_one.gather(1, actions), q_values_two.gather(1, actions)

    def _get_critic_loss(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        q_target: torch.Tensor,
        weights: torch.Tensor = None,
    ) -> CriticLossInfo:
        """
        Calculates critic loss using standard MSE loss between Q-values and target Q-values.

        :param state: Batch of states from replay buffer experiences
        :type state: torch.Tensor
        :param actions: Batch of actions taken at those states
        :type actions: torch.Tensor
        :param q_target: Target Q-values for the states
        :type q_target: torch.Tensor
        :param weights: Optional batch of weights for PER
        :type weights: torch.Tensor, optional
        :return: Dataclass containing critic losses and extra info
        :rtype: CriticLossInfo
        """
        q_values_one, q_values_two = self._get_state_action_q_values(
            state, actions, self.critic_net
        )
        critic_loss_one = F.mse_loss(q_values_one, q_target)
        critic_loss_two = F.mse_loss(q_values_two, q_target)

        if weights is not None:
            critic_loss_one = (critic_loss_one * weights).mean()
            critic_loss_two = (critic_loss_two * weights).mean()

        critic_loss_info = CriticLossInfo(
            q_values_one=q_values_one,
            q_values_two=q_values_two,
            critic_loss_one=critic_loss_one,
            critic_loss_two=critic_loss_two,
        )

        return critic_loss_info

    def _get_clipped_critic_loss(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        q_target: torch.Tensor,
        weights: torch.Tensor = None,
    ) -> CriticLossInfo:
        """
        Calculates critic loss using clipped Q-values to prevent large updates.

        :param state: Batch of states from replay buffer experiences
        :type state: torch.Tensor
        :param actions: Batch of actions taken at those states
        :type actions: torch.Tensor
        :param q_target: Target Q-values for the states
        :type q_target: torch.Tensor
        :param weights: Optional batch of weights for PER
        :type weights: torch.Tensor, optional
        :return: Dataclass containing critic losses and extra info
        :rtype: CriticLossInfo
        """
        info = {}

        # Get q value estimate from training and target critic networks for each action
        q_values_one, q_values_two = self._get_state_action_q_values(
            state, actions, self.critic_net
        )
        q_target_one, q_target_two = self._get_state_action_q_values(
            state, actions, self.target_critic_net
        )

        # Compute clipped q value and select max loss using standard and clipped q values
        clipped_q1 = q_target_one + torch.clamp(
            q_values_one - q_target_one, -self.q_clip_epsilon, self.q_clip_epsilon
        )
        q1_std_loss = F.mse_loss(q_values_one, q_target)
        q1_clp_loss = F.mse_loss(clipped_q1, q_target)
        critic_loss_one = torch.maximum(q1_std_loss, q1_clp_loss)
        info["clipped_q1"] = clipped_q1.mean().item()

        # Repeat for critic 2
        clipped_qf2 = q_target_two + torch.clamp(
            q_values_two - q_target_two, -self.q_clip_epsilon, self.q_clip_epsilon
        )
        q2_std_loss = F.mse_loss(q_values_two, q_target)
        q2_clp_loss = F.mse_loss(clipped_qf2, q_target)
        critic_loss_two = torch.maximum(q2_std_loss, q2_clp_loss)
        info["clipped_q2"] = clipped_qf2.mean().item()

        # Compute proportion of: clipped q value losses >= standard q value losses
        clipq_ratio = torch.mean((q1_clp_loss >= q1_std_loss).float()).item()
        clipq_ratio += torch.mean((q2_clp_loss >= q2_std_loss).float()).item()
        clipq_ratio /= 2.0
        info["clip_ratio"] = clipq_ratio

        if weights is not None:
            critic_loss_one = (critic_loss_one * weights).mean()
            critic_loss_two = (critic_loss_two * weights).mean()

        return CriticLossInfo(
            q_values_one=q_values_one,
            q_values_two=q_values_two,
            critic_loss_one=critic_loss_one,
            critic_loss_two=critic_loss_two,
            extra_info=MappingProxyType(info),
        )

    def act(
        self, observation: SARLObservation, evaluation: bool = False
    ) -> ActionSample[int]:
        """
        Passes the state from the action context through the actor network that returns a categorical distribution over the action space.

        Depending on whether evaluation mode is set, return either a sampled action (training) or the best action (eval).

        :param observation: The current observation
        :type observation: SARLObservation
        :param evaluation: Whether to evaluate the policy
        :type evaluation: bool
        :return: The selected action
        :rtype: ActionSample[int]
        """

        self.actor_net.eval()

        state = observation.vector_state

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)

            if self.normalise_state:
                state_tensor = state_tensor / 255.0
            state_tensor = state_tensor.unsqueeze(0)

            if evaluation:
                _, _, action = self.actor_net(state_tensor)
            else:
                action, probs, _ = self.actor_net(state_tensor)
                action_probs, log_action_probs = probs
                self.entropy = -torch.sum(action_probs * log_action_probs, dim=-1)
        self.actor_net.train()

        entropy = None
        if self.entropy is not None:
            entropy = self.entropy.item()

        return ActionSample(
            action=action.item(), source="policy", extras={"entropy": entropy}
        )

    def _compute_next_state_q_value(
        self, next_states: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the n-step bootstrapped value estimate for the next states using the target critic networks.

        :param next_states: Batch of next states from replay buffer experiences
        :type next_states: torch.Tensor
        :param rewards: Batch of rewards from replay buffer experiences
        :type rewards: torch.Tensor
        :param dones: Batch of done flags from replay buffer experiences
        :type dones: torch.Tensor
        :return: Batch of bootstrapped value estimates
        :rtype: torch.Tensor
        """
        # Make sure we are not training target networks
        with torch.no_grad():
            # Set actor to eval to avoid any potential batchnorm/dropout issues and compute entropies
            with fnc.evaluating(self.actor_net):
                _, (action_probs, log_actions_probs), _ = self.actor_net(next_states)
            next_state_entropies = -torch.sum(
                action_probs * log_actions_probs, dim=-1
            ).squeeze()

            # Use target critics to get q-value estimates across actions for the next state after n-steps
            next_target_one, next_target_two = self.target_critic_net(next_states)

            # Consolidate q-value estimates based on choice of average or minimum
            min_next_q_target = self._get_min_q_target(next_target_one, next_target_two)

            # Compute expected q-value of the next state across all actions and add entropy term
            expected_next_q_value = (min_next_q_target * action_probs).sum(
                dim=-1
            ) + self.alpha * next_state_entropies

            # Discount the q-value estimate over n-steps and add discounted rewards
            discounted_next_q_value = (
                expected_next_q_value * self.gamma**self.n_step
            ).unsqueeze(dim=-1)
            bootstrapped_q_value = (
                rewards * self.reward_scale + (1.0 - dones) * discounted_next_q_value
            )

        return (
            next_state_entropies,
            min_next_q_target,
            expected_next_q_value,
            bootstrapped_q_value,
        )

    def _update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[dict[str, float], np.ndarray]:
        """
        Updates the critic networks using the sampled batch of experiences.

        :param states: Batch of states from replay buffer experiences
        :type states: torch.Tensor
        :param actions: Batch of actions from replay buffer experiences
        :type actions: torch.Tensor
        :param rewards: Batch of rewards from replay buffer experiences
        :type rewards: torch.Tensor
        :param next_states: Batch of next states from replay buffer experiences
        :type next_states: torch.Tensor
        :param dones: Batch of done flags from replay buffer experiences
        :type dones: torch.Tensor
        :param weights: Batch of weights from replay buffer experiences
        :type weights: torch.Tensor
        :return: Critic loss info for logging and PER priorities
        :rtype: tuple[dict[str, float], np.ndarray]
        """
        info: dict[str, Any] = {}

        (
            next_state_entropies,
            min_next_q_target,
            expected_next_q_value,
            bootstrapped_q_value,
        ) = self._compute_next_state_q_value(next_states, rewards, dones)

        # Calculate critic loss and update critic networks
        act = actions.long().unsqueeze(-1)
        critic_loss = self._get_critic_loss(
            states, act, bootstrapped_q_value, weights=weights
        )
        self.critic_net_optimiser.zero_grad()
        critic_loss.total_loss.backward()
        self.critic_net_optimiser.step()

        priorities = None
        if self.use_per_buffer:
            # Update the Priorities
            td_error_one = (critic_loss.q_values_one - bootstrapped_q_value).abs()
            td_error_two = (critic_loss.q_values_two - bootstrapped_q_value).abs()
            priorities = (
                torch.max(td_error_one, td_error_two)
                .clamp(self.min_priority)
                .pow(self.per_alpha)
                .cpu()
                .data.numpy()
                .flatten()
            )

        with torch.no_grad():
            # --- Target decomposition ---
            info["target_min_q_mean"] = min_next_q_target.mean().item()
            info["entropy_bonus_mean"] = (
                (self.alpha * next_state_entropies).mean().item()
            )  # TODO: Discuss difference with henry
            info["soft_value_mean"] = expected_next_q_value.mean().item()

            # --- Bellman target scale ---
            info["q_target_mean"] = bootstrapped_q_value.mean().item()
            info["q_target_std"] = bootstrapped_q_value.std(unbiased=False).item()

            # --- Critic value scale ---
            info["q1_mean"] = critic_loss.q_values_one.mean().item()
            info["q2_mean"] = critic_loss.q_values_two.mean().item()
            info["q_twin_gap_abs_mean"] = (
                (critic_loss.q_values_one - critic_loss.q_values_two)
                .abs()
                .mean()
                .item()
            )

            # --- TD error diagnostics ---
            td1 = critic_loss.q_values_one - bootstrapped_q_value
            td2 = critic_loss.q_values_two - bootstrapped_q_value

            td_abs = torch.maximum(td1.abs(), td2.abs()).squeeze(1)
            info["td_abs_mean"] = td_abs.mean().item()
            info["td_abs_p95"] = td_abs.quantile(0.95).item()
            info["td_abs_max"] = td_abs.max().item()

            # --- Loss ---
            info["critic_loss_one"] = critic_loss.critic_loss_one.item()
            info["critic_loss_two"] = critic_loss.critic_loss_two.item()
            info["critic_loss_total"] = critic_loss.total_loss.item()

        return info | dict(critic_loss.extra_info), priorities

    def _update_actor_alpha(
        self,
        states: torch.Tensor,
        old_entropies: torch.Tensor = None,
    ) -> tuple[float, float]:
        info = {}

        _, (action_probs, log_action_probs), _ = self.actor_net(states)

        with fnc.evaluating(self.critic_net):
            qf1_pi, qf2_pi = self.critic_net(states)

        q_target = self._get_min_q_target(
            qf1_pi, qf2_pi
        )  # TODO: Should we have option between min and avg here?

        entropies = -(action_probs * log_action_probs).sum(dim=-1)
        actor_loss = -(
            self.alpha * entropies + (action_probs * q_target).sum(dim=-1)
        ).mean()

        if hasattr(self, "entropy_penalty_beta"):
            entropy_penalty = self.entropy_penalty_beta * F.mse_loss(
                old_entropies.squeeze(), entropies
            )
            actor_loss += entropy_penalty
            info["entropy_penalty"] = entropy_penalty.item()

        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        # update the temperature (alpha)
        if self.auto_entropy_tuning:
            alpha_loss = self._update_alpha(entropies)
            info["alpha_loss"] = alpha_loss.item()

        with torch.no_grad():
            # --- Policy distribution health ---
            info["entropy_mean"] = entropies.mean().item()
            info["entropy_std"] = entropies.std(unbiased=False).item()

            # Action distribution sharpness
            max_prob = action_probs.max(dim=-1).values
            info["max_prob_mean"] = max_prob.mean().item()
            info["max_prob_std"] = max_prob.quantile(0.95).item()

            info["policy_prob_std_mean"] = action_probs.std(dim=1).mean().item()

            # --- Q signal to actor ---
            info["q_target_mean"] = q_target.mean().item()
            info["q_target_std"] = q_target.std(unbiased=False).item()

            # --- Entropy calibration ---
            entropy_gap = entropies - self.target_entropy
            info["entropy_gap_mean"] = entropy_gap.mean().item()

            # --- Losses & temperature ---
            info["actor_loss"] = actor_loss.item()
            info["alpha"] = self.alpha.item()
            info["log_alpha"] = self.log_alpha.item()

        return info

    def _update_alpha(self, entropy: torch.Tensor) -> torch.Tensor:
        # update the temperature (alpha)
        log_prob = -entropy.detach() + self.target_entropy
        alpha_loss = -(self.log_alpha * log_prob).mean()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return alpha_loss

    def _update_autoencoder(self, states: torch.Tensor) -> float:
        # Leaving this function in case this needs to be extended again in the future
        ae_loss = self.autoencoder.update_autoencoder(states)
        return ae_loss.item()

    def train(
        self,
        memory_buffer: SARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:
        self.learn_counter += 1

        sample_tensor, indices = memory_sampler.sample(
            memory=memory_buffer,
            batch_size=self.batch_size,
            device=self.device,
            use_per_buffer=self.use_per_buffer,
            per_sampling_strategy=self.per_sampling_strategy,
            per_weight_normalisation=self.per_weight_normalisation,
        )

        if self.use_entropy_penalty:
            old_entropies_tensor = torch.Tensor(
                [item["entropy"] for item in sample_tensor.train_data]
            ).to(self.device)
        else:
            old_entropies_tensor = None

        info = {}

        obs_state_tensor = sample_tensor.observation.vector_state
        next_obs_state_tensor = sample_tensor.next_observation.vector_state
        if self.normalise_state:
            obs_state_tensor = obs_state_tensor / 255.0
            next_obs_state_tensor = next_obs_state_tensor / 255.0

        # Update the Critic
        critic_info, priorities = self._update_critic(
            obs_state_tensor,
            sample_tensor.action,
            sample_tensor.reward,
            next_obs_state_tensor,
            sample_tensor.done,
            sample_tensor.weights,
        )
        info.update(critic_info)

        if self.learn_counter % self.policy_update_freq == 0:
            # Update the Actor and Alpha
            actor_info = self._update_actor_alpha(
                obs_state_tensor, old_entropies_tensor
            )

            info.update(actor_info)

        if self.learn_counter % self.target_update_freq == 0:
            self.soft_update_params(self.critic_net, self.target_critic_net, self.tau)

        if self.use_per_buffer:
            memory_buffer.update_priorities(indices, priorities)

        return info

    def _calculate_value(self, state: np.ndarray, action: np.ndarray) -> float:  # type: ignore[override]
        return 0.0
