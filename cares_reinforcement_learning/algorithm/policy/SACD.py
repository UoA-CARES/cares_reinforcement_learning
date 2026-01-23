"""
Original Paper: https://arxiv.org/pdf/1910.07207
Code based on: https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/actor_critic_agents/SAC_Discrete.py

This code runs automatic entropy tuning
"""

from dataclasses import dataclass
from typing import Any, Mapping
from types import MappingProxyType
import copy

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.encoders.vanilla_autoencoder as Encoding
from cares_reinforcement_learning.networks.mlp import MLP
import cares_reinforcement_learning.util.helpers as hlp
import cares_reinforcement_learning.util.training_utils as tu
from cares_reinforcement_learning.algorithm.policy import SAC
from cares_reinforcement_learning.networks.SACD import Actor, Critic
from cares_reinforcement_learning.util.configurations import ImageEncoderType, SACDConfig, VanillaAEConfig
from cares_reinforcement_learning.util.training_context import (
    ActionContext,
    TrainingContext,
)


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
        env_entropy: float,
        config: SACDConfig,
        device: torch.device,
    ):
        super().__init__(actor_network, critic_network, config, device)
        # Override typing for actor and critic networks
        self.actor_net: Actor
        self.critic_net: Critic
        self.policy_type = "discrete_policy"

        # Set base SAC configs
        self.action_num = self.actor_net.num_actions
        self.target_entropy = (np.log(self.action_num) * config.target_entropy_multiplier)
        
        # Assumes that q average and clip should always be used together
        if config.use_clipped_q:
            self.q_clip_epsilon = config.q_clip_epsilon
            self._get_critic_loss = self._get_clipped_critic_loss

        if config.use_average_q:
            self._get_q_target = self._get_avg_q_target

        if config.use_entropy_penalty:
            self.entropy_penalty_beta = config.entropy_penalty_beta

        self.env_entropy = env_entropy
        self.entropy = None

        if config.encoder_type is not None:
            self.normalise_image = getattr(config, "normalise_image", True)
            self._set_encoding(config)


    def _get_q_target(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
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
    
    
    def _get_state_action_q_values(self, state: torch.Tensor, actions: torch.Tensor, network: Critic) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves Q-values for given states and actions from the specified critic network.
        
        :param state: Batch of states
        :type state: torch.Tensor
        :param actions: Batch of actions taken at those states
        :type actions: torch.Tensor
        :param network: Critic network to use for Q-value estimation
        :type network: Critic
        :return: Q-values corresponding to the states and actions
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        q_values_one, q_values_two = network(state)
        return q_values_one.gather(1, actions), q_values_two.gather(1, actions)
    
    
    def _get_critic_loss(self, state: torch.Tensor, actions: torch.Tensor, q_target: torch.Tensor, weights: torch.Tensor = None) -> CriticLossInfo:
        """
        Calculates critic loss using standard MSE loss between Q-values and target Q-values.
        
        :param state: Batch of states from replay buffer experiences
        :type state: torch.Tensor
        :param actions: Batch of actions taken at those states
        :type actions: torch.Tensor
        :param q_target: Target Q-values for the states
        :type q_target: torch.Tensor
        :return: Dataclass containing critic losses and extra info
        :rtype: CriticLossInfo
        """
        q_values_one, q_values_two = self._get_state_action_q_values(state, actions, self.critic_net)
        critic_loss_one = F.mse_loss(q_values_one, q_target)
        critic_loss_two = F.mse_loss(q_values_two, q_target)

        if weights is not None:
            critic_loss_one = (critic_loss_one * weights).mean()
            critic_loss_two = (critic_loss_two * weights).mean()

        return CriticLossInfo(
            q_values_one=q_values_one,
            q_values_two=q_values_two,
            critic_loss_one=critic_loss_one,
            critic_loss_two=critic_loss_two,
        )
    

    def _get_clipped_critic_loss(self, state: torch.Tensor, actions: torch.Tensor, q_target: torch.Tensor, weights: torch.Tensor = None) -> CriticLossInfo:
        """
        Calculates critic loss using clipped Q-values to prevent large updates.
        
        :param state: Batch of states from replay buffer experiences
        :type state: torch.Tensor
        :param actions: Batch of actions taken at those states
        :type actions: torch.Tensor
        :param q_target: Target Q-values for the states
        :type q_target: torch.Tensor
        :return: Dataclass containing critic losses and extra info
        :rtype: CriticLossInfo
        """
        info = {}

        # Get q value estimate from training and target critic networks for each action
        q_values_one, q_values_two = self._get_state_action_q_values(state, actions, self.critic_net)
        q_target_one, q_target_two = self._get_state_action_q_values(state, actions, self.target_critic_net)
        
        # Compute clipped q value and select max loss using standard and clipped q values
        clipped_q1 = q_target_one + torch.clamp(q_values_one - q_target_one, -self.q_clip_epsilon, self.q_clip_epsilon)
        q1_std_loss = F.mse_loss(q_values_one, q_target)
        q1_clp_loss = F.mse_loss(clipped_q1, q_target)
        critic_loss_one = torch.maximum(q1_std_loss, q1_clp_loss)
        info['clipped_q1'] = clipped_q1.mean().item()

        # Repeat for critic 2
        clipped_qf2 = q_target_two + torch.clamp(q_values_two - q_target_two, -self.q_clip_epsilon, self.q_clip_epsilon)
        q2_std_loss = F.mse_loss(q_values_two, q_target)
        q2_clp_loss = F.mse_loss(clipped_qf2, q_target)
        critic_loss_two = torch.maximum(q2_std_loss, q2_clp_loss)
        info['clipped_q2'] = clipped_qf2.mean().item()

        # Compute proportion of: clipped q value losses >= standard q value losses
        clipq_ratio = torch.mean((q1_clp_loss >= q1_std_loss).float()).item() 
        clipq_ratio += torch.mean((q2_clp_loss >= q2_std_loss).float()).item()
        clipq_ratio /= 2.0
        info['clip_ratio'] = clipq_ratio

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
    

    def select_action_from_policy(self, action_context: ActionContext) -> np.ndarray:
        """
        Passes the state from the action context through the actor network that returns a categorical distribution over the action space.

        Depending on whether evaluation mode is set, return either a sampled action (training) or the best action (eval).
        
        :param action_context: Object containing state and action evaluation flag
        :type action_context: ActionContext
        :return: The selected action
        :rtype: ndarray
        """
        self.actor_net.eval()

        state = action_context.state
        evaluation = action_context.evaluation

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            if self.normalise_image:
                state_tensor = state_tensor / 255.0
            state_tensor = state_tensor.unsqueeze(0)
            if evaluation:
                _, _, action = self.actor_net(state_tensor)
            else:
                action, (action_probs, log_action_probs), _ = self.actor_net(state_tensor)
                self.entropy = -torch.sum(action_probs * log_action_probs, dim=-1)
        self.actor_net.train()
        return action.cpu().numpy().flatten()
    

    def get_extras(self) -> list[Any]:
        """
        Get any extra information from the algorithm.
        Here, we return the environment's entropy.
        """
        if self.entropy_penalty_beta is None:
            return []
        
        if self.entropy is None:
            return [self.env_entropy]
        
        return [self.entropy.item()]


    def _get_bootstrapped_value_estimate(self, next_states: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
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
        # TODO: Would gradients propagate into targets?
        # Make sure we are not training target networks
        with torch.no_grad():
            # Set actor to eval to avoid any potential batchnorm/dropout issues and compute entropies
            with hlp.evaluating(self.actor_net):
                _, (action_probs, log_actions_probs), _ = self.actor_net(next_states)
            entropies = -torch.sum(action_probs * log_actions_probs, dim=-1).squeeze()

            # Use target critics to get q-value estimates across actions for the next state after n-steps
            next_target_one, next_target_two = self.target_critic_net(next_states)
            # Consolidate q-value estimates using average or minimum of two critics
            next_target = self._get_q_target(next_target_one, next_target_two)
            # Compute expected q-value of the next state across all actions and add entropy term
            next_target = (next_target * action_probs).sum(dim=-1) + self.alpha * entropies
            # Discount the q-value estimate over n-steps and add discounted rewards
            next_target = (next_target * self.gamma ** self.n_step).unsqueeze(dim=-1)
            next_target = (rewards * self.reward_scale + (1.0 - dones) * next_target)

        return next_target


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
        q_target = self._get_bootstrapped_value_estimate(next_states, rewards, dones)

        # Perform critic loss calculation and back propagation
        act = actions.long()
        critic_loss = self._get_critic_loss(states, act, q_target)
        self.critic_net_optimiser.zero_grad()
        critic_loss.total_loss.backward()
        self.critic_net_optimiser.step()

        # Update the Priorities - PER only
        td_error_one = (critic_loss.q_values_one - q_target).abs()
        td_error_two = (critic_loss.q_values_two - q_target).abs()
        priorities = (
            torch.max(td_error_one, td_error_two)
            .clamp(self.min_priority)
            .pow(self.per_alpha)
            .cpu()
            .data.numpy()
            .flatten()
        )

        return critic_loss.log_info, priorities


    def _update_actor_alpha(self, states: torch.Tensor, old_entropies: torch.Tensor = None) -> tuple[float, float]:
        info = {}
        _, (action_probs, log_action_probs), _ = self.actor_net(states)

        with torch.no_grad():
            with hlp.evaluating(self.critic_net):
                qf1_pi, qf2_pi = self.critic_net(states)
            q_target = self._get_q_target(qf1_pi, qf2_pi)

        entropies = -(action_probs * log_action_probs).sum(dim=-1)
        actor_loss = - (self.alpha * entropies + (action_probs * q_target).sum(dim=-1)).mean()

        if hasattr(self, 'entropy_penalty_beta'):
            entropy_penalty = self.entropy_penalty_beta * F.mse_loss(old_entropies.squeeze(), entropies)
            actor_loss += entropy_penalty
            info["entropy_penalty"] = entropy_penalty.item()

        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        info["actor_loss"] = actor_loss.item()
        info["avg_entropy"] = entropies.mean().item()

        # update the temperature (alpha)
        if self.auto_entropy_tuning:
            alpha_loss = self._update_alpha(entropies)
            info["alpha_loss"] = alpha_loss.item()
            info["alpha"] = self.alpha.item()

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


    def train_policy(self, training_context: TrainingContext) -> dict[str, Any]:
        self.learn_counter += 1

        memory = training_context.memory
        batch_size = training_context.batch_size

        # Use the helper to sample and prepare tensors in one step
        (
            states,
            actions_tensor,
            rewards_tensor,
            next_states,
            dones_tensor,
            old_entropies_tensor,
            weights_tensor,
            indices,
        ) = tu.sample_batch_to_tensors(
            memory=memory,
            batch_size=batch_size,
            device=self.device,
            use_per_buffer=self.use_per_buffer,
            per_sampling_strategy=self.per_sampling_strategy,
            per_weight_normalisation=self.per_weight_normalisation,
        )

        info = {}
        if self.normalise_image:
            states = states / 255.0
            next_states = next_states / 255.0

        # Update the Critic
        critic_info, priorities = self._update_critic(
            states,
            actions_tensor,
            rewards_tensor,
            next_states,
            dones_tensor,
            weights_tensor,
        )
        info |= critic_info

        # Update Autoencoder
        if hasattr(self, "autoencoder"):
            if hasattr(states, "image"):
                ae_loss = self._update_autoencoder(states["image"])
                info["ae_loss"] = ae_loss
            else:
                ae_loss = self._update_autoencoder(states)
                info["ae_loss"] = ae_loss

        if self.learn_counter % self.policy_update_freq == 0:
            # Update the Actor and Alpha
            actor_info = self._update_actor_alpha(states, old_entropies_tensor)
            
            info |= actor_info

        if self.learn_counter % self.target_update_freq == 0:
            hlp.soft_update_params(self.critic_net, self.target_critic_net, self.tau)

        if self.use_per_buffer:
            memory.update_priorities(indices, priorities)

        return info


    def _calculate_value(self, state: np.ndarray, action: np.ndarray) -> float:  # type: ignore[override]
        return 0.0
    

    def _set_encoding(
            self, 
            config: SACDConfig, 
        ) -> None:
        """Sets the encoder for the actor and critic networks."""
        encoder_type = ImageEncoderType(config.encoder_type)
        encoder_config = config.autoencoder_config

        if encoder_type == ImageEncoderType.CONV_NET:
            from cares_reinforcement_learning.encoders.vanilla_autoencoder import NewEncoder

            encoder = NewEncoder(
                observation_size=encoder_config.observation_size,
                latent_dim=encoder_config.latent_dim,
                num_layers=encoder_config.num_layers,
                num_filters=encoder_config.num_filters,
                kernel_size=encoder_config.kernel_size,
                custom_network_config=getattr(config, "conv_config", None),
                detach_at_convs=False
            )
            self.actor_net.set_encoder(encoder)            

            if config.shared_conv_net:
                critic_encoder = encoder
            else:
                critic_encoder = NewEncoder(
                    observation_size=encoder_config.observation_size,
                    latent_dim=encoder_config.latent_dim,
                    num_layers=encoder_config.num_layers,
                    num_filters=encoder_config.num_filters,
                    kernel_size=encoder_config.kernel_size,
                    custom_network_config=getattr(config, "conv_config", None),
                    detach_at_convs=False
                )
            self.critic_net.set_encoder(critic_encoder)
        elif encoder_type == ImageEncoderType.VANILLA_AUTOENCODER:
            from cares_reinforcement_learning.encoders.vanilla_autoencoder import VanillaAutoencoder
            detach_at_convs = getattr(config, "detach_at_convs", True)

            # Initialize autoencoder and set encoders for actor and critic networks
            autoencoder = VanillaAutoencoder(
                observation_size=encoder_config.observation_size,
                latent_dim=encoder_config.latent_dim,
                num_layers=encoder_config.num_layers,
                num_filters=encoder_config.num_filters,
                kernel_size=encoder_config.kernel_size,
                custom_encoder_config=getattr(config, "conv_config", None),
                detach_at_convs=detach_at_convs,
            )
            self.autoencoder = autoencoder
            self.actor_net.set_encoder(autoencoder.encoder.get_detached_encoder()) # Actor trains only FC layer
            self.critic_net.set_encoder(autoencoder.encoder.get_encoder()) # Critic trains FC layer and convs

        # Update target critic net to include encoder
        self.target_critic_net = copy.deepcopy(self.critic_net).to(hlp.get_device())
        self.target_critic_net.eval()

        # Update optimisers to include encoder parameters
        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=config.actor_lr, **config.actor_lr_params
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=config.critic_lr, **config.critic_lr_params
        )
