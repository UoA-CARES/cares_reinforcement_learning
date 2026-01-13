"""
Original Paper: https://arxiv.org/pdf/1803.11485
"""

import copy
import logging
import os
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.memory.memory_sampler as memory_sampler
import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.algorithm.algorithm import VectorAlgorithm
from cares_reinforcement_learning.networks.QMIX import (
    QMixer,
    SharedMultiAgentNetwork,
)
from cares_reinforcement_learning.types.interaction import ActionContext
from cares_reinforcement_learning.types.training import TrainingContext
from cares_reinforcement_learning.util.configurations import QMIXConfig
from cares_reinforcement_learning.util.helpers import EpsilonScheduler


class QMIX(VectorAlgorithm):
    def __init__(
        self,
        network: SharedMultiAgentNetwork,
        mixer: QMixer,
        config: QMIXConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="value", config=config, device=device)

        self.network = network.to(device)
        self.target_network = copy.deepcopy(self.network).to(device)
        self.target_network.eval()

        self.mixer = mixer.to(device)
        self.target_mixer = copy.deepcopy(self.mixer).to(device)
        self.target_mixer.eval()

        self.tau = config.tau
        self.gamma = config.gamma
        self.target_update_freq = config.target_update_freq

        self.max_grad_norm = config.max_grad_norm

        self.num_agents = network.num_agents
        self.num_actions = network.num_actions

        # Epsilon
        self.epsilon_scheduler = EpsilonScheduler(
            start_epsilon=config.start_epsilon,
            end_epsilon=config.end_epsilon,
            decay_steps=config.decay_steps,
        )
        self.epsilon = self.epsilon_scheduler.get_epsilon(0)

        # Double DQN
        self.use_double_dqn = config.use_double_dqn

        # PER
        self.use_per_buffer = config.use_per_buffer
        self.per_sampling_strategy = config.per_sampling_strategy
        self.per_weight_normalisation = config.per_weight_normalisation
        self.min_priority = config.min_priority
        self.per_alpha = config.per_alpha

        # n-step
        self.n_step = config.n_step

        self.network_optimiser = torch.optim.Adam(
            list(self.network.parameters()) + list(self.mixer.parameters()),
            lr=config.lr,
        )

        self.learn_counter = 0

    def _stack_obs(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Convert dict[str → (B, obs_dim)] into (B, n_agents, obs_dim).
        QMIX requires identical obs_dim for all agents.
        """
        agent_names = list(obs_dict.keys())
        obs_list = [obs_dict[a] for a in agent_names]
        return torch.stack(obs_list, dim=1)

    def select_action_from_policy(self, action_context: ActionContext):
        """
        Epsilon-greedy per-agent action selection.
        Each agent decides independently whether to explore or exploit.
        """
        state = action_context.observation
        evaluation = action_context.evaluation
        available_actions = action_context.available_actions

        actions = []

        # Get greedy actions for all agents once
        observation_tensors = memory_sampler.observation_to_tensors(
            [state], self.device
        )

        assert observation_tensors.agent_states_tensor is not None
        assert observation_tensors.avail_actions_tensor is not None
        # obs_dict_tensors, _, avail_actions_tensor = tu.marl_states_to_tensors(
        #     [state], self.device
        # )

        # [1, num_agents, obs_dim]
        obs_tensors = self._stack_obs(observation_tensors.agent_states_tensor)

        self.network.eval()
        with torch.no_grad():
            q_values = self.network(obs_tensors)  # [1, num_agents, num_actions]
            mask = observation_tensors.avail_actions_tensor == 0
            q_values = q_values.masked_fill(mask, -1e9)
            greedy_actions = q_values.argmax(dim=2).squeeze(0)  # [num_agents]
        self.network.train()

        for agent_id in range(self.num_agents):
            if evaluation:
                # Always exploit in evaluation mode
                actions.append(int(greedy_actions[agent_id]))
            else:
                # Each agent decides independently
                if random.random() < self.epsilon:
                    avail_actions_ind = np.nonzero(available_actions[agent_id])[0]
                    actions.append(int(np.random.choice(avail_actions_ind)))
                else:
                    actions.append(int(greedy_actions[agent_id]))

        return actions

    # def _calculate_value(self, state: np.ndarray, action: int) -> float:  # type: ignore[override]
    #     state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
    #     state_tensor = state_tensor.unsqueeze(0)

    #     with torch.no_grad():
    #         q_values = self.network(state_tensor)
    #         q_value = q_values[0][action].item()

    #     return q_value

    def _compute_loss(
        self,
        obs_tensors: torch.Tensor,
        next_obs_tensors: torch.Tensor,
        states_tensors: torch.Tensor,
        next_states_tensors: torch.Tensor,
        actions_tensors: torch.Tensor,
        next_avail_actions_tensors: torch.Tensor,
        rewards_tensors: torch.Tensor,
        dones_tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Computes the elementwise loss for QMIX. If use_double_dqn=True, applies Double DQN logic."""

        loss_info: dict[str, float] = {}

        q_values = self.network(obs_tensors)
        next_q_values_target = self.target_network(next_obs_tensors)

        # Get Q-values for chosen actions
        best_q_values = q_values.gather(
            dim=2, index=actions_tensors.unsqueeze(-1)
        ).squeeze(-1)

        if self.use_double_dqn:
            # Online network selects best actions
            next_q_values_online = self.network(next_obs_tensors)

            # Mask unavailable actions in next state
            mask = (next_avail_actions_tensors == 0).bool()
            next_q_values_online = next_q_values_online.masked_fill(mask, -1e9)

            # Select argmax among valid next actions
            next_actions = next_q_values_online.argmax(dim=2, keepdim=True)

            # Target network evaluates those actions
            best_next_q_values = next_q_values_target.gather(
                dim=2, index=next_actions
            ).squeeze(2)
        else:
            # Standard DQN: mask next Q-values before taking max
            mask = (next_avail_actions_tensors == 0).bool()
            next_q_values_target = next_q_values_target.masked_fill(mask, -1e9)
            best_next_q_values = next_q_values_target.max(dim=2)[0]

        # Apply mixer network to combine agent Q-values
        # QMIX mixes: Q_total = mixer(best_q_values, global_state)
        q_total = self.mixer(best_q_values, states_tensors)
        q_total_target = self.target_mixer(best_next_q_values, next_states_tensors)

        q_target = (
            rewards_tensors
            + (self.gamma**self.n_step) * (1 - dones_tensors) * q_total_target.detach()
        )

        elementwise_loss = F.mse_loss(q_total, q_target, reduction="none")

        loss_info["td_error_mean"] = elementwise_loss.mean().item()
        loss_info["td_error_std"] = elementwise_loss.std().item()
        loss_info["q_total_mean"] = q_total.mean().item()
        loss_info["q_target_mean"] = q_target.mean().item()
        loss_info["q_next_mean"] = best_next_q_values.mean().item()

        return elementwise_loss, loss_info

    def train_policy(self, training_context: TrainingContext) -> dict[str, Any]:
        info: dict[str, Any] = {}

        self.learn_counter += 1

        memory = training_context.memory
        batch_size = training_context.batch_size
        training_step = training_context.training_step

        self.epsilon = self.epsilon_scheduler.get_epsilon(training_step)

        if len(memory) < batch_size:
            return {}

        # Use training_utils to sample and prepare batch
        (
            observation_tensor,
            actions_tensor,
            rewards_tensor,
            next_observation_tensor,
            dones_tensor,
            weights_tensor,
            indices,
        ) = memory_sampler.sample(
            memory=memory,
            batch_size=batch_size,
            device=self.device,
            use_per_buffer=self.use_per_buffer,
            per_sampling_strategy=self.per_sampling_strategy,
            per_weight_normalisation=self.per_weight_normalisation,
            action_dtype=torch.long,  # DQN uses discrete actions
        )

        assert observation_tensor.agent_states_tensor is not None
        assert next_observation_tensor.agent_states_tensor is not None
        assert next_observation_tensor.avail_actions_tensor is not None

        # Compress rewards and dones to 1D tensors for cooperative setting
        rewards_tensor = rewards_tensor.sum(dim=1, keepdim=True)
        dones_tensor = dones_tensor.any(dim=1, keepdim=True).float()

        # Reshape tensors to match DQN's expected dimensions
        rewards_tensor = rewards_tensor.view(-1)
        dones_tensor = dones_tensor.view(-1)
        weights_tensor = weights_tensor.view(-1)

        obs_tensors = self._stack_obs(observation_tensor.agent_states_tensor)
        next_obs_tensors = self._stack_obs(next_observation_tensor.agent_states_tensor)

        # Calculate loss - overriden by C51
        elementwise_loss, loss_info = self._compute_loss(
            obs_tensors=obs_tensors,
            next_obs_tensors=next_obs_tensors,
            states_tensors=observation_tensor.vector_state_tensor,
            next_states_tensors=next_observation_tensor.vector_state_tensor,
            actions_tensors=actions_tensor,
            rewards_tensors=rewards_tensor,
            next_avail_actions_tensors=next_observation_tensor.avail_actions_tensor,
            dones_tensors=dones_tensor,
        )
        info |= loss_info

        if self.use_per_buffer:
            # Update the Priorities
            priorities = (
                elementwise_loss.clamp(self.min_priority)
                .pow(self.per_alpha)
                .cpu()
                .data.numpy()
                .flatten()
            )

            memory.update_priorities(indices, priorities)

            loss = torch.mean(elementwise_loss * weights_tensor)
        else:
            # Calculate loss
            loss = elementwise_loss.mean()

        info["loss"] = loss.item()
        info["epsilon"] = self.epsilon

        self.network_optimiser.zero_grad()
        loss.backward()

        # Apply gradient clipping if max_grad_norm is set
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self.network.parameters()) + list(self.mixer.parameters()),
                max_norm=self.max_grad_norm,
            )

        self.network_optimiser.step()

        # Update target network - a tau of 1.0 equates to a hard update.
        if self.learn_counter % self.target_update_freq == 0:
            hlp.soft_update_params(self.network, self.target_network, self.tau)
            hlp.soft_update_params(self.mixer, self.target_mixer, self.tau)

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        checkpoint = {
            "network": self.network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "mixer": self.mixer.state_dict(),
            "target_mixer": self.target_mixer.state_dict(),
            "optimizer": self.network_optimiser.state_dict(),
            "learn_counter": self.learn_counter,
        }
        torch.save(checkpoint, f"{filepath}/{filename}_checkpoint.pth")
        logging.info("models and optimiser have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        checkpoint = torch.load(f"{filepath}/{filename}_checkpoint.pth")

        self.network.load_state_dict(checkpoint["network"])
        self.target_network.load_state_dict(checkpoint["target_network"])

        self.mixer.load_state_dict(checkpoint["mixer"])
        self.target_mixer.load_state_dict(checkpoint["target_mixer"])

        self.network_optimiser.load_state_dict(checkpoint["optimizer"])

        self.learn_counter = checkpoint.get("learn_counter", 0)

        logging.info("models, optimiser, and learn_counter have been loaded...")
