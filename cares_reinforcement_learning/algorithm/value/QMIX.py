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

import cares_reinforcement_learning.util.helpers as hlp
import cares_reinforcement_learning.util.training_utils as tu
from cares_reinforcement_learning.algorithm.algorithm import VectorAlgorithm
from cares_reinforcement_learning.networks.QMIX import (
    SharedMultiAgentNetwork,
    QMixer,
)
from cares_reinforcement_learning.util.configurations import QMIXConfig
from cares_reinforcement_learning.util.helpers import EpsilonScheduler
from cares_reinforcement_learning.util.training_context import (
    ActionContext,
    TrainingContext,
)


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

    def select_action_from_policy(self, action_context: ActionContext):
        """
        Epsilon-greedy per-agent action selection.
        Each agent decides independently whether to explore or exploit.
        """
        state = action_context.state
        evaluation = action_context.evaluation
        available_actions = action_context.available_actions

        assert isinstance(state, dict)

        actions = []

        # Get greedy actions for all agents once
        state_tensor = tu.marl_states_to_tensors([state], self.device)
        obs_tensor = state_tensor["obs"]
        avail_actions_tensor = state_tensor["avail_actions"]

        self.network.eval()
        with torch.no_grad():
            q_values = self.network(obs_tensor)  # [1, num_agents, num_actions]
            mask = avail_actions_tensor == 0
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
        states_tensor: dict[str, torch.Tensor],
        actions_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        next_states_tensor: dict[str, torch.Tensor],
        dones_tensor: torch.Tensor,
        batch_size: int,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Computes the elementwise loss for DQN. If use_double_dqn=True, applies Double DQN logic."""

        obs_tensor = states_tensor["obs"]
        next_obs_tensor = next_states_tensor["obs"]

        global_states_tensor = states_tensor["state"]
        next_global_states_tensor = next_states_tensor["state"]

        avail_actions_tensor = states_tensor["avail_actions"]
        next_avail_actions_tensor = next_states_tensor["avail_actions"]

        q_values = self.network(obs_tensor)
        next_q_values_target = self.target_network(next_obs_tensor)

        # Get Q-values for chosen actions
        best_q_values = q_values.gather(
            dim=2, index=actions_tensor.unsqueeze(-1)
        ).squeeze(-1)

        if self.use_double_dqn:
            # Online network selects best actions
            next_q_values_online = self.network(next_obs_tensor)

            # Mask unavailable actions in next state
            mask = (next_avail_actions_tensor == 0).bool()
            next_q_values_online = next_q_values_online.masked_fill(mask, -1e9)

            # Select argmax among valid next actions
            next_actions = next_q_values_online.argmax(dim=2, keepdim=True)

            # Target network evaluates those actions
            best_next_q_values = next_q_values_target.gather(
                dim=2, index=next_actions
            ).squeeze(2)
        else:
            # Standard DQN: mask next Q-values before taking max
            mask = (next_avail_actions_tensor == 0).bool()
            next_q_values_target = next_q_values_target.masked_fill(mask, -1e9)
            best_next_q_values = next_q_values_target.max(dim=2)[0]

        # Apply mixer network to combine agent Q-values
        # QMIX mixes: Q_total = mixer(best_q_values, global_state)
        q_total = self.mixer(best_q_values, global_states_tensor)
        q_total_target = self.target_mixer(
            best_next_q_values, next_global_states_tensor
        )

        q_target = (
            rewards_tensor
            + (self.gamma**self.n_step) * (1 - dones_tensor) * q_total_target.detach()
        )

        elementwise_loss = F.mse_loss(q_total, q_target, reduction="none")

        return elementwise_loss

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
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
            weights_tensor,
            indices,
        ) = tu.sample_marl_batch_to_tensors(
            memory,
            batch_size,
            self.device,
            use_per_buffer=self.use_per_buffer,
            per_sampling_strategy=self.per_sampling_strategy,
            per_weight_normalisation=self.per_weight_normalisation,
            action_dtype=torch.long,  # DQN uses discrete actions
        )

        # Reshape tensors to match DQN's expected dimensions
        rewards_tensor = rewards_tensor.view(-1)
        dones_tensor = dones_tensor.view(-1)
        weights_tensor = weights_tensor.view(-1)

        # Calculate loss - overriden by C51
        elementwise_loss = self._compute_loss(
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
            batch_size,
        )

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
