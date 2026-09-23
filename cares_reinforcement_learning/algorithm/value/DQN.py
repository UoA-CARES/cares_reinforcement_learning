"""
DQN (Deep Q-Network)
---------------------

Original Paper: https://arxiv.org/abs/1312.5602

DQN is an off-policy, value-based reinforcement learning algorithm
for discrete action spaces. It learns an action-value function
Q(s, a) with a neural network and selects actions via ε-greedy
exploration.

Core Idea:
- Approximate Q*(s,a) with Qθ(s,a).
- Train with TD-learning using replay and a target network
  to stabilize the moving Bellman target.

Data / Replay:
- Transitions are stored in a replay buffer:
      (s, a, r, s', done)
- Minibatches are sampled uniformly (PER is an optional extension).
- Replay breaks temporal correlations and improves data efficiency.

Critic (Q-network) update:
- Bootstrapped target:
      y = r + γ (1 - done) max_{a'} Qθ¯(s', a')
  where Qθ¯ is a slowly-updated target network.
- Loss (typically MSE or Huber):
      L = (Qθ(s,a) - y)^2

Target Network:
- Updated periodically (hard update) or via Polyak averaging.
- Reduces instability from chasing a rapidly-changing target.

Exploration:
- ε-greedy:
      with prob ε: random action
      else: a = argmax_a Qθ(s,a)

Key Behaviour:
- Learns from off-policy data via replay.
- Bootstrapping enables efficient credit assignment.
- Most effective in discrete actions; continuous control
  typically requires actor-critic methods.

DQN = Q-learning + neural function approximation
      + replay buffer + target network.
"""

import copy
import logging
import os
import random
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.memory.memory_sampler as memory_sampler
from cares_reinforcement_learning.algorithm.algorithm import SARLAlgorithm
from cares_reinforcement_learning.algorithm.configurations import DQNConfig
from cares_reinforcement_learning.algorithm.schedulers import LinearScheduler
from cares_reinforcement_learning.memory.memory_buffer import SARLMemoryBuffer
from cares_reinforcement_learning.networks.DQN import BaseNetwork
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import SARLObservation


class DQN(SARLAlgorithm[int]):
    def __init__(
        self,
        network: BaseNetwork,
        config: DQNConfig,
        action_sampler: Callable[[], int],
        device: torch.device,
    ):
        super().__init__(
            policy_type="value",
            config=config,
            action_sampler=action_sampler,
            device=device,
        )

        self.network = network.to(device)
        self.target_network = copy.deepcopy(self.network).to(device)
        self.target_network.eval()

        self.tau = config.tau
        self.gamma = config.gamma
        self.target_update_freq = config.target_update_freq

        self.max_grad_norm = config.max_grad_norm

        # Epsilon
        self.epsilon_scheduler = LinearScheduler(
            start_value=config.start_epsilon,
            end_value=config.end_epsilon,
            decay_steps=config.decay_steps,
        )
        self.epsilon = self.epsilon_scheduler.get_value(0)

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
            self.network.parameters(), lr=config.lr
        )

        self.learn_counter = 0

    def _exploit(self, state: np.ndarray) -> int:
        self.network.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            state_tensor = state_tensor.unsqueeze(0)
            q_values = self.network(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())

        self.network.train()

        return action

    def act(
        self, observation: SARLObservation, evaluation: bool = False
    ) -> ActionSample[int]:
        """
        Select an action from the policy based on epsilon-greedy strategy.
        """
        state = observation.vector_state

        if evaluation:
            return ActionSample(action=self._exploit(state), source="policy")

        if random.random() < self.epsilon:
            return self._explore()

        return ActionSample(action=self._exploit(state), source="policy")

    def train_act(
        self,
        observation: SARLObservation,
        training_step: int,  # pylint: disable=unused-argument
    ) -> ActionSample[int]:
        return self.act(observation, evaluation=False)

    def _calculate_value(self, state: SARLObservation, action: int) -> float:  # type: ignore[override]
        state_tensor = torch.tensor(
            state.vector_state, dtype=torch.float32, device=self.device
        )
        state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            q_values = self.network(state_tensor)
            q_value = q_values[0][action].item()

        return q_value

    def _compute_loss(
        self,
        states_tensor: torch.Tensor,
        actions_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        next_states_tensor: torch.Tensor,
        dones_tensor: torch.Tensor,
        batch_size: int,  # pylint: disable=unused-argument
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Computes the elementwise loss for DQN. If use_double_dqn=True, applies Double DQN logic."""

        q_values = self.network(states_tensor)
        next_q_values_target = self.target_network(next_states_tensor)

        # Get Q-values for chosen actions
        best_q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        if self.use_double_dqn:
            # Online network selects best actions
            next_q_values_online = self.network(next_states_tensor)
            next_actions = next_q_values_online.argmax(dim=1, keepdim=True)
            # Target network estimates their values
            best_next_q_values = next_q_values_target.gather(1, next_actions).squeeze(1)
        else:
            # Standard DQN: Use target network to select best Q-values
            best_next_q_values = torch.max(next_q_values_target, dim=1).values

        q_target = (
            rewards_tensor
            + (self.gamma**self.n_step) * (1 - dones_tensor) * best_next_q_values
        )
        elementwise_loss = F.mse_loss(best_q_values, q_target, reduction="none")

        # -----------------------
        # Logging / diagnostics (DQN)
        # -----------------------
        with torch.no_grad():
            info: dict[str, Any] = {}

            # -------------------------------------------------
            # Greedy action distribution
            # -------------------------------------------------
            greedy_actions = q_values.argmax(dim=1)
            num_actions = self.network.num_actions

            counts = torch.bincount(
                greedy_actions,
                minlength=num_actions,
            ).float()

            probs = counts / counts.sum().clamp(min=1.0)

            # 0 = same greedy action for every state
            # higher = more diverse preferred actions
            entropy = -(probs * (probs + 1e-12).log()).sum()

            info["greedy_action_entropy"] = entropy.item()
            info["greedy_action_max_prob"] = probs.max().item()
            info["greedy_action_probs"] = probs.cpu().tolist()

            # Normalised entropy:
            # 0 = complete action collapse
            # 1 = evenly distributed greedy actions
            if num_actions > 1:
                info["greedy_action_entropy_normalised"] = (
                    entropy / np.log(num_actions)
                ).item()
            else:
                info["greedy_action_entropy_normalised"] = 0.0

            # -------------------------------------------------
            # Greedy action margin
            # -------------------------------------------------
            # Difference between the best and second-best Q-value.
            # Larger values indicate a stronger action preference.
            if num_actions > 1:
                top_two_q_values = torch.topk(q_values, k=2, dim=1).values

                action_margin = top_two_q_values[:, 0] - top_two_q_values[:, 1]

                info["greedy_action_margin_mean"] = action_margin.mean().item()
                info["greedy_action_margin_std"] = action_margin.std().item()
            else:
                info["greedy_action_margin_mean"] = 0.0
                info["greedy_action_margin_std"] = 0.0

            # -------------------------------------------------
            # Action-value separation
            # -------------------------------------------------
            # Average spread of Q-values between available actions
            # for each sampled state.
            q_action_std = q_values.std(dim=1, unbiased=False)

            info["q_action_std_mean"] = q_action_std.mean().item()

            # -------------------------------------------------
            # TD error
            # -------------------------------------------------
            # Existing convention:
            # Q(s,a) - target
            td_error = q_target - best_q_values

            info["td_error_mean"] = td_error.mean().item()
            info["td_error_std"] = td_error.std().item()
            info["td_error_abs_mean"] = td_error.abs().mean().item()

            # -------------------------------------------------
            # Current Q-value statistics
            # -------------------------------------------------
            info["q_value_mean"] = best_q_values.mean().item()
            info["q_value_max"] = best_q_values.max().item()
            info["q_value_std"] = best_q_values.std().item()

            # -------------------------------------------------
            # Next-state/bootstrap Q-value statistics
            # -------------------------------------------------
            info["q_value_next_mean"] = best_next_q_values.mean().item()
            info["q_value_next_max"] = best_next_q_values.max().item()
            info["q_value_next_std"] = best_next_q_values.std().item()

            # -------------------------------------------------
            # Q-space statistics
            # -------------------------------------------------

            # Overall Q magnitude across all states and actions
            info["q_all_mean"] = q_values.mean().item()
            info["q_all_std"] = q_values.std().item()
            info["q_all_min"] = q_values.min().item()
            info["q_all_max"] = q_values.max().item()
            info["q_all_abs_mean"] = q_values.abs().mean().item()

            # Average range between best and worst action for each state
            q_action_range = q_values.max(dim=1).values - q_values.min(dim=1).values

            info["q_action_range_mean"] = q_action_range.mean().item()

            # Per-action Q statistics across the sampled states
            info["q_per_action_mean"] = q_values.mean(dim=0).cpu().tolist()

            info["q_per_action_std"] = q_values.std(dim=0).cpu().tolist()

            # -------------------------------------------------
            # TD target statistics
            # -------------------------------------------------
            info["q_target_mean"] = q_target.mean().item()
            info["q_target_max"] = q_target.max().item()
            info["q_target_std"] = q_target.std().item()

            # -------------------------------------------------
            # Reward statistics
            # -------------------------------------------------
            info["reward_mean"] = rewards_tensor.mean().item()
            info["reward_std"] = rewards_tensor.std().item()

            reward_eps = 1e-8

            info["reward_zero_fraction"] = (
                (rewards_tensor.abs() <= reward_eps).float().mean().item()
            )

            info["reward_positive_fraction"] = (
                (rewards_tensor > reward_eps).float().mean().item()
            )

            info["reward_negative_fraction"] = (
                (rewards_tensor < -reward_eps).float().mean().item()
            )

            # -------------------------------------------------
            # Replay batch composition
            # -------------------------------------------------
            # Fraction of sampled transitions that are terminal.
            info["terminal_fraction"] = dones_tensor.float().mean().item()

        return elementwise_loss, info

    def train(
        self,
        memory_buffer: SARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {}

        self.learn_counter += 1

        training_step = episode_context.training_step

        self.epsilon = self.epsilon_scheduler.get_value(training_step)

        if len(memory_buffer) < self.batch_size:
            return {}

        # Use training_utils to sample and prepare batch
        sample, indices = memory_sampler.sample(
            memory=memory_buffer,
            batch_size=self.batch_size,
            device=self.device,
            use_per_buffer=self.use_per_buffer,
            per_sampling_strategy=self.per_sampling_strategy,
            per_weight_normalisation=self.per_weight_normalisation,
            action_dtype=torch.long,  # DQN uses discrete actions
        )

        sample_size = len(indices)

        # Reshape tensors to match DQN's expected dimensions
        rewards_tensor = sample.reward.view(-1)
        dones_tensor = sample.done.view(-1)
        weights_tensor = sample.weights.view(-1)

        # Calculate loss - overriden by C51
        elementwise_loss, train_info = self._compute_loss(
            sample.observation.vector_state,
            sample.action,
            rewards_tensor,
            sample.next_observation.vector_state,
            dones_tensor,
            sample_size,
        )
        info |= train_info

        if self.use_per_buffer:
            # Update the Priorities
            priorities = (
                elementwise_loss.clamp(self.min_priority)
                .pow(self.per_alpha)
                .cpu()
                .data.numpy()
                .flatten()
            )

            info["per_priority_mean"] = priorities.mean()
            info["per_priority_max"] = priorities.max()
            info["per_priority_min"] = priorities.min()
            info["per_priority_std"] = priorities.std()

            memory_buffer.update_priorities(indices, priorities)

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
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), max_norm=self.max_grad_norm
            )
            info["grad_norm"] = grad_norm.item()
            info["grad_clipped"] = grad_norm.item() > self.max_grad_norm

        self.network_optimiser.step()

        # Update target network - a tau of 1.0 equates to a hard update.
        if self.learn_counter % self.target_update_freq == 0:
            self.soft_update_params(self.network, self.target_network, self.tau)

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        checkpoint = {
            "network": self.network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.network_optimiser.state_dict(),
            "learn_counter": self.learn_counter,
        }
        torch.save(checkpoint, f"{filepath}/{filename}_checkpoint.pth")
        logging.info("models and optimiser have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        checkpoint = torch.load(f"{filepath}/{filename}_checkpoint.pth")

        self.network.load_state_dict(checkpoint["network"])
        self.target_network.load_state_dict(checkpoint["target_network"])

        self.network_optimiser.load_state_dict(checkpoint["optimizer"])

        self.learn_counter = checkpoint.get("learn_counter", 0)

        logging.info("models, optimiser, and learn_counter have been loaded...")
