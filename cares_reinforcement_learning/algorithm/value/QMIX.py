"""
QMIX (Monotonic Value Function Factorisation)
----------------------------------------------

Original Paper: https://arxiv.org/pdf/1803.11485

QMIX is a value-based multi-agent RL algorithm for cooperative
tasks with a shared team reward. It enables centralized training
with decentralized execution by factorizing the joint action-value.

Core Idea:
- Learn per-agent utility functions:
      Q_i(o_i, a_i)
- Combine them into a joint action-value:
      Q_tot(s, a_1..a_n)
  using a mixing network that enforces a monotonic constraint:
      ∂Q_tot / ∂Q_i >= 0  for all agents i

This guarantees that maximizing each agent's Q_i independently
also maximizes Q_tot, enabling decentralized greedy action
selection at execution time.

Architecture:
- Agent networks: estimate Q_i from local observation/history.
- Mixing network: produces Q_tot from {Q_i} and global state s.
- Hypernetworks: generate mixing weights conditioned on s,
  allowing state-dependent coordination while keeping monotonicity.

Training (centralized):
- Use TD-learning on Q_tot with a team reward:
      y = r + γ max_{a'} Q_tot(s', a')
- Loss:
      L = (Q_tot(s, a) - y)^2
- Typically uses target networks and replay buffer (DQN-style).

Execution (decentralized):
- Each agent selects action greedily from its own utility:
      a_i = argmax_a Q_i(o_i, a)

Rationale:
- Pure independent Q-learning fails due to non-stationarity.
- Full joint Q(s, a_1..a_n) is intractable as agents scale.
- QMIX captures coordination via state-conditioned mixing while
  preserving decentralizable argmax through monotonicity.

QMIX = per-agent Q-learning + centralized monotonic mixing
       for cooperative MARL.
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
from cares_reinforcement_learning.algorithm.algorithm import MARLAlgorithm
from cares_reinforcement_learning.algorithm.configurations import QMIXConfig
from cares_reinforcement_learning.algorithm.schedulers import LinearScheduler
from cares_reinforcement_learning.memory.memory_buffer import MARLMemoryBuffer
from cares_reinforcement_learning.networks.QMIX import QMixer, SharedMultiAgentNetwork
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import MARLObservation


class QMIX(MARLAlgorithm[list[int]]):
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

    def act(
        self, observation: MARLObservation, evaluation: bool = False
    ) -> ActionSample[list[int]]:
        """
        Epsilon-greedy per-agent action selection.
        Each agent decides independently whether to explore or exploit.
        """
        actions = []

        # Get greedy actions for all agents once
        observation_tensors = memory_sampler.observation_to_tensors(
            [observation], self.device
        )

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
                    avail_actions_ind = np.nonzero(observation.avail_actions[agent_id])[
                        0
                    ]
                    actions.append(int(np.random.choice(avail_actions_ind)))
                else:
                    actions.append(int(greedy_actions[agent_id]))

        return ActionSample(action=actions, source="policy")

    def _compute_loss(
        self,
        obs_tensors: torch.Tensor,
        next_obs_tensors: torch.Tensor,
        states_tensors: torch.Tensor,
        next_states_tensors: torch.Tensor,
        actions_tensors: torch.Tensor,
        avail_actions: torch.Tensor,
        next_avail_actions_tensors: torch.Tensor,
        rewards_tensors: torch.Tensor,
        dones_tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Computes the elementwise loss for QMIX. If use_double_dqn=True, applies Double DQN logic."""

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

        # ----------------------------
        # Logging / diagnostics (QMIX)
        # ----------------------------
        loss_info: dict[str, float] = {}
        with torch.no_grad():
            td_total = (q_total - q_target).view(-1)  # [B]

            loss_info["td_total_mean"] = td_total.mean().item()
            loss_info["td_total_std"] = td_total.std().item()
            loss_info["td_total_abs_mean"] = td_total.abs().mean().item()
            loss_info["mse_total_mean"] = elementwise_loss.mean().item()

            loss_info["q_total_mean"] = q_total.mean().item()
            loss_info["q_target_mean"] = q_target.mean().item()

            # Per-agent utilities (chosen + bootstrap)
            loss_info["q_i_chosen_mean"] = best_q_values.mean().item()
            loss_info["q_i_chosen_std"] = best_q_values.std().item()
            loss_info["q_i_next_mean"] = best_next_q_values.mean().item()
            loss_info["q_i_next_std"] = best_next_q_values.std().item()

            # --- Current-state action masking for meaningful greedy/diversity metrics ---
            avail = avail_actions.to(q_values.device)  # [B, n_agents, n_actions]
            masked_q_values = q_values.masked_fill(avail == 0, -1e9)

            # Max feasible utility per agent
            q_i_max = masked_q_values.max(dim=2).values  # [B, n_agents]
            loss_info["q_i_max_mean"] = q_i_max.mean().item()
            loss_info["q_i_max_std"] = q_i_max.std().item()

            # Mixer vs sum baseline
            sum_q_i = best_q_values.sum(dim=1, keepdim=True)  # [B, 1]
            diff = q_total - sum_q_i  # [B, 1]

            loss_info["sum_q_i_mean"] = sum_q_i.mean().item()
            loss_info["sum_q_i_abs_mean"] = sum_q_i.abs().mean().item()

            loss_info["q_total_minus_sum_q_i_mean"] = diff.mean().item()
            loss_info["q_total_minus_sum_q_i_std"] = diff.std().item()
            loss_info["q_total_minus_sum_q_i_abs_mean"] = diff.abs().mean().item()

            # Scale-stable "how big is mixer output vs sum" using absolute means
            loss_info["q_total_abs_mean"] = q_total.abs().mean().item()
            loss_info["q_total_abs_over_sum_q_i_abs_mean"] = (
                q_total.abs().mean() / (sum_q_i.abs().mean() + 1e-6)
            ).item()

            # Correlation between q_total and sum_q_i (are they at least monotonic-ish?)
            sum_centered = sum_q_i - sum_q_i.mean()
            qt_centered = q_total - q_total.mean()
            corr = (sum_centered * qt_centered).mean() / (
                (sum_centered.pow(2).mean().sqrt() * qt_centered.pow(2).mean().sqrt())
                + 1e-6
            )
            loss_info["q_total_sum_q_i_corr"] = corr.item()

            # Availability constraints (next-state)
            loss_info["next_avail_action_frac"] = (
                next_avail_actions_tensors.float().mean().item()
            )
            loss_info["next_avail_actions_per_agent"] = (
                next_avail_actions_tensors.float().sum(dim=2).mean().item()
            )

            # Sanity: are actions in replay valid under current avail_actions?
            chosen_is_valid = avail.gather(2, actions_tensors.unsqueeze(-1)).squeeze(
                -1
            )  # [B, n_agents]
            loss_info["invalid_action_frac"] = (
                (chosen_is_valid == 0).float().mean().item()
            )
            loss_info["no_action_available_frac"] = (
                (avail.float().sum(dim=2) == 0).float().mean().item()
            )

            # Greedy action diversity (MASKED, feasible actions only)
            greedy_actions = masked_q_values.argmax(dim=2)  # [B, n_agents]
            entropies = []
            max_probs = []
            for ag in range(self.num_agents):
                counts = torch.bincount(
                    greedy_actions[:, ag], minlength=self.num_actions
                ).float()
                probs = counts / counts.sum().clamp(min=1.0)
                entropies.append(-(probs * (probs + 1e-12).log()).sum())
                max_probs.append(probs.max())

            loss_info["greedy_action_entropy_mean_agents"] = (
                torch.stack(entropies).mean().item()
            )
            loss_info["greedy_action_max_prob_mean_agents"] = (
                torch.stack(max_probs).mean().item()
            )

        return elementwise_loss, loss_info

    def train(
        self,
        memory_buffer: MARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {}

        self.learn_counter += 1

        training_step = episode_context.training_step

        self.epsilon = self.epsilon_scheduler.get_value(training_step)

        if len(memory_buffer) < self.batch_size:
            return {}

        # Use training_utils to sample and prepare batch
        (
            observation_tensor,
            actions_tensor,
            rewards_tensor,
            next_observation_tensor,
            dones_tensor,
            weights_tensor,
            _,  # extras ignored
            indices,
        ) = memory_sampler.sample(
            memory=memory_buffer,
            batch_size=self.batch_size,
            device=self.device,
            use_per_buffer=self.use_per_buffer,
            per_sampling_strategy=self.per_sampling_strategy,
            per_weight_normalisation=self.per_weight_normalisation,
            action_dtype=torch.long,  # DQN uses discrete actions
        )

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
            states_tensors=observation_tensor.global_state_tensor,
            next_states_tensors=next_observation_tensor.global_state_tensor,
            actions_tensors=actions_tensor,
            rewards_tensors=rewards_tensor,
            avail_actions=observation_tensor.avail_actions_tensor,
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
            torch.nn.utils.clip_grad_norm_(
                list(self.network.parameters()) + list(self.mixer.parameters()),
                max_norm=self.max_grad_norm,
            )

        self.network_optimiser.step()

        # Update target network - a tau of 1.0 equates to a hard update.
        if self.learn_counter % self.target_update_freq == 0:
            self.soft_update_params(self.network, self.target_network, self.tau)
            self.soft_update_params(self.mixer, self.target_mixer, self.tau)

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
