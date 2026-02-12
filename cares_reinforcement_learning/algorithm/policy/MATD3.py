"""
MATD3 (Multi-Agent TD3) implementation notes
--------------------------------------------

This algorithm extends MADDPG with TD3 improvements:
twin critics, delayed policy updates, and target policy smoothing.

Replay sampling:
- A single minibatch is sampled per training iteration and reused across agents.
- This preserves unbiased updates while reducing variance and keeping joint
  transitions consistent for centralized critics.
- TD3 introduces explicit variance-reduction mechanisms (twin critics and
  target smoothing), making shared minibatch updates more stable than the
  original MADDPG per-agent sampling scheme.

Critic updates:
- Twin critics are trained using TD3-style targets with target policy smoothing.
- Noise is applied only to NEXT actions for critic targets to reduce
  overestimation bias.

Actor updates:
- Actors are deterministic and updated with a delayed frequency.
- When updating agent i, only agent i's action is replaced with the current
  actor output; other agents' actions come from the replay buffer.
- This mirrors MADDPG and avoids unnecessary coupling of agent updates.

No joint action resampling:
- TD3's stochasticity is confined to target policy smoothing.
- Resampling other agents' current actions is unnecessary and can increase
  variance without benefit for deterministic policy gradients.
"""

import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.memory.memory_sampler as memory_sampler
from cares_reinforcement_learning.algorithm.algorithm import Algorithm
from cares_reinforcement_learning.algorithm.policy.TD3 import TD3
from cares_reinforcement_learning.memory.memory_buffer import MARLMemoryBuffer
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import (
    MARLObservation,
    SARLObservation,
)
from cares_reinforcement_learning.util.configurations import MATD3Config


class MATD3(Algorithm[MARLObservation, list[np.ndarray], MARLMemoryBuffer]):
    def __init__(
        self,
        agents: list[TD3],
        config: MATD3Config,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        self.agent_networks = agents
        self.num_agents = len(agents)

        self.gamma = config.gamma
        self.tau = config.tau

        self.policy_update_freq = config.policy_update_freq

        self.policy_noise = config.policy_noise
        self.policy_noise_clip = config.policy_noise_clip

        self.max_grad_norm = config.max_grad_norm

        self.learn_counter = 0

    def act(
        self,
        observation: MARLObservation,
        evaluation: bool = False,
    ) -> ActionSample[list[np.ndarray]]:
        agent_states = observation.agent_states
        avail_actions = observation.avail_actions

        agent_ids = list(agent_states.keys())
        actions = []

        for i, agent in enumerate(self.agent_networks):
            agent_name = agent_ids[i]  # consistent ordering in dict
            obs_i = agent_states[agent_name]
            avail_i = avail_actions[i]

            agent_observation = SARLObservation(
                vector_state=obs_i,
                avail_actions=avail_i,
            )

            agent_sample = agent.act(agent_observation, evaluation)
            actions.append(agent_sample.action)

        return ActionSample(action=actions, source="policy")

    def _update_critic(
        self,
        agent: TD3,
        global_states: torch.Tensor,
        joint_actions: torch.Tensor,  # (B, N * act_dim) from replay
        rewards_i: torch.Tensor,  # (B, 1)
        next_global_states: torch.Tensor,
        next_actions_tensor: torch.Tensor,  # (B, N, act_dim) from target actors
        dones_i: torch.Tensor,
    ):
        # --- Step 1: build next joint actions ---
        next_joint_actions = next_actions_tensor.view(next_actions_tensor.size(0), -1)

        # --- Step 2: TD target ---
        with torch.no_grad():
            target_q_values_one, target_q_values_two = agent.target_critic_net(
                next_global_states, next_joint_actions
            )
            target_q = torch.min(target_q_values_one, target_q_values_two)
            q_target = rewards_i + self.gamma * (1 - dones_i) * target_q

        # --- Step 3: critic regression on *current* joint_actions (unperturbed) ---
        q_values_one, q_values_two = agent.critic_net(global_states, joint_actions)

        critic_loss_one = F.mse_loss(q_values_one, q_target)
        critic_loss_two = F.mse_loss(q_values_two, q_target)

        critic_loss_total = critic_loss_one + critic_loss_two

        agent.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                agent.critic_net.parameters(), self.max_grad_norm
            )

        agent.critic_net_optimiser.step()

        return {
            "critic_loss_one": critic_loss_one.item(),
            "critic_loss_two": critic_loss_two.item(),
            "critic_loss_total": critic_loss_total.item(),
        }

    def _update_actor(
        self,
        agent: TD3,
        agent_index: int,
        obs_tensors: dict[str, torch.Tensor],
        global_states: torch.Tensor,
        actions_tensor: torch.Tensor,  # (B, N, act_dim)
    ):
        """
        Paper-faithful MATD3 actor update:
        - For j ≠ agent_index: use replay-buffer actions
        - For j == agent_index: use current actor output
        """

        agent_ids = list(obs_tensors.keys())
        batch_size = global_states.shape[0]

        # ---------------------------------------------------------
        # Step 1: Start from replay-buffer joint actions
        #         actions_all: (B, N, A)
        # ---------------------------------------------------------
        actions_all = actions_tensor.clone()  # clone so we can overwrite

        # ---------------------------------------------------------
        # Step 2: Replace ONLY agent_i action with differentiable action
        # ---------------------------------------------------------
        obs_i = obs_tensors[agent_ids[agent_index]]  # (B, obs_dim_i)
        pred_action_i = agent.actor_net(obs_i)  # differentiable

        actions_all[:, agent_index, :] = pred_action_i  # keep others from buffer

        # ---------------------------------------------------------
        # Step 4: Compute actor loss: -Q_i(x, a_1,...,a_i,...,a_N)
        # ---------------------------------------------------------
        joint_actions_flat = actions_all.reshape(batch_size, -1)
        q_val, _ = agent.critic_net(global_states, joint_actions_flat)

        # regularization as in TF code
        reg = (pred_action_i**2).mean() * 1e-3

        actor_loss = -q_val.mean() + reg

        # ---------------------------------------------------------
        # Step 5: Backprop
        # ---------------------------------------------------------
        agent.actor_net_optimiser.zero_grad()
        actor_loss.backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                agent.actor_net.parameters(), self.max_grad_norm
            )

        agent.actor_net_optimiser.step()

        return {"actor_loss": actor_loss.item()}

    def train(
        self,
        memory_buffer: MARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:

        self.learn_counter += 1

        info: dict[str, Any] = {}

        # ---------------------------------------------------------
        # Sample ONCE for all agents (recommended for TD3/SAC)
        # Shared minibatch: We draw one minibatch per training iteration and reuse it across agent updates.
        # This preserves an unbiased estimator of each update while reducing sampling-induced variance and
        # keeping joint transitions consistent for centralized critics.
        # ---------------------------------------------------------
        (
            observation_tensor,
            actions_tensor,
            rewards_tensor,
            next_observation_tensor,
            dones_tensor,
            _,
            _,
            indices,
        ) = memory_sampler.sample(
            memory=memory_buffer,
            batch_size=self.batch_size,
            device=self.device,
            use_per_buffer=0,
        )

        batch_size = len(indices)

        global_states = observation_tensor.global_state_tensor
        next_global_states = next_observation_tensor.global_state_tensor

        agent_states = observation_tensor.agent_states_tensor
        next_agent_states = next_observation_tensor.agent_states_tensor

        agent_ids = list(agent_states.keys())

        # Flatten replay-buffer joint actions (used for critic current Q)
        joint_actions = actions_tensor.reshape(batch_size, -1)

        # ---------------------------------------------------------
        # Build NEXT actions using TARGET actors (clean)
        # ---------------------------------------------------------
        next_actions = []
        for agent, agent_id in zip(self.agent_networks, agent_ids):
            obs_next = next_agent_states[agent_id]
            next_actions.append(agent.target_actor_net(obs_next))

        # (B, N, act_dim)
        next_actions_tensor = torch.stack(next_actions, dim=1)

        # ---------------------------------------------------------
        # TD3 TARGET POLICY SMOOTHING (ONCE)
        # ---------------------------------------------------------
        # This affects ONLY critic targets
        noise = torch.randn_like(next_actions_tensor) * self.policy_noise
        noise = noise.clamp(-self.policy_noise_clip, self.policy_noise_clip)

        # assumes tanh policy -> [-1, 1]
        next_actions_noisy = (next_actions_tensor + noise).clamp(-1.0, 1.0)

        # ---------------------------------------------------------
        # CRITIC UPDATES (every step)
        # ---------------------------------------------------------
        for agent_index, agent in enumerate(self.agent_networks):
            rewards_i = rewards_tensor[:, agent_index]
            dones_i = dones_tensor[:, agent_index]

            critic_info = self._update_critic(
                agent=agent,
                global_states=global_states,
                joint_actions=joint_actions,
                rewards_i=rewards_i,
                next_global_states=next_global_states,
                next_actions_tensor=next_actions_noisy,  # <-- noisy version
                dones_i=dones_i,
            )
            for key, value in critic_info.items():
                info[f"agent_{agent_index}_{key}"] = value

        # ---------------------------------------------------------
        # ACTOR + TARGET UPDATES (DELAYED — TD3)
        # ---------------------------------------------------------
        if self.learn_counter % self.policy_update_freq == 0:
            for agent_index, agent in enumerate(self.agent_networks):
                actor_info = self._update_actor(
                    agent=agent,
                    agent_index=agent_index,
                    obs_tensors=agent_states,
                    global_states=global_states,
                    actions_tensor=actions_tensor,
                )
                for key, value in actor_info.items():
                    info[f"agent_{agent_index}_{key}"] = value

            # TD3: target networks updated on SAME cadence as actor
            for agent in self.agent_networks:
                agent.update_target_networks()

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        for i, agent in enumerate(self.agent_networks):
            agent_filepath = os.path.join(filepath, f"agent_{i}")
            agent_filename = f"{filename}_agent_{i}_checkpoint"
            agent.save_models(agent_filepath, agent_filename)

        logging.info("models and optimisers have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        for i, agent in enumerate(self.agent_networks):
            agent_filepath = os.path.join(filepath, f"agent_{i}")
            agent_filename = f"{filename}_agent_{i}_checkpoint"
            agent.load_models(agent_filepath, agent_filename)

        logging.info("models and optimisers have been loaded...")
