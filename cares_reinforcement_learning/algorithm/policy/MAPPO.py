import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.memory.memory_sampler as memory_sampler
from cares_reinforcement_learning.algorithm.algorithm import Algorithm
from cares_reinforcement_learning.algorithm.policy.PPO import PPO
from cares_reinforcement_learning.networks.MAPPO import Critic
from cares_reinforcement_learning.memory.memory_buffer import MARLMemoryBuffer
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import (
    MARLObservation,
    SARLObservation,
)
from cares_reinforcement_learning.util.configurations import MAPPOConfig


class MAPPO(Algorithm[MARLObservation, list[np.ndarray], MARLMemoryBuffer]):
    def __init__(
        self,
        agents: list[PPO],
        central_critic: Critic,
        config: MAPPOConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        self.agent_networks = agents
        self.num_agents = len(agents)

        self.minibatch_size = config.minibatch_size
        self.updates_per_iteration = config.updates_per_iteration
        self.eps_clip = config.eps_clip
        self.target_kl = config.target_kl
        self.entropy_coef = config.entropy_coef
        self.max_grad_norm = config.max_grad_norm

        self.gae_lambda = config.gae_lambda

        # For MAPPO, we assume a shared critic architecture where all agents share the same critic network.
        self.central_critic = central_critic.to(device)
        self.central_critic_optimiser = torch.optim.Adam(
            self.central_critic.parameters(), lr=config.critic_lr
        )

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
        log_probs = []

        for i, agent in enumerate(self.agent_networks):
            agent_name = agent_ids[i]  # consistent ordering in dict
            obs_i = agent_states[agent_name]
            avail_i = avail_actions[i]

            agent_observation = SARLObservation(
                vector_state=obs_i,
                avail_actions=avail_i,
            )

            agent_sample = agent.act(
                agent_observation, evaluation, calculate_value=False
            )
            actions.append(agent_sample.action)
            log_probs.append(agent_sample.extras["log_prob"])

        return ActionSample(
            action=actions, source="policy", extras={"log_prob": log_probs}
        )

    def train_policy(
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
        sample = memory_buffer.flush()
        batch_size = len(sample.experiences)

        if batch_size == 0:
            return {}

        # Convert to tensors using helper method (no next_states needed for PPO, so pass dummy data)
        (
            observation_tensor,
            actions_tensor,
            rewards_tensor,
            next_observation_tensor,
            dones_tensor,
            _,  # weights not needed
            _,
        ) = memory_sampler.sample_to_tensors(sample, self.device)

        global_states = observation_tensor.global_state_tensor
        next_global_states = next_observation_tensor.global_state_tensor

        agent_states = observation_tensor.agent_states_tensor

        # IMPORTANT: dones are per-agent for generic case
        dones = dones_tensor.squeeze(-1).float()  # [T, N]

        # Old log_probs + values stored at action time
        old_log_probs = [
            experience.train_data["log_prob"] for experience in sample.experiences
        ]
        old_log_probs_tensor = torch.tensor(
            np.asarray(old_log_probs), dtype=torch.float32, device=self.device
        )

        agent_ids = list(agent_states.keys())

        # ---------- Central critic values ----------
        with torch.no_grad():
            values = self.central_critic(global_states)  # [T, num_agents]
            values = values.view(batch_size, self.num_agents)

            last_next_state = next_global_states[-1].unsqueeze(0)  # [1, 54]
            last_value = self.central_critic(last_next_state).view(-1)

            last_done = dones[-1].bool()
            last_value = last_value * (~last_done).to(last_value.dtype)

        rewards_tensor = rewards_tensor.view(batch_size, self.num_agents)

        advantages_all = torch.zeros((batch_size, self.num_agents), device=self.device)
        returns_all = torch.zeros((batch_size, self.num_agents), device=self.device)

        for i in range(self.num_agents):
            adv_i, ret_i = self.agent_networks[i]._calculate_gae(
                rewards=rewards_tensor[:, i],
                dones=dones[:, i],
                values=values[:, i],
                last_value=last_value[i],
                gae_lambda=self.gae_lambda,
            )
            advantages_all[:, i] = adv_i
            returns_all[:, i] = ret_i

        adv_flat = advantages_all.view(-1)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std(unbiased=False) + 1e-8)
        advantages_all = adv_flat.view(batch_size, self.num_agents)

        mb_size = min(self.minibatch_size, batch_size)

        # Track per-agent KL early-stop (actor-only)
        agent_kl_early_stop = [False] * self.num_agents
        agent_kl_sum = [0.0] * self.num_agents
        agent_kl_n = [0] * self.num_agents

        # Track critic loss
        critic_loss_sum = 0.0
        num_critic_mb = 0

        # ---------- Epochs / minibatches ----------
        for _ in range(self.updates_per_iteration):
            idx = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, mb_size):
                mb = idx[start : start + mb_size]

                # ----- Actor updates (same mb across agents) -----
                for agent_idx, agent in enumerate(self.agent_networks):
                    if agent_kl_early_stop[agent_idx]:
                        continue  # this agent already KL-stopped this rollout

                    agent_name = agent_ids[agent_idx]
                    states_mb = agent_states[agent_name][mb]  # [mb, obs_dim]
                    actions_mb = actions_tensor[mb, agent_idx, :]  # [mb, act_dim]
                    old_logp_mb = old_log_probs_tensor[mb, agent_idx]  # [mb]
                    advantages_mb = advantages_all[mb, agent_idx]  # [mb]

                    # Use the new PPO helper (includes KL pre-check + may skip update)
                    kl_early_stop, actor_info = agent.update_actor_minibatch(
                        states_mb=states_mb,
                        actions_mb=actions_mb,
                        old_logp_mb=old_logp_mb,
                        advantages_mb=advantages_mb,
                    )

                    # Accumulate KL stats if present
                    if self.target_kl is not None and "approx_kl" in actor_info:
                        agent_kl_sum[agent_idx] += float(actor_info["approx_kl"])
                        agent_kl_n[agent_idx] += 1

                    # If the helper skipped the update due to KL, stop this agent for remainder
                    agent_kl_early_stop[agent_idx] = kl_early_stop

                # ----- Central critic update (same mb indices) -----
                v_pred = self.central_critic(global_states[mb])  # [mb, N]
                v_targ = returns_all[mb]  # [mb, N]
                critic_loss = F.mse_loss(v_pred, v_targ)

                self.central_critic_optimiser.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.central_critic.parameters(), self.max_grad_norm
                )
                self.central_critic_optimiser.step()

                critic_loss_sum += float(critic_loss.item())
                num_critic_mb += 1

        # ---------- Logging ----------
        info["critic_loss"] = critic_loss_sum / max(num_critic_mb, 1)

        if self.target_kl is not None:
            for i in range(self.num_agents):
                if agent_kl_n[i] > 0:
                    info[f"agent{i}_approx_kl"] = agent_kl_sum[i] / agent_kl_n[i]
                info[f"agent{i}_kl_early_stop"] = int(agent_kl_early_stop[i])

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
