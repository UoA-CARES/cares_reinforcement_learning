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

    def _calculate_gae(
        self,
        rewards: torch.Tensor,  # [N] or [N,1]
        dones: torch.Tensor,  # [N] or [N,1] (1.0 if done else 0.0)
        values: torch.Tensor,  # [N]
        last_value: torch.Tensor,  # scalar tensor, V(s_T) for bootstrap
        gae_lambda: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rewards = rewards.view(-1)
        dones = dones.view(-1)
        values = values.view(-1)

        batch_size = rewards.shape[0]
        advantages = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        gae = torch.zeros((), dtype=torch.float32, device=self.device)
        next_value = last_value

        for t in reversed(range(batch_size)):
            mask = 1.0 - dones[t]  # 0 if terminal else 1
            delta = rewards[t] + self.gamma * mask * next_value - values[t]
            gae = delta + self.gamma * gae_lambda * mask * gae
            advantages[t] = gae
            next_value = values[t]

        returns = advantages + values
        return advantages, returns

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

        # Old log_probs + values stored at action time
        old_log_probs = [
            experience.train_data["log_prob"] for experience in sample.experiences
        ]
        old_log_probs_tensor = torch.tensor(
            np.asarray(old_log_probs), dtype=torch.float32, device=self.device
        )

        agent_ids = list(agent_states.keys())

        with torch.no_grad():
            values = self.central_critic(global_states)  # [T, num_agents]
            values = values.view(batch_size, self.num_agents)

            last_next_state = next_global_states[-1].unsqueeze(0)  # [1, 54]
            last_value = self.central_critic(last_next_state).view(-1)

            last_done = dones_tensor.reshape(-1)[-1].bool()
            last_value = last_value * (~last_done).to(last_value.dtype)

        rewards_tensor = rewards_tensor.view(batch_size, self.num_agents)

        advantages_all = torch.zeros((batch_size, self.num_agents), device=self.device)
        returns_all = torch.zeros((batch_size, self.num_agents), device=self.device)

        for i in range(self.num_agents):
            adv_i, ret_i = self._calculate_gae(
                rewards=rewards_tensor[:, i],
                dones=dones_tensor,
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

        for agent_idx, agent in enumerate(self.agent_networks):
            agent_name = agent_ids[agent_idx]
            states_i = agent_states[agent_name].view(batch_size, -1)  # [T, obs_dim]
            actions_i = actions_tensor[:, agent_idx, :].view(
                batch_size, -1
            )  # [T, act_dim]
            old_logp_i = old_log_probs_tensor[:, agent_idx].detach()  # [T]
            adv_i = advantages_all[:, agent_idx]  # [T]

            kl_early_stop = False
            sum_kl = 0.0
            num_mbs = 0

            for _ in range(self.updates_per_iteration):
                idx = torch.randperm(batch_size, device=self.device)
                for start in range(0, batch_size, mb_size):
                    mb = idx[start : start + mb_size]

                    s_mb = states_i[mb]
                    a_mb = actions_i[mb]
                    old_lp_mb = old_logp_i[mb]
                    adv_mb = adv_i[mb]

                    mean = agent.actor_net(s_mb)
                    dist = agent._dist(mean)
                    pre_tanh = agent._atanh(a_mb)
                    curr_lp = agent._squashed_log_prob(dist, pre_tanh)

                    log_ratio = curr_lp - old_lp_mb
                    ratios = torch.exp(log_ratio)

                    # KL early stop (per-agent)
                    if self.target_kl is not None:
                        with torch.no_grad():
                            approx_kl = (ratios - 1 - log_ratio).mean()
                            sum_kl += float(approx_kl.item())
                        if approx_kl > self.target_kl:
                            kl_early_stop = True
                            break

                    # PPO clipped objective
                    unclipped = ratios * adv_mb
                    clipped = (
                        torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip)
                        * adv_mb
                    )
                    policy_obj = torch.min(unclipped, clipped)
                    actor_loss = -policy_obj.mean()

                    # entropy bonus (base Gaussian)
                    entropy = dist.entropy().sum(dim=-1).mean()
                    actor_loss = actor_loss - self.entropy_coef * entropy

                    agent.actor_net_optimiser.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.actor_net.parameters()) + [agent.log_std],
                        self.max_grad_norm,
                    )
                    agent.actor_net_optimiser.step()

                    # project log_std after step
                    with torch.no_grad():
                        agent.log_std.clamp_(agent.min_log_std, agent.max_log_std)

                    num_mbs += 1

                if kl_early_stop:
                    break

        for _ in range(self.updates_per_iteration):  # or fewer, see note below
            idx = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, mb_size):
                mb = idx[start : start + mb_size]

                v_pred = self.central_critic(global_states[mb])  # [mb, N]
                v_targ = returns_all[mb]  # [mb, N]

                critic_loss = F.mse_loss(v_pred, v_targ)

                self.central_critic_optimiser.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.central_critic.parameters(), self.max_grad_norm
                )
                self.central_critic_optimiser.step()

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
