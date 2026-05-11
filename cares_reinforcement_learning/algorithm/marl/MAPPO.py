"""
MAPPO (Multi-Agent Proximal Policy Optimization) implementation notes
--------------------------------------------------------------------

Original Paper: https://arxiv.org/abs/2103.01955

This algorithm extends PPO to the multi-agent setting using decentralized actors
and a centralized value function (critic). Policies are executed per-agent from
local observations, while value estimation can condition on global state.

Data collection (on-policy):
- Rollouts are collected on-policy using the current actor parameters.
- For each timestep, we store per-agent: observation, action (network action space),
  reward, done, and the log-probability under the behavior policy (old_logp).
- PPO assumes the batch is on-policy; mixing data from multiple policy versions
  in the same update can inflate KL and stall learning.

Replay / sampling:
- MAPPO is on-policy: we "flush" the rollout buffer once per iteration.
- A single permutation of indices is generated per epoch and reused across agents
  so all agents update on aligned joint transitions (important for centralized critics).

Central critic (value) updates:
- The centralized critic takes the global state and outputs V(s) for each agent
  (shape [T, num_agents]) or equivalent multi-head value estimates.
- Targets are computed using Generalized Advantage Estimation (GAE) per agent:
    delta_t = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
    A_t = discounted sum of deltas with gae_lambda
    R_t = A_t + V(s_t)
- The critic is trained with an MSE loss between V(s_t) and returns R_t.

Actor updates (decentralized PPO):
- Each agent has its own stochastic policy pi_i(a_i | o_i) with tanh-squashed actions.
- PPO updates use importance sampling ratios computed from stored old log-probs:
    ratio = exp(logp_curr - logp_old)
- The clipped surrogate objective is optimized per agent:
    L = -E[min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)] - entropy_coef * entropy
- Only agent i's actor parameters receive gradients during agent i's update.

KL monitoring / early stopping:
- We optionally track approximate KL divergence per minibatch and can stop further
  actor updates for an agent if KL exceeds target_kl (stability safeguard).
- If KL spikes immediately at the start of an epoch, it typically indicates an
  on-policy violation or a mismatch between stored actions/log-probs and the
  update-time log-prob computation.

Rationale:
- PPO is on-policy and relies on per-sample log-probs from the behavior policy;
  MAPPO preserves this while enabling coordination via a centralized value function.
- Using a centralized critic reduces variance and improves credit assignment in
  cooperative tasks like MPE simple_spread, while keeping execution decentralized.
"""

import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.memory.memory_sampler as memory_sampler
from cares_reinforcement_learning.algorithm.algorithm import MARLAlgorithm
from cares_reinforcement_learning.algorithm.configurations import MAPPOConfig
from cares_reinforcement_learning.algorithm.policy.PPO import PPO
from cares_reinforcement_learning.algorithm.schedulers import ExponentialScheduler
from cares_reinforcement_learning.memory.memory_buffer import MARLMemoryBuffer
from cares_reinforcement_learning.networks.MAPPO import Critic
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import (
    MARLObservation,
    SARLObservation,
)


class MAPPO(MARLAlgorithm[dict[str, np.ndarray]]):
    def __init__(
        self,
        agents: dict[str, PPO],
        central_critic: Critic,
        config: MAPPOConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        self.agent_networks = agents
        self.agent_ids = list(self.agent_networks.keys())
        self.num_agents = len(agents)

        self.minibatch_size = config.minibatch_size
        self.updates_per_iteration = config.updates_per_iteration

        self.entropy_scheduler = ExponentialScheduler(
            start_value=config.entropy_start,
            end_value=config.entropy_end,
            decay_steps=config.entropy_decay,
        )
        # initial entropy coefficient
        self.entropy_coef = self.entropy_scheduler.get_value(0)

        self.target_kl = config.target_kl

        self.max_grad_norm = config.max_grad_norm

        self.gae_lambda = config.gae_lambda

        # For MAPPO, we assume a shared critic architecture where all agents share the same critic network.
        self.central_critic = central_critic.to(device)
        self.central_critic_optimiser = torch.optim.Adam(
            self.central_critic.parameters(), lr=config.critic_lr
        )

    def act(
        self,
        observation: MARLObservation,
        evaluation: bool = False,
    ) -> ActionSample[dict[str, np.ndarray]]:
        agent_states = observation.agent_states
        available_actions = observation.available_actions

        actions = {}
        log_probs = {}

        for agent_name, agent_network in self.agent_networks.items():
            obs_i = agent_states[agent_name]
            avail_i = available_actions[agent_name]

            agent_observation = SARLObservation(
                vector_state=obs_i,
                available_actions=avail_i,
            )

            agent_sample = agent_network.act(
                agent_observation, evaluation, calculate_value=False
            )
            actions[agent_name] = agent_sample.action
            log_probs[agent_name] = agent_sample.extras["log_prob"]

        return ActionSample(
            action=actions, source="policy", extras={"log_prob": log_probs}
        )

    def train(
        self,
        memory_buffer: MARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:

        info: dict[str, Any] = {}

        self.entropy_coef = self.entropy_scheduler.get_value(
            episode_context.training_step
        )

        # ---------------------------------------------------------
        # Sample ONCE for all agents (recommended for PPO/TD3/SAC)
        # Shared minibatch: We draw one minibatch per training iteration and reuse it across agent updates.
        # This preserves an unbiased estimator of each update while reducing sampling-induced variance and
        # keeping joint transitions consistent for centralized critics.
        # ---------------------------------------------------------
        sample = memory_buffer.flush()
        batch_size = len(sample.experiences)

        if batch_size == 0:
            return {}

        # Convert to tensors using helper method (no next_states needed for PPO, so pass dummy data)
        sample_tensor = memory_sampler.sample_to_tensors(sample, self.device)

        global_states = sample_tensor.observation.global_state
        next_global_states = sample_tensor.next_observation.global_state

        agent_states = sample_tensor.observation.agent_states

        actions_tensor_dict = sample_tensor.action
        rewards_tensor_dict = sample_tensor.reward
        dones_tensor_dict = sample_tensor.done

        # actions_tensor: (T, N, act_dim)
        actions_tensor = torch.stack(
            [actions_tensor_dict[a] for a in self.agent_ids],
            dim=1,
        )

        # IMPORTANT: dones are per-agent for generic case
        # Stack per-agent rewards/dones into canonical order
        # rewards_tensor: (T, N)
        rewards_tensor = torch.stack(
            [rewards_tensor_dict[a] for a in self.agent_ids],
            dim=1,
        ).squeeze(-1)

        # dones: (T, N)
        dones = (
            torch.stack(
                [dones_tensor_dict[a] for a in self.agent_ids],
                dim=1,
            )
            .squeeze(-1)
            .float()
        )

        # Old log_probs + values stored at action time
        old_log_probs = [
            [experience.train_data["log_prob"][agent_id] for agent_id in self.agent_ids]
            for experience in sample.experiences
        ]

        old_log_probs_tensor = torch.tensor(
            np.asarray(old_log_probs), dtype=torch.float32, device=self.device
        )

        # ---------- Central critic values ----------
        with torch.no_grad():
            values = self.central_critic(global_states)  # [T, num_agents]
            values = values.view(batch_size, self.num_agents)

            last_next_state = next_global_states[-1].unsqueeze(0)  # [1, 54]
            last_value = self.central_critic(last_next_state).view(-1)

            last_done = dones[-1].bool()
            last_value = last_value * (~last_done).to(last_value.dtype)

        advantages_all = torch.zeros((batch_size, self.num_agents), device=self.device)
        returns_all = torch.zeros((batch_size, self.num_agents), device=self.device)

        for agent_index, agent_id in enumerate(self.agent_ids):
            adv_i, ret_i = self.agent_networks[agent_id]._calculate_gae(
                rewards=rewards_tensor[:, agent_index],
                dones=dones[:, agent_index],
                values=values[:, agent_index],
                last_value=last_value[agent_index],
                gae_lambda=self.gae_lambda,
            )

            advantages_all[:, agent_index] = adv_i
            returns_all[:, agent_index] = ret_i

        adv_flat = advantages_all.view(-1)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std(unbiased=False) + 1e-8)
        advantages_all = adv_flat.view(batch_size, self.num_agents)

        mb_size = min(self.minibatch_size, batch_size)

        # Track per-agent KL early-stop (actor-only)
        agent_kl_early_stop = {agent_id: False for agent_id in self.agent_ids}

        # Track critic loss
        critic_loss_sum = 0.0
        num_critic_mb = 0

        agent_actor_sums: dict[str, dict[str, float]] = {
            agent_id: {} for agent_id in self.agent_ids
        }
        agent_max_kl = {agent_id: 0.0 for agent_id in self.agent_ids}
        # minibatches that actually updated (for averaging stats)
        agent_updates = {agent_id: 0 for agent_id in self.agent_ids}

        # ---------- Epochs / minibatches ----------
        for _ in range(self.updates_per_iteration):
            idx = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, mb_size):
                mb = idx[start : start + mb_size]

                # ----- Actor updates (same mb across agents) -----
                for agent_name, agent_network in self.agent_networks.items():
                    agent_idx = self.agent_ids.index(agent_name)
                    if agent_kl_early_stop[agent_name]:
                        continue  # this agent already KL-stopped this rollout

                    states_mb = agent_states[agent_name][mb]  # [mb, obs_dim]
                    actions_mb = actions_tensor[mb, agent_idx, :]  # [mb, act_dim]
                    old_logp_mb = old_log_probs_tensor[mb, agent_idx]  # [mb]
                    advantages_mb = advantages_all[mb, agent_idx]  # [mb]

                    # Use the new PPO helper (includes KL pre-check + may skip update)
                    kl_early_stop, actor_info = agent_network.update_actor_minibatch(
                        states_mb=states_mb,
                        actions_mb=actions_mb,
                        old_logp_mb=old_logp_mb,
                        advantages_mb=advantages_mb,
                        entropy_coef=self.entropy_coef,
                    )

                    agent_updates[agent_name] += 1
                    for k in actor_info.keys():
                        if k not in agent_actor_sums[agent_name]:
                            agent_actor_sums[agent_name][k] = 0.0
                        agent_actor_sums[agent_name][k] += float(actor_info[k])

                    # Track max KL seen regardless (if target_kl is enabled, approx_kl is meaningful)
                    if self.target_kl is not None:
                        agent_max_kl[agent_name] = max(
                            agent_max_kl[agent_name], float(actor_info["approx_kl"])
                        )

                    # If the helper skipped the update due to KL, stop this agent for remainder
                    agent_kl_early_stop[agent_name] = kl_early_stop

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
        info["entropy_coef"] = self.entropy_coef

        with torch.no_grad():
            # returns vs values
            td_err = returns_all - values  # [T, N]

            for i, agent_name in enumerate(self.agent_ids):
                info[f"{agent_name}_adv_mean"] = float(
                    advantages_all[:, i].mean().item()
                )
                info[f"{agent_name}_adv_std"] = float(
                    advantages_all[:, i].std(unbiased=False).item()
                )

                info[f"{agent_name}_ret_mean"] = float(returns_all[:, i].mean().item())
                info[f"{agent_name}_ret_std"] = float(
                    returns_all[:, i].std(unbiased=False).item()
                )

                info[f"{agent_name}_v_mean"] = float(values[:, i].mean().item())
                info[f"{agent_name}_v_std"] = float(
                    values[:, i].std(unbiased=False).item()
                )

                info[f"{agent_name}_td_mean"] = float(td_err[:, i].mean().item())
                info[f"{agent_name}_td_std"] = float(
                    td_err[:, i].std(unbiased=False).item()
                )
                info[f"{agent_name}_td_mae"] = float(td_err[:, i].abs().mean().item())

                y = returns_all[:, i]
                yhat = values[:, i]
                var_y = torch.var(y, unbiased=False)
                ev = 1.0 - torch.var(y - yhat, unbiased=False) / (var_y + 1e-8)
                info[f"{agent_name}_explained_var"] = float(ev.item())

                denom = max(agent_updates[agent_name], 1)

                info[f"{agent_name}_actor_updates"] = int(agent_updates[agent_name])
                info[f"{agent_name}_kl_early_stop"] = int(
                    agent_kl_early_stop[agent_name]
                )

                for k, v in agent_actor_sums[agent_name].items():
                    info[f"{agent_name}_{k}"] = v / denom

                if self.target_kl is not None:
                    info[f"{agent_name}_max_kl_seen"] = agent_max_kl[agent_name]

        for k in agent_actor_sums[self.agent_ids[0]].keys():
            values = [info[f"{agent_name}_{k}"] for agent_name in self.agent_ids]
            info[f"mean_{k}"] = float(np.mean(values))

        for metric in [
            "ret_mean",
            "ret_std",
            "v_mean",
            "v_std",
            "td_mae",
            "explained_var",
        ]:
            values = [info[f"{agent_name}_{metric}"] for agent_name in self.agent_ids]
            info[f"mean_{metric}"] = float(np.mean(values))
            info[f"std_{metric}"] = float(np.std(values))
            info[f"max_{metric}"] = float(np.max(values))
            info[f"min_{metric}"] = float(np.min(values))

        stopped = sum(int(x) for x in agent_kl_early_stop.values())
        info["num_agents_kl_stopped"] = stopped
        info["any_agent_kl_stopped"] = int(stopped > 0)
        info["all_agents_kl_stopped"] = int(stopped == self.num_agents)

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        for agent_name, agent_network in self.agent_networks.items():
            agent_filepath = os.path.join(filepath, f"{agent_name}")
            agent_filename = f"{filename}_agent_{agent_name}_checkpoint"
            agent_network.save_models(agent_filepath, agent_filename)

        # Save central critic
        critic_filepath = os.path.join(filepath, "central_critic")
        if not os.path.exists(critic_filepath):
            os.makedirs(critic_filepath)
        critic_checkpoint = {
            "critic_state_dict": self.central_critic.state_dict(),
            "critic_optimizer_state_dict": self.central_critic_optimiser.state_dict(),
        }
        torch.save(
            critic_checkpoint,
            os.path.join(critic_filepath, f"{filename}_central_critic_checkpoint.pt"),
        )

        logging.info("models and optimisers have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        for agent_name, agent_network in self.agent_networks.items():
            agent_filepath = os.path.join(filepath, f"{agent_name}")
            agent_filename = f"{filename}_agent_{agent_name}_checkpoint"
            agent_network.load_models(agent_filepath, agent_filename)

        # Load central critic
        critic_filepath = os.path.join(filepath, "central_critic")
        critic_checkpoint = torch.load(
            os.path.join(critic_filepath, f"{filename}_central_critic_checkpoint.pt")
        )
        self.central_critic.load_state_dict(critic_checkpoint["critic_state_dict"])
        self.central_critic_optimiser.load_state_dict(
            critic_checkpoint["critic_optimizer_state_dict"]
        )

        logging.info("models and optimisers have been loaded...")
