"""
MAPPO (Multi-Agent Proximal Policy Optimization) implementation notes
--------------------------------------------------------------------

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
from cares_reinforcement_learning.algorithm.algorithm import Algorithm
from cares_reinforcement_learning.algorithm.policy.PPO import PPO
from cares_reinforcement_learning.memory.memory_buffer import MARLMemoryBuffer
from cares_reinforcement_learning.networks.MAPPO import Critic
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import (
    MARLObservation,
    SARLObservation,
)
from cares_reinforcement_learning.util.configurations import MAPPOConfig
from cares_reinforcement_learning.util.helpers import EpsilonScheduler


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

        self.epsilon_scheduler = EpsilonScheduler(
            start_epsilon=config.entropy_start,
            end_epsilon=config.entropy_end,
            decay_steps=config.entropy_decay,
        )
        # initial entropy coefficient
        self.entropy_coef = self.epsilon_scheduler.get_epsilon(0)

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

        info: dict[str, Any] = {}

        self.entropy_coef = self.epsilon_scheduler.get_epsilon(
            episode_context.training_step
        )

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

        # Track critic loss
        critic_loss_sum = 0.0
        num_critic_mb = 0

        agent_sums = [
            {
                k: 0.0
                for k in [
                    "actor_loss",
                    "entropy",
                    "approx_kl",
                    "clip_frac",
                    "ratio_mean",
                    "ratio_std",
                    "action_sat_rate",
                    "u_abs_mean",
                    "u_abs_max",
                    "log_ratio_mean",
                    "log_ratio_std",
                    "log_ratio_max_abs",
                ]
            }
            for _ in range(self.num_agents)
        ]
        agent_max_kl = [0.0 for _ in range(self.num_agents)]
        # minibatches that actually updated (for averaging stats)
        agent_updates = [0 for _ in range(self.num_agents)]

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
                        entropy_coef=self.entropy_coef,
                    )

                    # Accumulate only if update happened
                    if not kl_early_stop:
                        agent_updates[agent_idx] += 1
                        for k in agent_sums[agent_idx].keys():
                            agent_sums[agent_idx][k] += float(actor_info[k])

                    # Track max KL seen regardless (if target_kl is enabled, approx_kl is meaningful)
                    if self.target_kl is not None:
                        agent_max_kl[agent_idx] = max(
                            agent_max_kl[agent_idx], float(actor_info["approx_kl"])
                        )

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

        with torch.no_grad():
            # returns vs values
            td_err = returns_all - values  # [T, N]

            for i in range(self.num_agents):
                info[f"agent{i}_adv_mean"] = float(advantages_all[:, i].mean().item())
                info[f"agent{i}_adv_std"] = float(
                    advantages_all[:, i].std(unbiased=False).item()
                )

                info[f"agent{i}_ret_mean"] = float(returns_all[:, i].mean().item())
                info[f"agent{i}_ret_std"] = float(
                    returns_all[:, i].std(unbiased=False).item()
                )

                info[f"agent{i}_v_mean"] = float(values[:, i].mean().item())
                info[f"agent{i}_v_std"] = float(values[:, i].std(unbiased=False).item())

                info[f"agent{i}_td_mean"] = float(td_err[:, i].mean().item())
                info[f"agent{i}_td_std"] = float(
                    td_err[:, i].std(unbiased=False).item()
                )
                info[f"agent{i}_td_mae"] = float(td_err[:, i].abs().mean().item())

                y = returns_all[:, i]
                yhat = values[:, i]
                var_y = torch.var(y, unbiased=False)
                ev = 1.0 - torch.var(y - yhat, unbiased=False) / (var_y + 1e-8)
                info[f"agent{i}_explained_var"] = float(ev.item())

                denom = max(agent_updates[i], 1)

                info[f"agent{i}_actor_updates"] = int(agent_updates[i])
                info[f"agent{i}_kl_early_stop"] = int(agent_kl_early_stop[i])

                for k, v in agent_sums[i].items():
                    info[f"agent{i}_{k}"] = v / denom

                if self.target_kl is not None:
                    info[f"agent{i}_max_kl_seen"] = agent_max_kl[i]

        stopped = sum(int(x) for x in agent_kl_early_stop)
        info["num_agents_kl_stopped"] = stopped
        info["any_agent_kl_stopped"] = int(stopped > 0)
        info["all_agents_kl_stopped"] = int(stopped == self.num_agents)

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        for i, agent in enumerate(self.agent_networks):
            agent_filepath = os.path.join(filepath, f"agent_{i}")
            agent_filename = f"{filename}_agent_{i}_checkpoint"
            agent.save_models(agent_filepath, agent_filename)

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
        for i, agent in enumerate(self.agent_networks):
            agent_filepath = os.path.join(filepath, f"agent_{i}")
            agent_filename = f"{filename}_agent_{i}_checkpoint"
            agent.load_models(agent_filepath, agent_filename)

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
