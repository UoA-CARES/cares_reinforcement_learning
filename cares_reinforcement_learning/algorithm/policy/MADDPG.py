"""
MADDPG (Multi-Agent DDPG) implementation notes
---------------------------------------------

Original Paper: https://arxiv.org/pdf/1706.02275

Original Code (TensorFlow): https://github.com/openai/maddpg/tree/master

Replay sampling:
- Each agent samples its own minibatch from the shared replay buffer.
- This follows the original MADDPG formulation (Lowe et al., 2017) and the
  reference TensorFlow implementation.
- Independent minibatches yield unbiased updates and help decorrelate agent
  learning in highly non-stationary multi-agent settings.

Critic updates:
- Each agent's critic is updated using the joint actions from the replay buffer
  and the target actions from the target actors.
- This is consistent with the original MADDPG and allows for stable critic learning.

Actor updates:
- Policies are deterministic.
- When updating agent i, the joint action is constructed by replacing only
  agent i's action with the current actor output; all other agents' actions
  are taken from the replay buffer.

Rationale:
- MADDPG does not optimize an expectation over a stochastic policy.
- Per-agent sampling is sufficient and avoids unnecessary coupling between
  agents' updates.
"""

import logging
import os
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.memory.memory_sampler as memory_sampler
from cares_reinforcement_learning.networks import functional as fnc
from cares_reinforcement_learning.algorithm.algorithm import MARLAlgorithm
from cares_reinforcement_learning.algorithm.policy.DDPG import DDPG
from cares_reinforcement_learning.memory.memory_buffer import MARLMemoryBuffer
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import (
    MARLObservation,
    SARLObservation,
)
from cares_reinforcement_learning.algorithm.configurations import MADDPGConfig


class MADDPG(MARLAlgorithm[dict[str, np.ndarray]]):
    def __init__(
        self,
        agents: list[DDPG],
        config: MADDPGConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        self.agent_networks = agents
        self.num_agents = len(agents)

        self.gamma = config.gamma
        self.tau = config.tau

        self.max_grad_norm = config.max_grad_norm

        # M3DDPG adversarial perturbation scale
        self.use_m3 = config.use_m3
        self.m3_alpha = config.m3_alpha

        # ERNIE adversarial regularization
        self.use_ernie = config.use_ernie
        self.ernie_lambda = config.ernie_lambda
        self.ernie_eps = config.ernie_eps
        self.ernie_k_steps = config.ernie_k_steps
        self.ernie_norm = config.ernie_norm

        self.ernie_step_size = (
            self.ernie_eps / self.ernie_k_steps if self.ernie_k_steps > 0 else 0.0
        )

        self.learn_counter = 0

    # TODO verify that the ordering of agents is consistent
    def act(
        self,
        observation: MARLObservation,
        evaluation: bool = False,
    ) -> ActionSample[dict[str, np.ndarray]]:
        agent_states = observation.agent_states
        avail_actions = observation.available_actions

        agent_ids = list(agent_states.keys())
        actions = {}

        for i, agent in enumerate(self.agent_networks):
            agent_name = agent_ids[i]  # consistent ordering in dict
            obs_i = agent_states[agent_name]
            avail_i = avail_actions[agent_name]

            agent_observation = SARLObservation(
                vector_state=obs_i,
                avail_actions=avail_i,
            )

            agent_sample = agent.act(agent_observation, evaluation)
            actions[agent_name] = agent_sample.action

        return ActionSample(action=actions, source="policy")

    @staticmethod
    def _project_l2_ball(delta: torch.Tensor, eps: float) -> torch.Tensor:
        """
        Project `delta` onto the L2 ball of radius `eps`, independently per batch element.

        Args:
            delta: (B, obs_dim)
            eps: radius

        Returns:
            projected delta: (B, obs_dim)
        """
        flat = delta.view(delta.size(0), -1)  # (B, D)
        norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)  # (B, 1)
        scale = (eps / norms).clamp(max=1.0)  # (B, 1)
        return (flat * scale).view_as(delta)

    # ERNIE methods
    def _ernie_adv_delta(
        self,
        actor_net: torch.nn.Module,
        obs: torch.Tensor,
        eps: float,
        k_steps: int,
        step_size: float,
        norm: Literal["linf", "l2"] = "linf",
    ) -> torch.Tensor:
        """
        ERNIE: inner maximization via PGD ascent to find delta that maximizes
        D(pi(o+delta), pi(o)) under ||delta|| <= eps.

        Args:
            actor_net: maps obs -> action. obs: (B, obs_dim) -> (B, act_dim)
            obs: (B, obs_dim)
            eps: perturbation budget
            k_steps: number of PGD steps
            step_size: PGD step size
            norm: "linf" or "l2"

        Returns:
            delta_adv: (B, obs_dim), detached
        """
        if eps <= 0.0 or k_steps <= 0 or step_size <= 0.0:
            return torch.zeros_like(obs)

        # Reference action (no gradients through this branch)
        with torch.no_grad():
            base_action = actor_net(obs)  # (B, act_dim)

        # Random init within constraint set
        delta = torch.empty_like(obs).uniform_(-eps, eps)
        if norm == "l2":
            delta = self._project_l2_ball(delta, eps)

        delta.requires_grad_(True)

        for _ in range(k_steps):
            pert_action = actor_net(obs + delta)

            # Maximize mean squared action deviation (stable for deterministic actors)
            objective = (pert_action - base_action).pow(2).mean()

            (grad,) = torch.autograd.grad(
                outputs=objective,
                inputs=delta,
                retain_graph=False,
                create_graph=False,
                only_inputs=True,
            )

            with torch.no_grad():
                if norm == "linf":
                    delta.add_(step_size * grad.sign())
                    delta.clamp_(-eps, eps)
                else:  # "l2"
                    delta.add_(step_size * grad)
                    delta.copy_(self._project_l2_ball(delta, eps))

            delta.requires_grad_(True)

        return delta.detach()

    # M3DDPG methods
    def _compute_adversarial_actions(
        self,
        agent_index: int,
        actions: torch.Tensor,  # (batch, n_agents, act_dim)
        global_states: torch.Tensor,  # (batch, state_dim)
        critic: torch.nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return actions_adv where for j != agent_index:
            a_j_adv = a_j + eps_j
        and eps_j is a 1-step gradient move that *decreases* Q_i.
        """
        if self.m3_alpha == 0.0:
            # Degenerates to original MADDPG
            return actions.detach(), torch.zeros_like(actions)

        # Clone and mark for gradient wrt actions only
        actions_for_grad = actions.detach().clone().requires_grad_(True)
        batch_size = actions_for_grad.shape[0]

        # Flatten to feed critic
        joint_actions_flat = actions_for_grad.view(batch_size, -1)
        q_vals = critic(global_states, joint_actions_flat).mean()  # scalar

        # Gradient of Q wrt all actions
        (grad_actions,) = torch.autograd.grad(
            q_vals,
            actions_for_grad,
            retain_graph=False,
            create_graph=False,
        )
        # grad_actions: (batch, n_agents, act_dim)

        # Scale by |a_j| as in Eq.(17)
        act_norm = actions_for_grad.norm(dim=-1, keepdim=True)
        grad_norm = grad_actions.norm(dim=-1, keepdim=True) + 1e-8

        eps = -self.m3_alpha * act_norm * grad_actions / grad_norm

        # Zero perturbation for the current agent i
        mask = torch.ones_like(eps)
        mask[:, agent_index, :] = 0.0
        eps = eps * mask

        actions_adv = actions_for_grad + eps
        return actions_adv.detach(), eps.detach()  # no gradients through perturbation

    def _update_critic(
        self,
        agent: DDPG,
        agent_index: int,
        global_states: torch.Tensor,
        joint_actions: torch.Tensor,  # (B, N * act_dim) from replay
        rewards_i: torch.Tensor,  # (B, 1)
        next_global_states: torch.Tensor,
        next_actions_tensor: torch.Tensor,  # (B, N, act_dim) from target actors
        dones_i: torch.Tensor,
    ):
        info: dict[str, Any] = {}

        # --- Step 1: build (possibly adversarial) next joint actions ---
        if self.use_m3:
            # M3DDPG: perturb OTHER agents' target actions for agent i
            next_actions_adv, eps = self._compute_adversarial_actions(
                agent_index=agent_index,
                actions=next_actions_tensor,  # (B, N, act_dim)
                global_states=next_global_states,  # (B, state_dim)
                critic=agent.target_critic_net,  # target critic
            )
            next_joint_actions = next_actions_adv.view(next_actions_adv.size(0), -1)
        else:
            # Plain MADDPG
            next_joint_actions = next_actions_tensor.view(
                next_actions_tensor.size(0), -1
            )

        # --- Step 2: TD target ---
        with torch.no_grad():
            target_q = agent.target_critic_net(next_global_states, next_joint_actions)
            q_target = rewards_i + self.gamma * (1 - dones_i) * target_q

        # --- Step 3: critic regression on *current* joint_actions (unperturbed) ---
        q_values = agent.critic_net(global_states, joint_actions)

        loss = F.mse_loss(q_values, q_target)

        agent.critic_net_optimiser.zero_grad()
        loss.backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                agent.critic_net.parameters(), self.max_grad_norm
            )

        agent.critic_net_optimiser.step()

        with torch.no_grad():

            td = q_values - q_target

            # --- Value scale ---
            info["q_mean"] = q_values.mean().item()
            info["q_std"] = q_values.std(unbiased=False).item()

            info["q_target_mean"] = q_target.mean().item()
            info["q_target_std"] = q_target.std(unbiased=False).item()

            # --- TD error diagnostics ---
            td_abs = td.abs()
            info["td_abs_mean"] = td_abs.mean().item()
            info["td_abs_p95"] = td_abs.quantile(0.95).item()
            info["td_abs_max"] = td_abs.max().item()

            # --- Signed bias ---
            info["td_mean"] = td.mean().item()

            if self.use_m3:
                info["critic_m3_eps_norm_mean"] = eps.norm(dim=-1).mean().item()
                info["critic_m3_eps_norm_p95"] = eps.norm(dim=-1).quantile(0.95).item()

            # --- Critic loss ---
            info["critic_loss"] = loss.item()

        return info

    def _update_actor(
        self,
        agent: DDPG,
        agent_index: int,
        obs_tensors: dict[str, torch.Tensor],
        global_states: torch.Tensor,
        actions_tensor: torch.Tensor,  # (B, N, act_dim)
    ):
        """
        Paper-faithful MADDPG actor update:
        - For j ≠ agent_index: use replay-buffer actions
        - For j == agent_index: use current actor output
        """
        info: dict[str, Any] = {}

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
        actions_i = agent.actor_net(obs_i)  # differentiable

        actions_all[:, agent_index, :] = actions_i  # keep others from buffer

        # ---------------------------------------------------------
        # Step 3a: Apply M3DDPG adversarial perturbation (if enabled)
        # ---------------------------------------------------------
        if self.use_m3:
            # compute perturbation on ALL actions (but this returns detached)
            actions_adv, eps = self._compute_adversarial_actions(
                agent_index=agent_index,
                actions=actions_all,
                global_states=global_states,
                critic=agent.critic_net,
            )

            # reinsert differentiable action for agent i
            actions_adv[:, agent_index, :] = actions_i
            actions_all = actions_adv

        # ---------------------------------------------------------
        # Step 3b: Apply ERNIE adversarial perturbation (if enabled)
        # ---------------------------------------------------------
        ernie_reg = torch.tensor(0.0, device=obs_i.device)
        if self.use_ernie:
            delta_adv = self._ernie_adv_delta(
                actor_net=agent.actor_net,
                obs=obs_i,
                eps=self.ernie_eps,
                k_steps=self.ernie_k_steps,
                step_size=self.ernie_step_size,
                norm=self.ernie_norm,
            )
            pred_action_adv = agent.actor_net(obs_i + delta_adv)
            ernie_reg = (pred_action_adv - actions_i).pow(2).mean()

        # ---------------------------------------------------------
        # Step 4: Compute actor loss: -Q_i(x, a_1,...,a_i,...,a_N)
        # ---------------------------------------------------------
        joint_actions_flat = actions_all.reshape(batch_size, -1)
        with fnc.evaluating(agent.critic_net):
            actor_q_values = agent.critic_net(global_states, joint_actions_flat)

        # regularization as in TF code
        reg = (actions_i**2).mean() * 1e-3

        actor_loss = -actor_q_values.mean() + reg + (self.ernie_lambda * ernie_reg)

        dq_da = torch.autograd.grad(
            outputs=-actor_q_values.mean(),  # NOTE: uses Q-term only, excludes regularizers
            inputs=actions_i,
            retain_graph=True,
            create_graph=False,
        )[0]
        with torch.no_grad():
            # - ~0 early: critic surface flat around actor actions (weak learning signal)
            # - very large: critic surface sharp -> unstable / exploitative actor updates
            info["dq_da_abs_mean"] = dq_da.abs().mean().item()
            info["dq_da_norm_mean"] = dq_da.norm(dim=1).mean().item()
            info["dq_da_norm_p95"] = dq_da.norm(dim=1).quantile(0.95).item()

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

        with torch.no_grad():
            # Policy Action Health (tanh policies in [-1, 1])
            # pi_action_saturation_frac:
            # High values (>0.8 early) often mean the actor is slamming bounds,
            # reducing effective gradient flow through tanh.
            info["pi_action_mean"] = actions_i.mean().item()
            info["pi_action_std"] = actions_i.std().item()
            info["pi_action_abs_mean"] = actions_i.abs().mean().item()
            info["pi_action_saturation_frac"] = (
                (actions_i.abs() > 0.95).float().mean().item()
            )

            # actor_q_mean should generally increase over training.
            # actor_q_std large + unstable may indicate critic inconsistency.
            info["actor_loss"] = actor_loss.item()
            info["actor_q_mean"] = actor_q_values.mean().item()
            info["actor_q_std"] = actor_q_values.std().item()

            # --- ERNIE diagnostics ---
            if self.use_ernie:
                info["ernie_reg"] = ernie_reg.item()

            if self.use_m3:
                info["actor_m3_eps_norm_mean"] = eps.norm(dim=-1).mean().item()
                info["actor_m3_eps_norm_p95"] = eps.norm(dim=-1).quantile(0.95).item()

            info["actor_loss"] = actor_loss.item()

        return info

    def train(
        self,
        memory_buffer: MARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:

        self.learn_counter += 1

        info: dict[str, Any] = {}

        for agent_index, current_agent in enumerate(self.agent_networks):
            # ---------------------------------------------------------
            # Update each agent
            # ---------------------------------------------------------

            # Update action noise for exploration (decayed over training)
            current_agent.action_noise = current_agent.action_noise_scheduler.get_value(
                episode_context.training_step
            )

            info[f"action_noise_agent_{agent_index}"] = float(
                current_agent.action_noise
            )

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

            sample_size = len(indices)

            states_tensors = observation_tensor.global_state_tensor
            next_states_tensors = next_observation_tensor.global_state_tensor

            agent_states_tensors = observation_tensor.agent_states_tensor
            next_agent_states_tensors = next_observation_tensor.agent_states_tensor

            agent_ids = list(agent_states_tensors.keys())

            # ---------------------------------------------------------
            # Build next_actions_tensor using TARGET actors
            # ---------------------------------------------------------
            next_actions = []
            for agent, agent_id in zip(self.agent_networks, agent_ids):
                obs_next_j = next_agent_states_tensors[agent_id]
                next_action_j = agent.target_actor_net(obs_next_j)
                next_actions.append(next_action_j)

            next_actions_tensor = torch.stack(next_actions, dim=1)

            # Flatten replay-buffer actions for this batch
            joint_actions = actions_tensor.reshape(sample_size, -1)

            with torch.no_grad():
                # ---------------------------------------------------------
                # Batch-level multi-agent diagnostics (this agent's draw)
                # ---------------------------------------------------------
                # Joint action volatility in the replay batch (all agents)
                info[f"agent_{agent_index}_joint_action_mean"] = (
                    actions_tensor.mean().item()
                )
                info[f"agent_{agent_index}_joint_action_std"] = actions_tensor.std(
                    unbiased=False
                ).item()

                # Per-agent action magnitude (detect frozen/saturated agent in replay)
                # actions_tensor: (B, N, A)
                per_agent_abs_mean = actions_tensor.abs().mean(dim=(0, 2))  # (N,)
                per_agent_std = actions_tensor.std(dim=(0, 2), unbiased=False)  # (N,)

                info[f"agent_{agent_index}_replay_action_abs_mean"] = (
                    per_agent_abs_mean.mean().item()
                )
                info[f"agent_{agent_index}_replay_action_abs_std_across_agents"] = (
                    per_agent_abs_mean.std(unbiased=False).item()
                )
                info[f"agent_{agent_index}_replay_action_std_mean"] = (
                    per_agent_std.mean().item()
                )

                # Coordination proxy: how aligned are agents' actions? (cheap)
                # Cos similarity between agents' action vectors per sample, averaged.
                # Flatten each agent action: (B, N, A) -> (B, N, A)
                a = actions_tensor  # (B,N,A)
                a_norm = a / a.norm(dim=2, keepdim=True).clamp_min(1e-6)

                # pairwise cosine for all agent pairs
                cos = torch.einsum("bna,bma->bnm", a_norm, a_norm)  # (B,N,N)
                # ignore diagonal
                n = cos.shape[1]
                mask = ~torch.eye(n, device=cos.device, dtype=torch.bool)
                info[f"agent_{agent_index}_replay_action_cos_mean"] = (
                    cos[:, mask].mean().item()
                )

                # Reward/done scale sanity for this agent (helps catch mis-scaling)
                info[f"agent_{agent_index}_reward_mean"] = rewards_tensor.mean().item()
                info[f"agent_{agent_index}_done_frac"] = (
                    dones_tensor.float().mean().item()
                )

            # ---------------------------------------------------------
            # Critic update for this agent
            # ---------------------------------------------------------
            rewards_i = rewards_tensor[:, agent_index]
            dones_i = dones_tensor[:, agent_index]

            critic_info = self._update_critic(
                agent=current_agent,
                agent_index=agent_index,
                global_states=states_tensors,
                joint_actions=joint_actions,
                rewards_i=rewards_i,
                next_global_states=next_states_tensors,
                next_actions_tensor=next_actions_tensor,
                dones_i=dones_i,
            )
            info.update({f"agent_{agent_index}_{k}": v for k, v in critic_info.items()})

            # ---------------------------------------------------------
            # Actor update
            # ---------------------------------------------------------
            actor_info = self._update_actor(
                agent=current_agent,
                agent_index=agent_index,
                obs_tensors=agent_states_tensors,
                global_states=states_tensors,
                actions_tensor=actions_tensor,
            )
            info.update({f"agent_{agent_index}_{k}": v for k, v in actor_info.items()})

        # --- Cross-agent diagnostics ---
        metrics = list(critic_info.keys()) + list(actor_info.keys())
        for metric in metrics:
            values = [info[f"agent_{i}_{metric}"] for i in range(self.num_agents)]
            info[f"mean_{metric}"] = float(np.mean(values))
            info[f"std_{metric}"] = float(np.std(values))
            info[f"max_{metric}"] = float(np.max(values))
            info[f"min_{metric}"] = float(np.min(values))

        # Update Target networks with soft update
        for current_agent in self.agent_networks:
            current_agent.update_target_networks()

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
