"""
SACD (Soft Actor-Critic for Discrete Action Settings)
------------------------------------------------------

Original Paper: https://arxiv.org/pdf/1910.07207
Original Code: https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/actor_critic_agents/SAC_Discrete.py

This variant keeps the current repo's discrete SAC update path while
preserving the legacy extension points that were added in older branches:
encoder support, average-Q targets, clipped-Q critic loss, entropy penalty,
and a train_policy compatibility shim.
"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import cares_reinforcement_learning.memory.memory_sampler as memory_sampler
from cares_reinforcement_learning.algorithm.algorithm import SARLAlgorithm
from cares_reinforcement_learning.algorithm.configurations import SACDConfig
from cares_reinforcement_learning.encoders.vanilla_autoencoder import (
    Encoder,
    VanillaAutoencoder,
)
from cares_reinforcement_learning.memory.memory_buffer import SARLMemoryBuffer
from cares_reinforcement_learning.networks import functional as fnc
from cares_reinforcement_learning.networks.SACD import Actor, Critic
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import (
    SARLObservation,
    SARLObservationTensors,
)


@dataclass(frozen=True, slots=True)
class CriticLossInfo:
    q_values_one: torch.Tensor
    q_values_two: torch.Tensor
    critic_loss_one: torch.Tensor
    critic_loss_two: torch.Tensor
    extra_info: Mapping[str, Any] = MappingProxyType({})

    @property
    def total_loss(self) -> torch.Tensor:
        return self.critic_loss_one + self.critic_loss_two

    @property
    def log_info(self) -> Mapping[str, Any]:
        info = {
            "critic_loss_one": self.critic_loss_one.item(),
            "critic_loss_two": self.critic_loss_two.item(),
            "critic_loss_total": self.total_loss.item(),
        }
        return info | dict(self.extra_info)


class _DiscreteEncoderActor(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        actor: Actor,
        add_vector_observation: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.actor = actor
        self.num_actions = actor.num_actions
        self.add_vector_observation = add_vector_observation

    def forward(
        self,
        state: SARLObservationTensors | torch.Tensor,
        detach_encoder: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if isinstance(state, torch.Tensor):
            return self.actor(state)

        if state.image_state_tensor is None:
            raise ValueError("Image state required for encoder-aware SACD actor.")

        state_latent = self.encoder(state.image_state_tensor, detach_cnn=detach_encoder)
        actor_input = state_latent
        if self.add_vector_observation:
            actor_input = torch.cat([state.vector_state_tensor, actor_input], dim=1)
        return self.actor(actor_input)


class _DiscreteEncoderCritic(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        critic: Critic,
        add_vector_observation: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.critic = critic
        self.add_vector_observation = add_vector_observation

    def forward(
        self,
        state: SARLObservationTensors | torch.Tensor,
        detach_encoder: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(state, torch.Tensor):
            return self.critic(state)

        if state.image_state_tensor is None:
            raise ValueError("Image state required for encoder-aware SACD critic.")

        state_latent = self.encoder(state.image_state_tensor, detach_cnn=detach_encoder)
        critic_input = state_latent
        if self.add_vector_observation:
            critic_input = torch.cat([state.vector_state_tensor, critic_input], dim=1)
        return self.critic(critic_input)


class SACD(SARLAlgorithm[int]):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: SACDConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="discrete_policy", config=config, device=device)

        self.actor_net = actor_network.to(device)
        self.critic_net = critic_network.to(device)
        self.target_critic_net = copy.deepcopy(self.critic_net).to(device)
        self.target_critic_net.eval()

        self.gamma = config.gamma
        self.tau = config.tau
        self.reward_scale = config.reward_scale

        self.learn_counter = 0
        self.policy_update_freq = config.policy_update_freq
        self.target_update_freq = config.target_update_freq

        self.action_num = self.actor_net.num_actions
        self.target_entropy = -np.log(1.0 / self.action_num) * getattr(
            config, "target_entropy_multiplier", 1.0
        )

        self.use_clipped_q = bool(getattr(config, "use_clipped_q", False))
        self.use_average_q = bool(getattr(config, "use_average_q", False))
        self.use_entropy_penalty = bool(getattr(config, "use_entropy_penalty", False))
        self.entropy_penalty_beta = getattr(config, "entropy_penalty_beta", 0.0)
        self.normalise_state = bool(getattr(config, "normalise_state", False))
        self.auto_entropy_tuning = bool(getattr(config, "auto_entropy_tuning", True))
        self.q_clip_epsilon = float(getattr(config, "q_clip_epsilon", 0.5))

        self.use_per_buffer = int(getattr(config, "use_per_buffer", 0))
        self.per_sampling_strategy = getattr(config, "per_sampling_strategy", "stratified")
        self.per_weight_normalisation = getattr(config, "per_weight_normalisation", "batch")
        self.min_priority = float(getattr(config, "min_priority", 1e-6))
        self.per_alpha = float(getattr(config, "per_alpha", 0.6))

        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=config.actor_lr
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=config.critic_lr
        )

        self.log_alpha = torch.tensor(np.log(1.0), device=device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)

        self._last_entropy: torch.Tensor | None = None
        self.autoencoder: Any | None = None

        self.apply_actor_film = False
        self.apply_critic_film = False
        self._set_film_support()

        if hasattr(config, "encoder_type") or hasattr(config, "autoencoder_config"):
            self._set_encoding(config)

    def _supports_observation_tensors(self) -> bool:
        return hasattr(self.actor_net, "actor") and hasattr(self.critic_net, "critic")

    def _set_film_support(self) -> None:
        self.apply_actor_film = hasattr(self.actor_net, "update_film_params")
        self.apply_critic_film = hasattr(self.critic_net, "update_film_params")

    def _extract_tasks_tensor(self, train_data: list[dict[str, Any]]) -> torch.Tensor | None:
        if not train_data:
            return None

        task_rows: list[np.ndarray] = []
        for item in train_data:
            if not isinstance(item, dict):
                return None

            tasks = item.get("tasks", item.get("task", None))
            if tasks is None:
                return None

            task_rows.append(np.asarray(tasks, dtype=np.float32).reshape(-1))

        if not task_rows:
            return None

        return torch.tensor(np.stack(task_rows), dtype=torch.float32, device=self.device)

    def _maybe_normalise_vector(self, vector_state_tensor: torch.Tensor) -> torch.Tensor:
        if self.normalise_state:
            return vector_state_tensor / 255.0
        return vector_state_tensor

    def _observation_to_tensors(self, observation: SARLObservation) -> SARLObservationTensors:
        vector_state_tensor = torch.tensor(
            observation.vector_state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        vector_state_tensor = self._maybe_normalise_vector(vector_state_tensor)

        image_state_tensor = None
        if observation.image_state is not None:
            image_state_tensor = torch.tensor(
                observation.image_state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            if image_state_tensor.max() > 1.0:
                image_state_tensor = image_state_tensor / 255.0

        return SARLObservationTensors(
            vector_state_tensor=vector_state_tensor,
            image_state_tensor=image_state_tensor,
        )

    def _forward_actor(
        self, states: SARLObservationTensors | torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return self.actor_net(states)  # type: ignore[arg-type]

    def _forward_critic(
        self, states: SARLObservationTensors | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.critic_net(states)  # type: ignore[arg-type]

    def _select_policy_input(
        self, states: SARLObservationTensors
    ) -> SARLObservationTensors | torch.Tensor:
        if self._supports_observation_tensors() and states.image_state_tensor is not None:
            return states
        return self._maybe_normalise_vector(states.vector_state_tensor)

    def _extract_train_data_tensor(
        self,
        train_data: list[dict[str, Any]],
        keys: tuple[str, ...],
    ) -> torch.Tensor | None:
        if not train_data:
            return None

        values: list[float] = []
        for item in train_data:
            if not isinstance(item, dict):
                return None

            found_value: Any | None = None
            for key in keys:
                if key in item:
                    found_value = item[key]
                    break

            if found_value is None:
                return None

            value_array = np.asarray(found_value, dtype=np.float32).reshape(-1)
            if value_array.size == 0:
                return None

            values.append(float(value_array.mean()))

        return torch.tensor(values, dtype=torch.float32, device=self.device).unsqueeze(-1)

    def _get_q_target(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        if self.use_average_q:
            return torch.mean(torch.stack((q1, q2), dim=-1), dim=-1)
        return torch.minimum(q1, q2)

    def _get_state_action_q_values(
        self,
        state: SARLObservationTensors | torch.Tensor,
        actions: torch.Tensor,
        network: Critic,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_values_one, q_values_two = network(state)  # type: ignore[arg-type]
        return q_values_one.gather(1, actions), q_values_two.gather(1, actions)

    def _get_critic_loss(
        self,
        state: SARLObservationTensors | torch.Tensor,
        actions: torch.Tensor,
        q_target: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> CriticLossInfo:
        q_values_one, q_values_two = self._get_state_action_q_values(
            state, actions, self.critic_net
        )

        critic_loss_one = F.mse_loss(q_values_one, q_target, reduction="none")
        critic_loss_two = F.mse_loss(q_values_two, q_target, reduction="none")

        if weights is not None:
            critic_loss_one = (critic_loss_one * weights).mean()
            critic_loss_two = (critic_loss_two * weights).mean()
        else:
            critic_loss_one = critic_loss_one.mean()
            critic_loss_two = critic_loss_two.mean()

        return CriticLossInfo(
            q_values_one=q_values_one,
            q_values_two=q_values_two,
            critic_loss_one=critic_loss_one,
            critic_loss_two=critic_loss_two,
        )

    def _get_clipped_critic_loss(
        self,
        state: SARLObservationTensors | torch.Tensor,
        actions: torch.Tensor,
        q_target: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> CriticLossInfo:
        info: dict[str, Any] = {}

        q_values_one, q_values_two = self._get_state_action_q_values(
            state, actions, self.critic_net
        )
        q_target_one, q_target_two = self._get_state_action_q_values(
            state, actions, self.target_critic_net
        )

        clipped_q1 = q_target_one + torch.clamp(
            q_values_one - q_target_one, -self.q_clip_epsilon, self.q_clip_epsilon
        )
        clipped_q2 = q_target_two + torch.clamp(
            q_values_two - q_target_two, -self.q_clip_epsilon, self.q_clip_epsilon
        )

        q1_std_loss = F.mse_loss(q_values_one, q_target, reduction="none")
        q1_clp_loss = F.mse_loss(clipped_q1, q_target, reduction="none")
        q2_std_loss = F.mse_loss(q_values_two, q_target, reduction="none")
        q2_clp_loss = F.mse_loss(clipped_q2, q_target, reduction="none")

        critic_loss_one = torch.maximum(q1_std_loss, q1_clp_loss)
        critic_loss_two = torch.maximum(q2_std_loss, q2_clp_loss)

        if weights is not None:
            critic_loss_one = (critic_loss_one * weights).mean()
            critic_loss_two = (critic_loss_two * weights).mean()
        else:
            critic_loss_one = critic_loss_one.mean()
            critic_loss_two = critic_loss_two.mean()

        info["clipped_q1"] = clipped_q1.mean().item()
        info["clipped_q2"] = clipped_q2.mean().item()
        info["clip_ratio"] = float((q1_clp_loss.mean() >= q1_std_loss.mean()).item())

        return CriticLossInfo(
            q_values_one=q_values_one,
            q_values_two=q_values_two,
            critic_loss_one=critic_loss_one,
            critic_loss_two=critic_loss_two,
            extra_info=MappingProxyType(info),
        )

    def _get_bootstrapped_value_estimate(
        self,
        next_states: SARLObservationTensors | torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            with fnc.evaluating(self.actor_net):
                _, (action_probs, log_actions_probs), _ = self._forward_actor(next_states)

            next_target_one, next_target_two = self._forward_critic(next_states)
            next_target = self._get_q_target(next_target_one, next_target_two)

            next_target = (next_target * action_probs).sum(dim=-1) - self.alpha * (
                action_probs * log_actions_probs
            ).sum(dim=-1)
            next_target = (next_target * self.gamma).unsqueeze(dim=-1)
            next_target = rewards * self.reward_scale + (1.0 - dones) * next_target

        return next_target

    def _update_critic(
        self,
        states: SARLObservationTensors | torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: SARLObservationTensors | torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> tuple[dict[str, Any], np.ndarray | None]:
        q_target = self._get_bootstrapped_value_estimate(next_states, rewards, dones)

        act = actions.long().unsqueeze(-1)
        critic_loss = (
            self._get_clipped_critic_loss(states, act, q_target, weights=weights)
            if self.use_clipped_q
            else self._get_critic_loss(states, act, q_target, weights=weights)
        )

        self.critic_net_optimiser.zero_grad()
        critic_loss.total_loss.backward()
        self.critic_net_optimiser.step()

        td_error_one = (critic_loss.q_values_one - q_target).abs()
        td_error_two = (critic_loss.q_values_two - q_target).abs()
        priorities: np.ndarray | None = None
        if self.use_per_buffer:
            priorities = (
                torch.max(td_error_one, td_error_two)
                .clamp(self.min_priority)
                .pow(self.per_alpha)
                .detach()
                .cpu()
                .numpy()
                .flatten()
            )

        info = dict(critic_loss.log_info)
        with torch.no_grad():
            info["q_target_mean"] = q_target.mean().item()
            info["q_target_std"] = q_target.std(unbiased=False).item()
            info["q1_mean"] = critic_loss.q_values_one.mean().item()
            info["q2_mean"] = critic_loss.q_values_two.mean().item()
            info["q_twin_gap_abs_mean"] = (
                (critic_loss.q_values_one - critic_loss.q_values_two).abs().mean().item()
            )
            td_abs = torch.maximum(td_error_one, td_error_two).squeeze(1)
            info["td_abs_mean"] = td_abs.mean().item()
            info["td_abs_p95"] = td_abs.quantile(0.95).item()
            info["td_abs_max"] = td_abs.max().item()

            qf1_next, qf2_next = self._forward_critic(next_states)
            min_q_next = self._get_q_target(qf1_next, qf2_next)
            info["target_min_q_mean"] = min_q_next.mean().item()

        return info, priorities

    def _update_actor_alpha(
        self,
        states: SARLObservationTensors | torch.Tensor,
        old_entropies: torch.Tensor | None = None,
        tasks: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {}

        if self.apply_actor_film and tasks is not None:
            self.actor_net.update_film_params(tasks)

        _, (action_probs, log_action_probs), _ = self._forward_actor(states)

        with fnc.evaluating(self.critic_net):
            qf1_pi, qf2_pi = self._forward_critic(states)

        min_qf_pi = self._get_q_target(qf1_pi, qf2_pi)

        inside_term = self.alpha * log_action_probs - min_qf_pi
        actor_loss = (action_probs * inside_term).sum(dim=1).mean()

        entropy = -(action_probs * log_action_probs).sum(dim=1)
        self._last_entropy = entropy.detach()

        if self.use_entropy_penalty and old_entropies is not None:
            entropy_penalty = self.entropy_penalty_beta * F.mse_loss(
                old_entropies.view(-1), entropy.view(-1)
            )
            actor_loss = actor_loss + entropy_penalty
            info["entropy_penalty"] = entropy_penalty.item()

        expected_log_prob = torch.sum(log_action_probs * action_probs, dim=1)

        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        if self.auto_entropy_tuning:
            alpha_loss = self._update_alpha(expected_log_prob)
        else:
            alpha_loss = torch.zeros((), device=self.device)

        with torch.no_grad():
            info["entropy_mean"] = entropy.mean().item()
            info["entropy_std"] = entropy.std(unbiased=False).item()

            max_prob = action_probs.max(dim=1).values
            info["max_action_prob_mean"] = max_prob.mean().item()
            info["max_action_prob_p95"] = max_prob.quantile(0.95).item()
            info["policy_prob_std_mean"] = action_probs.std(dim=1).mean().item()

            info["min_q_pi_mean"] = min_qf_pi.mean().item()
            info["min_q_pi_std"] = min_qf_pi.std(unbiased=False).item()

            entropy_gap = -(expected_log_prob + self.target_entropy)
            info["entropy_gap_mean"] = entropy_gap.mean().item()

            info["actor_loss"] = actor_loss.item()
            info["alpha_loss"] = alpha_loss.item()
            info["alpha"] = self.alpha.item()
            info["log_alpha"] = self.log_alpha.item()

        return info

    def act(
        self, observation: SARLObservation, evaluation: bool = False
    ) -> ActionSample[int]:
        self.actor_net.eval()

        with torch.no_grad():
            if self._supports_observation_tensors():
                if observation.image_state is None:
                    raise ValueError("Encoder-aware SACD requires image observations.")
                state = self._select_policy_input(self._observation_to_tensors(observation))
            else:
                state = torch.tensor(
                    observation.vector_state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                state = self._maybe_normalise_vector(state)

            if evaluation:
                _, _, action = self._forward_actor(state)
            else:
                action, _, _ = self._forward_actor(state)

        self.actor_net.train()
        return ActionSample(action=action.item(), source="policy")

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action_from_policy(self, action_context: Any) -> np.ndarray:
        state = getattr(action_context, "state", action_context)
        evaluation = bool(getattr(action_context, "evaluation", False))

        tasks = None
        extras = getattr(action_context, "extras", None)
        if isinstance(extras, dict):
            tasks = extras.get("tasks")

        if tasks is not None and hasattr(self.actor_net, "update_film_params"):
            tasks_tensor = torch.tensor(
                np.asarray(tasks), dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            self.actor_net.update_film_params(tasks_tensor)

        if not isinstance(state, SARLObservation):
            state = SARLObservation(vector_state=np.asarray(state))

        return np.asarray(self.act(state, evaluation=evaluation).action)

    def get_extras(self) -> list[Any]:
        if self._last_entropy is None:
            return []
        return [self._last_entropy.mean().item()]

    def train(
        self,
        memory_buffer: SARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:
        del episode_context

        self.learn_counter += 1

        (
            observation_tensor,
            actions_tensor,
            rewards_tensor,
            next_observation_tensor,
            dones_tensor,
            weights_tensor,
            train_data,
            indices,
        ) = memory_sampler.sample(
            memory=memory_buffer,
            batch_size=self.batch_size,
            device=self.device,
            use_per_buffer=self.use_per_buffer,
            per_sampling_strategy=self.per_sampling_strategy,
            per_weight_normalisation=self.per_weight_normalisation,
        )

        states: SARLObservationTensors | torch.Tensor = observation_tensor
        next_states: SARLObservationTensors | torch.Tensor = next_observation_tensor

        if self._supports_observation_tensors():
            if observation_tensor.image_state_tensor is None or next_observation_tensor.image_state_tensor is None:
                raise ValueError("Encoder-aware SACD requires image observations in the replay batch.")
        else:
            states = self._maybe_normalise_vector(observation_tensor.vector_state_tensor)
            next_states = self._maybe_normalise_vector(next_observation_tensor.vector_state_tensor)

        old_entropies = self._extract_train_data_tensor(
            train_data,
            (
                "entropy",
                "avg_entropy",
                "policy_entropy",
                "action_entropy",
                "entropy_mean",
                "old_entropy",
            ),
        )
        tasks_tensor = self._extract_tasks_tensor(train_data)

        info: dict[str, Any] = {}

        if self.apply_critic_film and tasks_tensor is not None:
            self.critic_net.update_film_params(tasks_tensor)

        critic_info, priorities = self._update_critic(
            states,
            actions_tensor,
            rewards_tensor,
            next_states,
            dones_tensor,
            weights=weights_tensor,
        )
        info.update(critic_info)

        if self.learn_counter % self.policy_update_freq == 0:
            actor_info = self._update_actor_alpha(
                states,
                old_entropies=old_entropies,
                tasks=tasks_tensor,
            )
            info.update(actor_info)

        if self.autoencoder is not None and observation_tensor.image_state_tensor is not None:
            info["ae_loss"] = self._update_autoencoder(observation_tensor.image_state_tensor)

        if self.learn_counter % self.target_update_freq == 0:
            self.soft_update_params(self.critic_net, self.target_critic_net, self.tau)

        if self.use_per_buffer and priorities is not None:
            memory_buffer.update_priorities(indices, priorities)

        return info

    def train_policy(self, training_context: Any) -> dict[str, Any]:
        memory = getattr(training_context, "memory", training_context)
        episode_context = getattr(training_context, "episode_context", None)
        return self.train(memory, episode_context)

    def _update_alpha(self, expected_log_prob: torch.Tensor) -> torch.Tensor:
        alpha_loss = -(
            self.log_alpha * (expected_log_prob + self.target_entropy).detach()
        ).mean()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return alpha_loss

    def _update_autoencoder(self, image_states: torch.Tensor) -> float:
        if self.autoencoder is None:
            return 0.0
        ae_loss = self.autoencoder.update_autoencoder(image_states)
        return float(ae_loss.item())

    def _calculate_value(self, state: np.ndarray, action: np.ndarray) -> float:  # type: ignore[override]
        del state, action
        return 0.0

    def _set_encoding(self, config: SACDConfig) -> None:
        if not hasattr(config, "autoencoder_config"):
            return

        if not hasattr(self.actor_net, "act_net") or not hasattr(self.critic_net, "Q1"):
            return

        ae_config = config.autoencoder_config
        add_vector_observation = bool(getattr(config, "vector_observation", 0))
        encoder_type = str(getattr(config, "encoder_type", "vanilla_autoencoder")).lower()

        sample_in_features = self.actor_net.act_net.input_size
        encoder_input_channels = 3
        encoder_observation_size = getattr(ae_config, "observation_size", None)
        if encoder_observation_size is None:
            raise ValueError(
                "SACD encoder support requires autoencoder_config.observation_size."
            )

        if isinstance(encoder_observation_size, dict):
            encoder_observation_size = encoder_observation_size.get("image")

        if (
            isinstance(encoder_observation_size, tuple)
            and len(encoder_observation_size) == 3
            and add_vector_observation
        ):
            encoder_input_channels = int(encoder_observation_size[0])
            vector_obs_size = max(0, sample_in_features - int(getattr(ae_config, "latent_dim", 0)))
            add_vector_observation = vector_obs_size > 0

        if encoder_type in {"vanilla_autoencoder", "vanilla", "ae"}:
            autoencoder = VanillaAutoencoder(
                observation_size=encoder_observation_size,
                latent_dim=ae_config.latent_dim,
                num_layers=ae_config.num_layers,
                num_filters=ae_config.num_filters,
                kernel_size=ae_config.kernel_size,
                latent_lambda=getattr(ae_config, "latent_lambda", 1e-6),
                encoder_optimiser_params=getattr(ae_config, "encoder_optim_kwargs", None),
                decoder_optimiser_params=getattr(ae_config, "decoder_optim_kwargs", None),
            ).to(self.device)
            actor_encoder = autoencoder.encoder
            critic_encoder = autoencoder.encoder
            self.autoencoder = autoencoder
        else:
            actor_encoder = Encoder(
                observation_size=encoder_observation_size,
                latent_dim=ae_config.latent_dim,
                num_layers=ae_config.num_layers,
                num_filters=ae_config.num_filters,
                kernel_size=ae_config.kernel_size,
            ).to(self.device)
            if bool(getattr(config, "shared_conv_net", False)):
                critic_encoder = actor_encoder
            else:
                critic_encoder = Encoder(
                    observation_size=encoder_observation_size,
                    latent_dim=ae_config.latent_dim,
                    num_layers=ae_config.num_layers,
                    num_filters=ae_config.num_filters,
                    kernel_size=ae_config.kernel_size,
                ).to(self.device)

        self.actor_net = _DiscreteEncoderActor(
            encoder=actor_encoder,
            actor=self.actor_net,
            add_vector_observation=add_vector_observation,
        ).to(self.device)
        self.critic_net = _DiscreteEncoderCritic(
            encoder=critic_encoder,
            critic=self.critic_net,
            add_vector_observation=add_vector_observation,
        ).to(self.device)

        self.target_critic_net = copy.deepcopy(self.critic_net).to(self.device)
        self.target_critic_net.eval()

        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=config.actor_lr
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=config.critic_lr
        )

        self._set_film_support()

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        checkpoint: dict[str, Any] = {
            "actor": self.actor_net.state_dict(),
            "critic": self.critic_net.state_dict(),
            "target_critic": self.target_critic_net.state_dict(),
            "actor_optimizer": self.actor_net_optimiser.state_dict(),
            "critic_optimizer": self.critic_net_optimiser.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu().item(),
            "log_alpha_optimizer": self.log_alpha_optimizer.state_dict(),
            "learn_counter": self.learn_counter,
        }
        if self.autoencoder is not None:
            checkpoint["autoencoder"] = self.autoencoder.state_dict()

        torch.save(checkpoint, f"{filepath}/{filename}_checkpoint.pth")
        logging.info("models, optimisers, and training state have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        checkpoint = torch.load(
            f"{filepath}/{filename}_checkpoint.pth", map_location=self.device
        )

        self.actor_net.load_state_dict(checkpoint["actor"])
        self.critic_net.load_state_dict(checkpoint["critic"])
        self.target_critic_net.load_state_dict(checkpoint["target_critic"])

        self.actor_net_optimiser.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_net_optimiser.load_state_dict(checkpoint["critic_optimizer"])

        self.log_alpha.data = torch.tensor(checkpoint["log_alpha"], device=self.device)
        self.log_alpha_optimizer.load_state_dict(checkpoint["log_alpha_optimizer"])

        if self.autoencoder is not None and "autoencoder" in checkpoint:
            self.autoencoder.load_state_dict(checkpoint["autoencoder"])

        self.learn_counter = checkpoint.get("learn_counter", 0)
        logging.info("models, optimisers, and training state have been loaded...")
