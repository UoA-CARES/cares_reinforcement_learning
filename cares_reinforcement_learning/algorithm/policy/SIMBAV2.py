"""
SimbaV2
-------

Hyperspherical Normalization for Scalable Deep Reinforcement Learning.

SimbaV2 uses SAC as its underlying algorithm and introduces:

- Running observation normalization.
- Hyperspherical feature and weight normalization.
- Learnable feature scalers and LERP residual connections.
- Categorical distributional critics.
- Running-return reward scaling.
- Unit-norm weight projection after gradient updates.
"""

from collections.abc import Callable
from dataclasses import replace
from typing import Any, Literal, cast

import numpy as np
import torch

from cares_reinforcement_learning.algorithm.configurations import SIMBAV2Config
from cares_reinforcement_learning.algorithm.observation_normaliser import (
    RunningObservationNormaliser,
)
from cares_reinforcement_learning.algorithm.policy import SAC
from cares_reinforcement_learning.algorithm.reward_normaliser import (
    RunningRewardNormaliser,
)
from cares_reinforcement_learning.networks import functional as fnc
from cares_reinforcement_learning.networks.SIMBAv2 import Actor, Critic
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.experience import (
    Experience,
    SingleAgentExperience,
)
from cares_reinforcement_learning.types.observation import (
    SARLObservation,
    SARLObservationTensors,
)


class SIMBAV2(SAC):
    actor_net: Actor
    critic_net: Critic
    target_critic_net: Critic

    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: SIMBAV2Config,
        device: torch.device,
    ):
        super().__init__(
            actor_network=actor_network,  # type: ignore[arg-type]
            critic_network=critic_network,  # type: ignore[arg-type]
            config=config,
            device=device,
        )

        self.observation_normaliser = RunningObservationNormaliser(
            shape=(actor_network.input_size,),
        )

        self.reward_normaliser = RunningRewardNormaliser(
            gamma=config.gamma,
            g_max=config.normalised_g_max,
            normalisation_epsilon=config.reward_normalisation_epsilon,
        )

        self.num_atoms = config.num_atoms
        self.v_min = config.v_min
        self.v_max = config.v_max

    def _calculate_value(
        self,
        state: SARLObservation,
        action: np.ndarray,
    ) -> float:
        normalised_state = replace(
            state,
            vector_state=self.observation_normaliser.normalise(state.vector_state),
        )

        normalised_value = super()._calculate_value(
            state=normalised_state,
            action=action,
        )

        reward_scale = self.reward_normaliser.get_statistics()["reward_norm_scale"]

        return normalised_value * reward_scale

    def act(
        self,
        observation: SARLObservation,
        evaluation: bool = False,
    ) -> ActionSample[np.ndarray]:
        if not evaluation:
            self.observation_normaliser.update(observation.vector_state)

        normalised_observation = replace(
            observation,
            vector_state=self.observation_normaliser.normalise(
                observation.vector_state
            ),
        )

        return super().act(
            observation=normalised_observation,
            evaluation=evaluation,
        )

    def explore(
        self,
        observation: SARLObservation,
        action_sampler: Callable[[], np.ndarray],
    ) -> ActionSample[np.ndarray]:
        self.observation_normaliser.update(observation.vector_state)

        return super().explore(
            observation=observation,
            action_sampler=action_sampler,
        )

    def on_environment_step(
        self,
        experience: Experience[SARLObservation],
    ) -> None:
        """
        Update reward statistics exactly once per environment transition.
        """
        sarl_experience = cast(SingleAgentExperience, experience)

        self.reward_normaliser.update(
            reward=sarl_experience.reward,
            done=sarl_experience.done_flag or sarl_experience.truncated_flag,
        )

    def _project_distribution(
        self,
        target_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_log_pi: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project the SimbaV2 distributional Bellman target onto the
        fixed categorical support.
        """
        support = self.critic_net.support

        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)
        next_log_pi = next_log_pi.reshape(-1, 1)

        target_probabilities = target_log_probs.exp()

        entropy_term = self.alpha * next_log_pi

        target_atoms = rewards + self.gamma * (support.unsqueeze(0) - entropy_term) * (
            1.0 - dones
        )

        target_atoms = target_atoms.clamp(
            self.v_min,
            self.v_max,
        )

        atom_delta = (self.v_max - self.v_min) / (self.num_atoms - 1)

        projection_positions = ((target_atoms - self.v_min) / atom_delta).clamp(
            0,
            self.num_atoms - 1,
        )

        lower = projection_positions.floor().long()
        upper = projection_positions.ceil().long()

        projected = torch.zeros_like(target_probabilities)

        lower_weight = upper.float() - projection_positions + (lower == upper).float()

        upper_weight = projection_positions - lower.float()

        projected.scatter_add_(
            dim=1,
            index=lower,
            src=target_probabilities * lower_weight,
        )

        projected.scatter_add_(
            dim=1,
            index=upper,
            src=target_probabilities * upper_weight,
        )

        return projected

    def _update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[dict[str, Any], np.ndarray]:
        info: dict[str, Any] = {}

        with torch.no_grad():
            with fnc.evaluating(self.actor_net):
                next_actions, next_log_pi, _ = self.actor_net(next_states)

            target_q1, target_log_probs1, target_q2, target_log_probs2 = (
                self.target_critic_net.forward_distribution(next_states, next_actions)
            )

            # Clipped double-Q selection is based on expected Q.
            # Use the full categorical distribution belonging to the
            # lower-valued critic rather than taking an atom-wise min.
            choose_q2 = target_q2 < target_q1

            target_log_probs = torch.where(
                choose_q2, target_log_probs2, target_log_probs1
            )

            target_distribution = self._project_distribution(
                target_log_probs=target_log_probs,
                rewards=rewards,
                dones=dones,
                next_log_pi=next_log_pi,
            )

            q_target = (target_distribution * self.critic_net.support.unsqueeze(0)).sum(
                dim=-1, keepdim=True
            )

        q1, log_probs1, q2, log_probs2 = self.critic_net.forward_distribution(
            states, actions
        )

        critic_loss_one = -(target_distribution * log_probs1).sum(dim=-1, keepdim=True)

        critic_loss_two = -(target_distribution * log_probs2).sum(dim=-1, keepdim=True)

        critic_loss_one = (critic_loss_one * weights).mean()

        critic_loss_two = (critic_loss_two * weights).mean()

        critic_loss_total = critic_loss_one + critic_loss_two

        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        # SimbaV2 explicitly constrains dense weight vectors to
        # the unit hypersphere after every gradient update.
        self.critic_net.project_weights()

        td_error_one = (q1 - q_target).abs()
        td_error_two = (q2 - q_target).abs()

        priorities = (
            torch.maximum(td_error_one, td_error_two)
            .clamp(self.min_priority)
            .pow(self.per_alpha)
            .detach()
            .cpu()
            .numpy()
            .flatten()
        )

        with torch.no_grad():
            info["q1_mean"] = q1.mean().item()
            info["q2_mean"] = q2.mean().item()

            info["q_twin_gap_abs_mean"] = (q1 - q2).abs().mean().item()

            info["target_q1_mean"] = target_q1.mean().item()
            info["target_q2_mean"] = target_q2.mean().item()

            info["target_q_twin_gap_abs_mean"] = (
                (target_q1 - target_q2).abs().mean().item()
            )

            info["q_target_mean"] = q_target.mean().item()
            info["q_target_std"] = q_target.std().item()

            td1 = q1 - q_target
            td2 = q2 - q_target

            info["td1_mean"] = td1.mean().item()
            info["td1_std"] = td1.std().item()
            info["td1_abs_mean"] = td1.abs().mean().item()

            info["td2_mean"] = td2.mean().item()
            info["td2_std"] = td2.std().item()
            info["td2_abs_mean"] = td2.abs().mean().item()

            info["critic_loss_one"] = critic_loss_one.item()
            info["critic_loss_two"] = critic_loss_two.item()
            info["critic_loss_total"] = critic_loss_total.item()

            info["reward_scaled_mean"] = rewards.mean().item()
            info["reward_scaled_std"] = rewards.std().item()

        return info, priorities

    def _update_actor_alpha(
        self,
        states: torch.Tensor,
        weights: torch.Tensor,
    ) -> dict[str, Any]:
        info = super()._update_actor_alpha(states=states, weights=weights)

        self.actor_net.project_weights()

        return info

    def update_from_batch(
        self,
        observation_tensor: SARLObservationTensors,
        actions_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        next_observation_tensor: SARLObservationTensors,
        dones_tensor: torch.Tensor,
        weights_tensor: torch.Tensor,
    ) -> tuple[dict[str, Any], np.ndarray]:
        observation_tensor = replace(
            observation_tensor,
            vector_state=self.observation_normaliser.normalise_tensor(
                observation_tensor.vector_state
            ),
        )

        next_observation_tensor = replace(
            next_observation_tensor,
            vector_state=self.observation_normaliser.normalise_tensor(
                next_observation_tensor.vector_state
            ),
        )

        rewards_tensor = self.reward_normaliser.normalise_tensor(rewards_tensor)

        return super().update_from_batch(
            observation_tensor=observation_tensor,
            actions_tensor=actions_tensor,
            rewards_tensor=rewards_tensor,
            next_observation_tensor=next_observation_tensor,
            dones_tensor=dones_tensor,
            weights_tensor=weights_tensor,
        )

    def get_statistics(
        self,
        episode_context,
    ) -> dict[str, float]:
        statistics = super().get_statistics(episode_context)

        statistics.update(self.observation_normaliser.get_statistics())

        statistics.update(self.reward_normaliser.get_statistics())

        return statistics

    def save_models(
        self,
        filepath: str,
        filename: str,
    ) -> None:
        super().save_models(filepath=filepath, filename=filename)

        checkpoint_path = f"{filepath}/{filename}_checkpoint.pth"

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        checkpoint["observation_normaliser"] = self.observation_normaliser.state_dict()

        checkpoint["reward_normaliser"] = self.reward_normaliser.state_dict()

        torch.save(checkpoint, checkpoint_path)

    def load_models(
        self,
        filepath: str,
        filename: str,
        load_mode: Literal[
            "resume",
            "transfer",
        ] = "resume",
    ) -> None:
        super().load_models(filepath=filepath, filename=filename, load_mode=load_mode)

        if load_mode == "transfer":
            return

        checkpoint_path = f"{filepath}/{filename}_checkpoint.pth"

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        self.observation_normaliser.load_state_dict(
            checkpoint["observation_normaliser"]
        )

        self.reward_normaliser.load_state_dict(checkpoint["reward_normaliser"])
