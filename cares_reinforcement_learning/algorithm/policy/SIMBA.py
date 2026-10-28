"""
SimBa (Simplicity Bias for Scaling Up Parameters in Deep Reinforcement Learning)
-------------------------------------------------------------------------------

Original Paper: https://arxiv.org/abs/2410.09754

SimBa is a neural network architecture for deep reinforcement learning
designed to improve scalability, sample efficiency, and network plasticity
through architectural simplicity bias.

Core Idea:
- Normalize observations using running statistics.
- Use residual feedforward blocks with pre-LayerNorm.
- Maintain a direct linear pathway through each residual block.
- Apply post-LayerNorm before the final prediction layer.
- Scale critic capacity while maintaining stable representations.

Architecture:
- Running Statistics Normalization (RSNorm) on observations.
- Linear input embedding.
- Residual blocks of the form:

      x' = x + MLP(LayerNorm(x))

- Each residual MLP uses an inverted bottleneck:

      hidden_dim -> 4 * hidden_dim -> hidden_dim

- ReLU activation inside the residual branch.
- Final LayerNorm before policy or value prediction.

SAC + SimBa:
- Uses SAC as the underlying reinforcement learning algorithm.
- Actor uses one SimBa residual block with hidden dimension 128.
- Critic uses two SimBa residual blocks with hidden dimension 512.
- Uses AdamW and increased update-to-data ratio.
- Observation statistics are updated online while raw observations
  remain stored in replay.

Key Behaviour:
- Residual pathways encourage simpler functional representations.
- Layer normalization controls feature magnitude.
- RSNorm prevents observation dimensions with large scales from
  dominating learning.
- Larger critic networks can be used without the degradation often
  observed when scaling standard MLP architectures.
- SimBa has been shown to better preserve network plasticity during
  reinforcement learning.

SimBa = SAC + RSNorm + Pre-LN Residual MLPs + Post-LN
        for scalable deep reinforcement learning.
"""

from collections.abc import Callable
from dataclasses import replace
from typing import Literal

import numpy as np
import torch

from cares_reinforcement_learning.algorithm.configurations import SIMBAConfig
from cares_reinforcement_learning.algorithm.observation_normaliser import (
    RunningObservationNormaliser,
)
from cares_reinforcement_learning.algorithm.policy import SAC
from cares_reinforcement_learning.networks.common import (
    TanhGaussianPolicy,
    TwinQNetwork,
)
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.observation import (
    SARLObservation,
    SARLObservationTensors,
)


class SIMBA(SAC):
    def __init__(
        self,
        actor_network: TanhGaussianPolicy,
        critic_network: TwinQNetwork,
        config: SIMBAConfig,
        device: torch.device,
    ):
        super().__init__(
            actor_network=actor_network,
            critic_network=critic_network,
            config=config,
            device=device,
        )

        self.observation_normaliser = RunningObservationNormaliser(
            shape=(actor_network.input_size,),
        )

    def _calculate_value(
        self,
        state: SARLObservation,
        action: np.ndarray,
    ) -> float:
        normalised_state = replace(
            state,
            vector_state=self.observation_normaliser.normalise(state.vector_state),
        )

        return super()._calculate_value(
            state=normalised_state,
            action=action,
        )

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

    def update_from_batch(
        self,
        observation_tensor: SARLObservationTensors,
        actions_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        next_observation_tensor: SARLObservationTensors,
        dones_tensor: torch.Tensor,
        weights_tensor: torch.Tensor,
    ) -> tuple[dict, np.ndarray]:

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

        return statistics

    def save_models(self, filepath: str, filename: str) -> None:
        super().save_models(
            filepath=filepath,
            filename=filename,
        )

        checkpoint_path = f"{filepath}/{filename}_checkpoint.pth"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        checkpoint["observation_normaliser"] = self.observation_normaliser.state_dict()

        torch.save(checkpoint, checkpoint_path)

    def load_models(
        self,
        filepath: str,
        filename: str,
        load_mode: Literal["resume", "transfer"] = "resume",
    ) -> None:
        super().load_models(filepath=filepath, filename=filename, load_mode=load_mode)

        if load_mode == "transfer":
            return

        checkpoint_path = f"{filepath}/{filename}_checkpoint.pth"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        self.observation_normaliser.load_state_dict(
            checkpoint["observation_normaliser"]
        )
