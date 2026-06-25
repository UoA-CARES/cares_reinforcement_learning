"""
IMARL (Independent Multi-Agent Reinforcement Learning)
-------------------------------------------------------

IMARL provides a general framework for Independent Multi-Agent
Reinforcement Learning, where each agent is trained as a
separate single-agent learner using its own policy and value
functions.

Core Idea:
- Each agent treats other agents as part of the environment.
- No centralized critic is used.
- Agents can either learn with one learner per agent or reuse one shared learner.
- Learning remains decentralized at the data interface.

Execution:
- At each timestep, the joint observation is split into
per-agent observations.
- Each agent independently selects its action:
      a_i = π_i(o_i)
- Joint action is returned to the environment.

Training:
- A shared multi-agent replay buffer stores joint transitions.
- During training, batches are sampled and split per agent.
- Each per-agent update uses only:
    (s_i, a_i, r_i, s'_i, done_i)
- When learners are shared, the same underlying learner is reused across agents,
  but each update still consumes only that agent's local transition slice.

Non-Stationarity:
- Since all agents learn simultaneously, from each agent's
  perspective the environment is non-stationary.
- IMARL does not correct for this (unlike centralized training
  approaches such as MADDPG or MAPPO).

Algorithm-Agnostic:
- Works with any single-agent algorithm:
      DDPG, TD3, SAC (off-policy)
      PPO (on-policy)
- Algorithm-specific subclasses simply forward batches
  to the corresponding single-agent update logic.

Replay / Sampling:
- Off-policy methods use uniform replay sampling.
- On-policy methods (e.g., PPO) override sampling to
  flush trajectories per update.

Rationale:
- Simple and scalable.
- No need for centralized state or joint action critics.
- Useful baseline for MARL comparison.
- Easily extensible to new single-agent algorithms.

IMARL = N local single-agent update streams interacting
through a shared environment, with either per-agent or shared parameters.
"""

import logging
import os
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

import numpy as np
import numpy.typing as npt
import torch

import cares_reinforcement_learning.algorithm.policy as pol
import cares_reinforcement_learning.memory.memory_sampler as memory_sampler
import cares_reinforcement_learning.algorithm.configurations as cfg
from cares_reinforcement_learning.algorithm.algorithm import (
    MARLAlgorithm,
    SARLAlgorithm,
)
from cares_reinforcement_learning.memory.memory_buffer import MARLMemoryBuffer
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import (
    MARLObservation,
    SARLObservation,
    SARLObservationTensors,
)

AgentType = TypeVar("AgentType", bound=SARLAlgorithm)

IMARLConfig = cfg.IDDPGConfig | cfg.ITD3Config | cfg.ISACConfig | cfg.IPPOConfig


@dataclass(frozen=True, slots=True)
class IMARLUpdateBatch:
    observation_tensor: SARLObservationTensors
    actions_tensor: torch.Tensor
    rewards_tensor: torch.Tensor
    next_observation_tensor: SARLObservationTensors
    dones_tensor: torch.Tensor
    weights_tensor: torch.Tensor
    train_data: list[dict[str, Any]]
    indices: npt.NDArray[np.int_]


class IMARL(MARLAlgorithm[dict[str, np.ndarray]], Generic[AgentType]):
    def __init__(
        self,
        learning_units: dict[str, AgentType],
        agent_id_to_learning_unit_id: dict[str, str],
        learning_unit_id_to_agent_ids: dict[str, list[str]],
        agent_identity_vectors: dict[str, npt.NDArray[np.float32]],
        team_identity_vectors: dict[str, npt.NDArray[np.float32]],
        agent_id_to_team_id: dict[str, str],
        config: IMARLConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        self.learning_units = learning_units
        self.agent_id_to_learning_unit_id = agent_id_to_learning_unit_id
        self.learning_unit_id_to_agent_ids = learning_unit_id_to_agent_ids
        self.agent_identity_vectors = agent_identity_vectors
        self.team_identity_vectors = team_identity_vectors
        self.agent_id_to_team_id = agent_id_to_team_id
        self.use_agent_id = config.use_agent_id
        self.use_team_id = config.use_team_id
        self.agent_ids = list(agent_id_to_learning_unit_id.keys())
        self.learning_unit_ids = list(learning_units.keys())
        self.num_agents = len(self.agent_ids)
        self._identity_vector_cache: dict[str, npt.NDArray[np.float32]] = {}
        self._identity_tensor_cache: dict[
            tuple[str, str, torch.dtype], torch.Tensor
        ] = {}

    def _get_agent_network(self, agent_name: str) -> AgentType:
        learning_unit_id = self.agent_id_to_learning_unit_id[agent_name]
        return self.learning_units[learning_unit_id]

    def _uses_shared_identity_conditioning(self, agent_name: str) -> bool:
        learning_unit_id = self.agent_id_to_learning_unit_id[agent_name]
        num_controlled_agents = len(
            self.learning_unit_id_to_agent_ids[learning_unit_id]
        )
        return bool(
            num_controlled_agents > 1 and (self.use_agent_id or self.use_team_id)
        )

    def _get_identity_vector(self, agent_id: str) -> npt.NDArray[np.float32]:
        cached_vector = self._identity_vector_cache.get(agent_id)
        if cached_vector is not None:
            return cached_vector

        identity_parts: list[npt.NDArray[np.float32]] = []
        if self.use_team_id:
            team_id = self.agent_id_to_team_id[agent_id]
            identity_parts.append(self.team_identity_vectors[team_id])
        if self.use_agent_id:
            identity_parts.append(self.agent_identity_vectors[agent_id])

        if identity_parts:
            identity_vector = np.concatenate(identity_parts, axis=-1)
        else:
            identity_vector = np.empty((0,), dtype=np.float32)

        self._identity_vector_cache[agent_id] = identity_vector
        return identity_vector

    def augment_observation(
        self,
        observation: npt.NDArray[np.generic] | torch.Tensor,
        agent_id: str,
    ) -> npt.NDArray[np.generic] | torch.Tensor:
        if not self._uses_shared_identity_conditioning(agent_id):
            return observation

        identity_vector = self._get_identity_vector(agent_id)
        if identity_vector.size == 0:
            return observation

        if isinstance(observation, torch.Tensor):
            cache_key = (agent_id, str(observation.device), observation.dtype)
            identity_tensor = self._identity_tensor_cache.get(cache_key)
            if identity_tensor is None:
                identity_tensor = torch.as_tensor(
                    identity_vector,
                    dtype=observation.dtype,
                    device=observation.device,
                )
                self._identity_tensor_cache[cache_key] = identity_tensor

            tensor_expand_shape = observation.shape[:-1] + (identity_tensor.shape[0],)
            identity_tensor = identity_tensor.expand(tensor_expand_shape)
            return torch.cat((observation, identity_tensor), dim=-1)

        identity_array = identity_vector.astype(observation.dtype, copy=False)
        array_expand_shape = observation.shape[:-1] + (identity_array.shape[0],)
        identity_array = np.broadcast_to(identity_array, array_expand_shape)
        return np.concatenate((observation, identity_array), axis=-1)

    def act(
        self,
        observation: MARLObservation,
        evaluation: bool = False,
    ) -> ActionSample[dict[str, np.ndarray]]:
        agent_states = observation.agent_states
        avail_actions = observation.available_actions

        actions = {}
        agent_extras = {}
        for agent_name in self.agent_ids:
            agent_network = self._get_agent_network(agent_name)
            obs_i = cast(
                np.ndarray,
                self.augment_observation(agent_states[agent_name], agent_name),
            )
            avail_i = avail_actions[agent_name]

            agent_observation = SARLObservation(
                vector_state=obs_i,
                available_actions=avail_i,
            )

            agent_sample = agent_network.act(agent_observation, evaluation)
            actions[agent_name] = agent_sample.action
            agent_extras[agent_name] = agent_sample.extras

        return ActionSample(action=actions, source="policy", extras=agent_extras)

    def _sample(
        self, memory_buffer: MARLMemoryBuffer
    ) -> tuple[memory_sampler.MARLTensorSample, npt.NDArray[np.int_]]:
        return memory_sampler.sample(
            memory=memory_buffer,
            batch_size=self.batch_size,
            device=self.device,
            use_per_buffer=0,  # IMARL uses uniform sampling
        )

    def _build_agent_update_batch(
        self,
        *,
        agent_name: str,
        sample_tensor: memory_sampler.MARLTensorSample,
        agent_train_data: dict[str, list[dict[str, Any]]],
        indices: npt.NDArray[np.int_],
        weights_tensor: torch.Tensor,
    ) -> IMARLUpdateBatch:
        observation_tensor = SARLObservationTensors(
            vector_state=cast(
                torch.Tensor,
                self.augment_observation(
                    sample_tensor.observation.agent_states[agent_name], agent_name
                ),
            ),
        )

        next_observation_tensor = SARLObservationTensors(
            vector_state=cast(
                torch.Tensor,
                self.augment_observation(
                    sample_tensor.next_observation.agent_states[agent_name],
                    agent_name,
                ),
            ),
        )

        return IMARLUpdateBatch(
            observation_tensor=observation_tensor,
            actions_tensor=sample_tensor.action[agent_name],
            rewards_tensor=sample_tensor.reward[agent_name],
            next_observation_tensor=next_observation_tensor,
            dones_tensor=sample_tensor.done[agent_name],
            weights_tensor=weights_tensor,
            train_data=agent_train_data[agent_name],
            indices=indices,
        )

    def _merge_agent_update_batches(
        self, agent_batches: list[IMARLUpdateBatch]
    ) -> IMARLUpdateBatch:
        if len(agent_batches) == 1:
            return agent_batches[0]

        observation_tensor = SARLObservationTensors(
            vector_state=torch.cat(
                [batch.observation_tensor.vector_state for batch in agent_batches],
                dim=0,
            )
        )
        actions_tensor = torch.cat(
            [batch.actions_tensor for batch in agent_batches],
            dim=0,
        )
        rewards_tensor = torch.cat(
            [batch.rewards_tensor for batch in agent_batches],
            dim=0,
        )
        next_observation_tensor = SARLObservationTensors(
            vector_state=torch.cat(
                [batch.next_observation_tensor.vector_state for batch in agent_batches],
                dim=0,
            )
        )
        dones_tensor = torch.cat(
            [batch.dones_tensor for batch in agent_batches],
            dim=0,
        )
        weights_tensor = torch.cat(
            [batch.weights_tensor for batch in agent_batches],
            dim=0,
        )

        train_data: list[dict[str, Any]] = []
        for batch in agent_batches:
            train_data.extend(batch.train_data)

        indices = np.concatenate([batch.indices for batch in agent_batches], axis=0)

        return IMARLUpdateBatch(
            observation_tensor=observation_tensor,
            actions_tensor=actions_tensor,
            rewards_tensor=rewards_tensor,
            next_observation_tensor=next_observation_tensor,
            dones_tensor=dones_tensor,
            weights_tensor=weights_tensor,
            train_data=train_data,
            indices=indices,
        )

    @abstractmethod
    def update_agent_from_batch(
        self,
        *,
        agent_name: str,
        episode_context: EpisodeContext,
        observation_tensor: SARLObservationTensors,
        actions_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        next_observation_tensor: SARLObservationTensors,
        dones_tensor: torch.Tensor,
        weights_tensor: torch.Tensor,
        train_data: list[dict[str, Any]],
        indices: npt.NDArray[np.int_],
    ) -> dict[str, Any]: ...

    def train(
        self,
        memory_buffer: MARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:

        info: dict[str, Any] = {}
        update_infos: dict[str, dict[str, Any]] = {}

        sample_tensor, indices = self._sample(memory_buffer=memory_buffer)
        weights_tensor = sample_tensor.weights
        train_data = sample_tensor.train_data

        agent_ids = list(sample_tensor.observation.agent_states.keys())
        batch_size = len(indices)

        agent_train_data: dict[str, list[dict[str, Any]]] = {}
        for agent_name in agent_ids:
            train_data_i: list[dict[str, Any]] = []
            for j in range(batch_size):
                if agent_name in train_data[j]:  # Check if agent's data is present
                    train_data_i.append(train_data[j][agent_name])
            agent_train_data[agent_name] = train_data_i

        for (
            learning_unit_id,
            controlled_agent_ids,
        ) in self.learning_unit_id_to_agent_ids.items():
            agent_batches = [
                self._build_agent_update_batch(
                    agent_name=agent_name,
                    sample_tensor=sample_tensor,
                    agent_train_data=agent_train_data,
                    indices=indices,
                    weights_tensor=weights_tensor,
                )
                for agent_name in controlled_agent_ids
            ]
            batch = self._merge_agent_update_batches(agent_batches)
            representative_agent_name = controlled_agent_ids[0]

            update_info = self.update_agent_from_batch(
                agent_name=representative_agent_name,
                episode_context=episode_context,
                observation_tensor=batch.observation_tensor,
                actions_tensor=batch.actions_tensor,
                rewards_tensor=batch.rewards_tensor,
                next_observation_tensor=batch.next_observation_tensor,
                dones_tensor=batch.dones_tensor,
                weights_tensor=batch.weights_tensor,
                train_data=batch.train_data,
                indices=batch.indices,
            )
            update_infos[learning_unit_id] = update_info

            for key, value in update_info.items():
                info[f"{learning_unit_id}_{key}"] = value

            info[f"{learning_unit_id}_num_controlled_agents"] = len(
                controlled_agent_ids
            )
            info[f"{learning_unit_id}_effective_batch_size"] = int(
                batch.actions_tensor.shape[0]
            )

        metric_names = sorted(
            {
                metric
                for update_info in update_infos.values()
                for metric in update_info.keys()
            }
        )
        for metric in metric_names:
            values = [
                update_info[metric]
                for update_info in update_infos.values()
                if metric in update_info
            ]
            numeric_values: list[float] = [
                cast(float, value)
                for value in values
                if isinstance(value, (int, float, np.integer, np.floating))
                and not isinstance(value, bool)
            ]

            if numeric_values:
                numeric_array = np.asarray(numeric_values, dtype=np.float32)
                info[f"mean_{metric}"] = float(np.mean(numeric_array))
                info[f"std_{metric}"] = float(np.std(numeric_array))
                info[f"max_{metric}"] = float(np.max(numeric_array))
                info[f"min_{metric}"] = float(np.min(numeric_array))

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        for learning_unit_id in self.learning_unit_ids:
            agent = self.learning_units[learning_unit_id]
            agent_filepath = os.path.join(filepath, f"{learning_unit_id}")
            agent_filename = f"{filename}_agent_{learning_unit_id}_checkpoint"
            agent.save_models(agent_filepath, agent_filename)

        logging.info("models and optimisers have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        for learning_unit_id in self.learning_unit_ids:
            agent = self.learning_units[learning_unit_id]
            agent_filepath = os.path.join(filepath, f"{learning_unit_id}")
            agent_filename = f"{filename}_agent_{learning_unit_id}_checkpoint"
            agent.load_models(agent_filepath, agent_filename)

        logging.info("models and optimisers have been loaded...")


# The following classes are specific implementations of IMARL for different algorithms.
# They can be extended with algorithm-specific methods if needed in the future,
# but for now they simply call the base class constructor with the appropriate agent types and configurations.
# This feels over the top but it allows instance checking on algorithm type
# and keeps the option open for future algorithm-specific extensions without modifying the base class.


class IDDPG(IMARL[pol.DDPG]):
    def __init__(
        self,
        learning_units: dict[str, pol.DDPG],
        agent_id_to_learning_unit_id: dict[str, str],
        learning_unit_id_to_agent_ids: dict[str, list[str]],
        agent_identity_vectors: dict[str, npt.NDArray[np.float32]],
        team_identity_vectors: dict[str, npt.NDArray[np.float32]],
        agent_id_to_team_id: dict[str, str],
        config: cfg.IDDPGConfig,
        device: torch.device,
    ):
        super().__init__(
            learning_units=learning_units,
            agent_id_to_learning_unit_id=agent_id_to_learning_unit_id,
            learning_unit_id_to_agent_ids=learning_unit_id_to_agent_ids,
            agent_identity_vectors=agent_identity_vectors,
            team_identity_vectors=team_identity_vectors,
            agent_id_to_team_id=agent_id_to_team_id,
            config=config,
            device=device,
        )

    def update_agent_from_batch(
        self,
        *,
        agent_name: str,
        episode_context: EpisodeContext,
        observation_tensor: SARLObservationTensors,
        actions_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        next_observation_tensor: SARLObservationTensors,
        dones_tensor: torch.Tensor,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        agent = self._get_agent_network(agent_name)
        return agent.update_from_batch(
            episode_context=episode_context,
            observation_tensor=observation_tensor,
            actions_tensor=actions_tensor,
            rewards_tensor=rewards_tensor,
            next_observation_tensor=next_observation_tensor,
            dones_tensor=dones_tensor,
        )


class ITD3(IMARL[pol.TD3]):
    def __init__(
        self,
        learning_units: dict[str, pol.TD3],
        agent_id_to_learning_unit_id: dict[str, str],
        learning_unit_id_to_agent_ids: dict[str, list[str]],
        agent_identity_vectors: dict[str, npt.NDArray[np.float32]],
        team_identity_vectors: dict[str, npt.NDArray[np.float32]],
        agent_id_to_team_id: dict[str, str],
        config: cfg.ITD3Config,
        device: torch.device,
    ):
        super().__init__(
            learning_units=learning_units,
            agent_id_to_learning_unit_id=agent_id_to_learning_unit_id,
            learning_unit_id_to_agent_ids=learning_unit_id_to_agent_ids,
            agent_identity_vectors=agent_identity_vectors,
            team_identity_vectors=team_identity_vectors,
            agent_id_to_team_id=agent_id_to_team_id,
            config=config,
            device=device,
        )

    def update_agent_from_batch(
        self,
        *,
        agent_name: str,
        episode_context: EpisodeContext,
        observation_tensor: SARLObservationTensors,
        actions_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        next_observation_tensor: SARLObservationTensors,
        dones_tensor: torch.Tensor,
        weights_tensor: torch.Tensor,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        agent = self._get_agent_network(agent_name)
        info, _ = agent.update_from_batch(
            episode_context=episode_context,
            observation_tensor=observation_tensor,
            actions_tensor=actions_tensor,
            rewards_tensor=rewards_tensor,
            next_observation_tensor=next_observation_tensor,
            dones_tensor=dones_tensor,
            weights_tensor=weights_tensor,
        )
        return info


class ISAC(IMARL[pol.SAC]):
    def __init__(
        self,
        learning_units: dict[str, pol.SAC],
        agent_id_to_learning_unit_id: dict[str, str],
        learning_unit_id_to_agent_ids: dict[str, list[str]],
        agent_identity_vectors: dict[str, npt.NDArray[np.float32]],
        team_identity_vectors: dict[str, npt.NDArray[np.float32]],
        agent_id_to_team_id: dict[str, str],
        config: cfg.ISACConfig,
        device: torch.device,
    ):
        super().__init__(
            learning_units=learning_units,
            agent_id_to_learning_unit_id=agent_id_to_learning_unit_id,
            learning_unit_id_to_agent_ids=learning_unit_id_to_agent_ids,
            agent_identity_vectors=agent_identity_vectors,
            team_identity_vectors=team_identity_vectors,
            agent_id_to_team_id=agent_id_to_team_id,
            config=config,
            device=device,
        )

    def update_agent_from_batch(
        self,
        *,
        agent_name: str,
        observation_tensor: SARLObservationTensors,
        actions_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        next_observation_tensor: SARLObservationTensors,
        dones_tensor: torch.Tensor,
        weights_tensor: torch.Tensor,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        agent = self._get_agent_network(agent_name)
        info, _ = agent.update_from_batch(
            observation_tensor=observation_tensor,
            actions_tensor=actions_tensor,
            rewards_tensor=rewards_tensor,
            next_observation_tensor=next_observation_tensor,
            dones_tensor=dones_tensor,
            weights_tensor=weights_tensor,
        )
        return info


class IPPO(IMARL[pol.PPO]):
    def __init__(
        self,
        learning_units: dict[str, pol.PPO],
        agent_id_to_learning_unit_id: dict[str, str],
        learning_unit_id_to_agent_ids: dict[str, list[str]],
        agent_identity_vectors: dict[str, npt.NDArray[np.float32]],
        team_identity_vectors: dict[str, npt.NDArray[np.float32]],
        agent_id_to_team_id: dict[str, str],
        config: cfg.IPPOConfig,
        device: torch.device,
    ):
        super().__init__(
            learning_units=learning_units,
            agent_id_to_learning_unit_id=agent_id_to_learning_unit_id,
            learning_unit_id_to_agent_ids=learning_unit_id_to_agent_ids,
            agent_identity_vectors=agent_identity_vectors,
            team_identity_vectors=team_identity_vectors,
            agent_id_to_team_id=agent_id_to_team_id,
            config=config,
            device=device,
        )

    def update_agent_from_batch(
        self,
        *,
        agent_name: str,
        episode_context: EpisodeContext,
        observation_tensor: SARLObservationTensors,
        actions_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        next_observation_tensor: SARLObservationTensors,
        dones_tensor: torch.Tensor,
        train_data: list[dict[str, Any]],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        agent = self._get_agent_network(agent_name)
        return agent.update_from_batch(
            episode_context=episode_context,
            observation_tensor=observation_tensor,
            actions_tensor=actions_tensor,
            rewards_tensor=rewards_tensor,
            next_observation_tensor=next_observation_tensor,
            dones_tensor=dones_tensor,
            train_data=train_data,
        )

    # PPO flushes the buffer override the default sampling method
    def _sample(
        self, memory_buffer: MARLMemoryBuffer
    ) -> tuple[memory_sampler.MARLTensorSample, np.ndarray]:
        sample = memory_buffer.flush()
        batch_size = len(sample.experiences)

        return memory_sampler.sample_to_tensors(sample, self.device), np.arange(
            batch_size
        )
