"""
IMARL (Independent Multi-Agent Reinforcement Learning)
-------------------------------------------------------

IMARL provides a general framework for Independent Multi-Agent
Reinforcement Learning, where each agent is trained as a
separate single-agent learner using its own policy and value
functions.

Core Idea:
- Each agent treats other agents as part of the environment.
- No centralized critic or parameter sharing is enforced.
- Learning is fully decentralized.

Execution:
- At each timestep, the joint observation is split into
per-agent observations.
- Each agent independently selects its action:
      a_i = π_i(o_i)
- Joint action is returned to the environment.

Training:
- A shared multi-agent replay buffer stores joint transitions.
- During training, batches are sampled and split per agent.
- Each agent updates only using:
      (s_i, a_i, r_i, s'_i, done_i)
- No gradients or value targets depend on other agents'
  networks.

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

IMARL = N independent single-agent learners interacting
through a shared environment.
"""

import logging
import os
from abc import abstractmethod
from typing import Any, Generic, TypeVar

import numpy as np
import torch

import cares_reinforcement_learning.algorithm.policy as pol
import cares_reinforcement_learning.memory.memory_sampler as memory_sampler
import cares_reinforcement_learning.util.configurations as cfg
from cares_reinforcement_learning.algorithm.algorithm import (
    MARLAlgorithm,
    SARLAlgorithm,
)
from cares_reinforcement_learning.memory.memory_buffer import MARLMemoryBuffer
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import (
    MARLObservation,
    MARLObservationTensors,
    SARLObservation,
    SARLObservationTensors,
)

AgentType = TypeVar("AgentType", bound=SARLAlgorithm)


class IMARL(MARLAlgorithm[list[np.ndarray]], Generic[AgentType]):
    def __init__(
        self,
        agents: list[AgentType],
        config: cfg.AlgorithmConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        self.agent_networks = agents
        self.num_agents = len(agents)

    def act(
        self,
        observation: MARLObservation,
        evaluation: bool = False,
    ) -> ActionSample[list[np.ndarray]]:
        agent_states = observation.agent_states
        avail_actions = observation.avail_actions

        agent_ids = list(agent_states.keys())
        actions = []
        agent_extras = {}
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
            agent_extras[agent_name] = agent_sample.extras

        return ActionSample(action=actions, source="policy", extras=agent_extras)

    def _sample(self, memory_buffer: MARLMemoryBuffer) -> tuple[
        MARLObservationTensors,
        torch.Tensor,
        torch.Tensor,
        MARLObservationTensors,
        torch.Tensor,
        torch.Tensor,
        list[dict[str, Any]],
        np.ndarray,
    ]:
        return memory_sampler.sample(
            memory=memory_buffer,
            batch_size=self.batch_size,
            device=self.device,
            use_per_buffer=0,  # IMARL uses uniform sampling
        )

    @abstractmethod
    def update_agent_from_batch(
        self,
        *,
        agent_id: int,
        episode_context: EpisodeContext,
        observation_tensor: SARLObservationTensors,
        actions_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        next_observation_tensor: SARLObservationTensors,
        dones_tensor: torch.Tensor,
        weights_tensor: torch.Tensor,
        train_data: list[dict[str, Any]],
        indices: np.ndarray,
    ) -> dict[str, Any]: ...

    def train(
        self,
        memory_buffer: MARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:

        info: dict[str, Any] = {}

        (
            observation_tensor,
            actions_tensor,
            rewards_tensor,
            next_observation_tensor,
            dones_tensor,
            weights_tensor,
            train_data,
            indices,
        ) = self._sample(memory_buffer=memory_buffer)

        agent_states = observation_tensor.agent_states_tensor
        agent_ids = list(agent_states.keys())
        batch_size = len(indices)

        agent_train_data = []
        for i in range(self.num_agents):
            train_data_i = []
            agent_name = agent_ids[i]
            for j in range(batch_size):
                if agent_name in train_data[j]:  # Check if agent's data is present
                    train_data_i.append(train_data[j][agent_name])
            agent_train_data.append(train_data_i)

        for i, agent in enumerate(self.agent_networks):
            agent_name = agent_ids[i]
            states_i = SARLObservationTensors(
                vector_state_tensor=observation_tensor.agent_states_tensor[agent_name],
            )
            actions_i = actions_tensor[:, i, :]
            rewards_i = rewards_tensor[:, i]
            next_states_i = SARLObservationTensors(
                vector_state_tensor=next_observation_tensor.agent_states_tensor[
                    agent_name
                ],
            )
            dones_i = dones_tensor[:, i]
            train_data_i = agent_train_data[i]

            agent_i_info = self.update_agent_from_batch(
                agent_id=i,
                episode_context=episode_context,
                observation_tensor=states_i,
                actions_tensor=actions_i,
                rewards_tensor=rewards_i,
                next_observation_tensor=next_states_i,
                dones_tensor=dones_i,
                weights_tensor=weights_tensor,
                train_data=train_data_i,
                indices=indices,
            )
            for key, value in agent_i_info.items():
                info[f"{agent_name}_{key}"] = value

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


# The following classes are specific implementations of IMARL for different algorithms.
# They can be extended with algorithm-specific methods if needed in the future,
# but for now they simply call the base class constructor with the appropriate agent types and configurations.
# This feels over the top but it allows instance checking on algorithm type
# and keeps the option open for future algorithm-specific extensions without modifying the base class.


class IDDPG(IMARL[pol.DDPG]):
    def __init__(
        self,
        agents: list[pol.DDPG],
        config: cfg.IDDPGConfig,
        device: torch.device,
    ):
        super().__init__(agents=agents, config=config, device=device)

    def update_agent_from_batch(
        self,
        *,
        agent_id: int,
        episode_context: EpisodeContext,
        observation_tensor: SARLObservationTensors,
        actions_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        next_observation_tensor: SARLObservationTensors,
        dones_tensor: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, Any]:
        agent = self.agent_networks[agent_id]
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
        agents: list[pol.TD3],
        config: cfg.ITD3Config,
        device: torch.device,
    ):
        super().__init__(agents=agents, config=config, device=device)

    def update_agent_from_batch(
        self,
        *,
        agent_id: int,
        episode_context: EpisodeContext,
        observation_tensor: SARLObservationTensors,
        actions_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        next_observation_tensor: SARLObservationTensors,
        dones_tensor: torch.Tensor,
        weights_tensor: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, Any]:
        agent = self.agent_networks[agent_id]
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
        agents: list[pol.SAC],
        config: cfg.ISACConfig,
        device: torch.device,
    ):
        super().__init__(agents=agents, config=config, device=device)

    def update_agent_from_batch(
        self,
        *,
        agent_id: int,
        observation_tensor: SARLObservationTensors,
        actions_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        next_observation_tensor: SARLObservationTensors,
        dones_tensor: torch.Tensor,
        weights_tensor: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, Any]:
        agent = self.agent_networks[agent_id]
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
        agents: list[pol.PPO],
        config: cfg.IPPOConfig,
        device: torch.device,
    ):
        super().__init__(agents=agents, config=config, device=device)

    def update_agent_from_batch(
        self,
        *,
        agent_id: int,
        episode_context: EpisodeContext,
        observation_tensor: SARLObservationTensors,
        actions_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        next_observation_tensor: SARLObservationTensors,
        dones_tensor: torch.Tensor,
        train_data: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        agent = self.agent_networks[agent_id]
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
    def _sample(self, memory_buffer: MARLMemoryBuffer) -> tuple[
        MARLObservationTensors,
        torch.Tensor,
        torch.Tensor,
        MARLObservationTensors,
        torch.Tensor,
        torch.Tensor,
        list[dict[str, Any]],
        np.ndarray,
    ]:
        sample = memory_buffer.flush()
        batch_size = len(sample.experiences)

        return *memory_sampler.sample_to_tensors(sample, self.device), np.arange(
            batch_size
        )
