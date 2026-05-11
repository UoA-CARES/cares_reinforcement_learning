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


class IMARL(MARLAlgorithm[dict[str, np.ndarray]], Generic[AgentType]):
    def __init__(
        self,
        agents: dict[str, AgentType],
        config: cfg.AlgorithmConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        self.agent_networks = agents
        self.agent_ids = list(agents.keys())
        self.num_agents = len(agents)

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
            obs_i = agent_states[agent_name]
            avail_i = avail_actions[agent_name]

            agent_observation = SARLObservation(
                vector_state=obs_i,
                available_actions=avail_i,
            )

            agent_sample = self.agent_networks[agent_name].act(
                agent_observation, evaluation
            )
            actions[agent_name] = agent_sample.action
            agent_extras[agent_name] = agent_sample.extras

        return ActionSample(action=actions, source="policy", extras=agent_extras)

    def _sample(
        self, memory_buffer: MARLMemoryBuffer
    ) -> tuple[memory_sampler.MARLTensorSample, np.ndarray]:
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
        agent_name: str,
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

        sample_tensor, indices = self._sample(memory_buffer=memory_buffer)
        weights_tensor = sample_tensor.weights
        train_data = sample_tensor.train_data

        agent_states = sample_tensor.observation.agent_states
        agent_ids = list(agent_states.keys())
        batch_size = len(indices)

        agent_train_data = {}
        for agent_name in agent_ids:
            train_data_i = []
            for j in range(batch_size):
                if agent_name in train_data[j]:  # Check if agent's data is present
                    train_data_i.append(train_data[j][agent_name])
            agent_train_data[agent_name] = train_data_i

        for agent_name in agent_ids:
            states_i = SARLObservationTensors(
                vector_state=sample_tensor.observation.agent_states[agent_name],
            )
            actions_i = sample_tensor.action[agent_name]
            rewards_i = sample_tensor.reward[agent_name]

            next_states_i = SARLObservationTensors(
                vector_state=sample_tensor.next_observation.agent_states[agent_name],
            )

            dones_i = sample_tensor.done[agent_name]
            train_data_i = agent_train_data[agent_name]

            agent_i_info = self.update_agent_from_batch(
                agent_name=agent_name,
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

        metrics = list(agent_i_info.keys())
        for metric in metrics:
            values = [info[f"{agent_name}_{metric}"] for agent_name in agent_ids]
            info[f"mean_{metric}"] = float(np.mean(values))
            info[f"std_{metric}"] = float(np.std(values))
            info[f"max_{metric}"] = float(np.max(values))
            info[f"min_{metric}"] = float(np.min(values))

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        for agent_name in self.agent_ids:
            agent = self.agent_networks[agent_name]
            agent_filepath = os.path.join(filepath, f"{agent_name}")
            agent_filename = f"{filename}_agent_{agent_name}_checkpoint"
            agent.save_models(agent_filepath, agent_filename)

        logging.info("models and optimisers have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        for agent_name in self.agent_ids:
            agent = self.agent_networks[agent_name]
            agent_filepath = os.path.join(filepath, f"{agent_name}")
            agent_filename = f"{filename}_agent_{agent_name}_checkpoint"
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
        agents: dict[str, pol.DDPG],
        config: cfg.IDDPGConfig,
        device: torch.device,
    ):
        super().__init__(agents=agents, config=config, device=device)

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
        **kwargs: Any,
    ) -> dict[str, Any]:
        agent = self.agent_networks[agent_name]
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
        agents: dict[str, pol.TD3],
        config: cfg.ITD3Config,
        device: torch.device,
    ):
        super().__init__(agents=agents, config=config, device=device)

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
        **kwargs: Any,
    ) -> dict[str, Any]:
        agent = self.agent_networks[agent_name]
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
        agents: dict[str, pol.SAC],
        config: cfg.ISACConfig,
        device: torch.device,
    ):
        super().__init__(agents=agents, config=config, device=device)

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
        **kwargs: Any,
    ) -> dict[str, Any]:
        agent = self.agent_networks[agent_name]
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
        agents: dict[str, pol.PPO],
        config: cfg.IPPOConfig,
        device: torch.device,
    ):
        super().__init__(agents=agents, config=config, device=device)

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
        **kwargs: Any,
    ) -> dict[str, Any]:
        agent = self.agent_networks[agent_name]
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
