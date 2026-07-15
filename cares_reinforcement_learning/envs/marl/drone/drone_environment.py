from functools import cached_property
from typing import Any, cast

import numpy as np
from gymnasium import spaces
from typing_extensions import Literal

from cares_reinforcement_learning.envs.configurations import MARLDroneConfig
from cares_reinforcement_learning.envs.marl.marl_environment import MARLEnvironment
from cares_reinforcement_learning.types.experience import MultiAgentExperience
from cares_reinforcement_learning.types.observation import MARLObservation

from drone_gym import task_factory


class MARLDroneEnvironment(MARLEnvironment):
    """
    CARES RL wrapper for drone_gym MARL environments.
    """

    def __init__(self, config: MARLDroneConfig, seed: int) -> None:
        super().__init__(config, seed)

        if config.use_simulator not in [0, 1]:
            raise ValueError("use_simulator must be 0 (real drone) or 1 (simulator)")

        self.env = task_factory.make(
            task_name=config.task,
            use_simulator=cast(Literal[0, 1], config.use_simulator),
            num_agents=config.num_agents,
        )

        self.possible_agents: list[str] = list(self.env.possible_agents)

        self.agent_teams = self._split_agents_by_team(self.possible_agents)

        self.observation: MARLObservation

        self.set_seed(self.seed)

    @cached_property
    def max_action_value(self) -> dict[str, np.ndarray]:
        max_action_values = {}

        for agent in self.possible_agents:
            action_space = self.env.action_space(agent)

            if not isinstance(action_space, spaces.Box):
                raise ValueError(
                    "MARLDroneEnvironment currently expects continuous Box "
                    f"action spaces, but got {type(action_space)} for {agent}."
                )

            max_action_values[agent] = action_space.high.astype(np.float32)

        return max_action_values

    @cached_property
    def min_action_value(self) -> dict[str, np.ndarray]:
        min_action_values = {}

        for agent in self.possible_agents:
            action_space = self.env.action_space(agent)

            if not isinstance(action_space, spaces.Box):
                raise ValueError(
                    "MARLDroneEnvironment currently expects continuous Box "
                    f"action spaces, but got {type(action_space)} for {agent}."
                )

            min_action_values[agent] = action_space.low.astype(np.float32)

        return min_action_values

    @cached_property
    def observation_space(self) -> dict[str, Any]:
        """
        Return observation and state dimensions for each agent and the global critic.
        """
        obs_spaces = {
            agent: int(self.env.observation_space(agent).shape[0])
            for agent in self.possible_agents
        }

        state_dim = self.env.state_space.shape[0]

        return {
            "obs": obs_spaces,
            "state": state_dim,
            "num_agents": self.num_agents,
            "teams": self.agent_teams,
        }

    @cached_property
    def action_num(self) -> int:
        first_agent = self.possible_agents[0]
        action_space = self.env.action_space(first_agent)

        if isinstance(action_space, spaces.Box):
            return int(action_space.shape[0])

        raise ValueError(f"Unhandled action space type: {type(action_space)}")

    @cached_property
    def num_agents(self) -> int:
        return len(self.possible_agents)

    def get_available_actions(self) -> dict[str, np.ndarray]:
        return {
            agent: np.ones(self.action_num, dtype=np.int32)
            for agent in self.possible_agents
        }

    # ---------------------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------------------
    def _empty_observation_for_agent(self, agent: str) -> np.ndarray:
        """
        Return a zero observation for an inactive/missing agent.
        """
        obs_dim = int(self.env.observation_space(agent).shape[0])
        return np.zeros(obs_dim, dtype=np.float32)

    def _complete_agent_states(
        self,
        agent_states: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """
        Ensure every possible agent has an observation entry.
        """
        complete_states: dict[str, np.ndarray] = {}

        for agent in self.possible_agents:
            if agent in agent_states:
                complete_states[agent] = np.asarray(
                    agent_states[agent],
                    dtype=np.float32,
                )
            else:
                complete_states[agent] = self._empty_observation_for_agent(agent)

        return complete_states

    def _complete_reward_dict(
        self,
        rewards: dict[str, float],
    ) -> dict[str, float]:
        return {agent: float(rewards.get(agent, 0.0)) for agent in self.possible_agents}

    def _complete_bool_dict(
        self,
        values: dict[str, bool],
        default: bool = True,
    ) -> dict[str, bool]:
        return {
            agent: bool(values.get(agent, default)) for agent in self.possible_agents
        }

    def _filter_action_to_active_agents(
        self,
        action: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """
        PettingZoo ParallelEnv step() should only receive actions for active agents.
        """
        active_agents = set(self.env.agents)

        return {
            agent: np.asarray(act, dtype=np.float32)
            for agent, act in action.items()
            if agent in active_agents
        }

    # ---------------------------------------------------------------------
    # Required MARLEnvironment methods
    # ---------------------------------------------------------------------

    def set_seed(self, seed: int) -> None:
        self.seed = seed
        self.env.set_seed(seed)

        for i, agent in enumerate(self.possible_agents):
            self.env.action_space(agent).seed(seed + i)
            self.env.observation_space(agent).seed(seed)

    def sample_action(self) -> dict[str, np.ndarray]:
        actions = {}

        for agent in self.possible_agents:
            action = self.env.action_space(agent).sample()
            actions[agent] = np.asarray(action, dtype=np.float32)

        return actions

    def reset(self, training: bool = True) -> MARLObservation:
        """
        Reset the drone_gym MARL task and return a CARES MARLObservation.
        """
        options = {"training": training}
        agent_states, _ = self.env.reset(options=options)

        complete_agent_states = self._complete_agent_states(agent_states)

        self.observation = MARLObservation(
            global_state=np.asarray(self.env.state(), dtype=np.float32),
            agent_states=complete_agent_states,
            available_actions=self.get_available_actions(),
        )

        return self.observation

    def step(self, action: dict[str, np.ndarray]) -> MultiAgentExperience:
        """
        Step the drone_gym MARL environment and return a CARES MultiAgentExperience.
        """

        # action = self._filter_action_to_active_agents(action)

        agent_states, rewards, dones, truncations, infos = self.env.step(action)

        complete_agent_states = self._complete_agent_states(agent_states)

        next_observation = MARLObservation(
            global_state=np.asarray(self.env.state(), dtype=np.float32),
            agent_states=complete_agent_states,
            available_actions=self.get_available_actions(),
        )

        complete_rewards = self._complete_reward_dict(rewards)
        complete_dones = self._complete_bool_dict(dones, default=True)
        complete_truncations = self._complete_bool_dict(truncations, default=True)

        experience = MultiAgentExperience(
            observation=self.observation.clone(),
            action=action,
            reward=complete_rewards,
            next_observation=next_observation.clone(),
            done=complete_dones,
            truncated=complete_truncations,
            info=infos,
        )

        self.observation = next_observation

        return experience

    def grab_frame(self, height: int = 240, width: int = 300) -> np.ndarray:
        return np.zeros((height, width, 3), dtype=np.uint8)

    def get_overlay_info(self) -> dict:
        return {}

    def close(self) -> None:
        self.env.close()
