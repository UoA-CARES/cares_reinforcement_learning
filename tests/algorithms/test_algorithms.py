import inspect
from random import randrange

import numpy as np
import pytest

from cares_reinforcement_learning.memory.memory_factory import MemoryFactory
from cares_reinforcement_learning.types.experience import (
    SingleAgentExperience,
    MultiAgentExperience,
)
from cares_reinforcement_learning.types.observation import (
    MARLObservation,
    SARLObservation,
)
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.util import configurations
from cares_reinforcement_learning.util.configurations import AlgorithmConfig
from cares_reinforcement_learning.util.network_factory import NetworkFactory


def create_sarl_observation(
    observation_size: dict, image_state: bool = False
) -> SARLObservation:
    """Create a SARL observation for testing."""
    vector_state = np.arange(observation_size["vector"], dtype=np.float32)

    if image_state:
        image = np.random.randint(255, size=observation_size["image"], dtype=np.uint8)
    else:
        image = None

    return SARLObservation(vector_state=vector_state, image_state=image)


def create_marl_observation(observation_size: dict, action_num: int) -> MARLObservation:
    """Create a MARL observation for testing."""
    global_state = np.arange(observation_size["state"], dtype=np.float32)

    agent_states = {
        agent_id: np.arange(obs_dim, dtype=np.float32)
        for agent_id, obs_dim in observation_size["obs"].items()
    }

    avail_actions = np.asarray(
        [
            np.ones(action_num, dtype=np.float32)
            for _ in range(observation_size["num_agents"])
        ]
    )

    return MARLObservation(
        global_state=global_state,
        agent_states=agent_states,
        avail_actions=avail_actions,
    )


def populate_buffer_sarl(
    memory_buffer,
    capacity: int,
    observation_size: dict,
    action_num: int,
    image_state: bool = False,
    discrete: bool = False,
):
    """Populate a SARL buffer with test experiences."""
    for _ in range(capacity):
        observation = create_sarl_observation(observation_size, image_state)
        next_observation = create_sarl_observation(observation_size, image_state)

        if discrete:
            action = np.array(randrange(action_num))
        else:
            action = np.array(list(range(action_num)), dtype=np.float32)

        experience = SingleAgentExperience(
            observation=observation,
            next_observation=next_observation,
            action=action,
            reward=10.0,
            done=False,
            truncated=False,
            info={},
        )
        memory_buffer.add(experience)

    return memory_buffer


def populate_buffer_marl(
    memory_buffer,
    capacity: int,
    observation_size: dict,
    action_num: int,
    discrete: bool = False,
):
    """Populate a MARL buffer with test experiences."""
    num_agents = observation_size["num_agents"]

    for _ in range(capacity):
        observation = create_marl_observation(observation_size, action_num)
        next_observation = create_marl_observation(observation_size, action_num)

        if discrete:
            actions = [np.array(randrange(action_num)) for _ in range(num_agents)]
        else:
            actions = [
                np.array(list(range(action_num)), dtype=np.float32)
                for _ in range(num_agents)
            ]

        experience = MultiAgentExperience(
            observation=observation,
            next_observation=next_observation,
            action=actions,
            reward=[10.0] * num_agents,
            done=[False] * num_agents,
            truncated=[False] * num_agents,
            info={},
        )
        memory_buffer.add(experience)

    return memory_buffer


def test_algorithms(tmp_path):
    factory = NetworkFactory()
    memory_factory = MemoryFactory()

    algorithm_configurations = {}
    for name, cls in inspect.getmembers(configurations, inspect.isclass):
        if issubclass(cls, AlgorithmConfig) and cls != AlgorithmConfig:
            name = name.replace("Config", "")
            algorithm_configurations[name] = cls

    capacity = 5

    observation_size_vector = 5

    observation_size_image = (9, 32, 32)

    observation_size_marl = {
        "obs": {"agent_0": 10, "agent_1": 10, "agent_2": 10},
        "state": 30,
        "num_agents": 3,
    }

    action_num = 2

    for algorithm, alg_config in algorithm_configurations.items():
        print(f"Testing training step for {algorithm}")
        alg_config = alg_config()

        memory_buffer = memory_factory.create_memory(alg_config)

        if alg_config.marl_observation:
            observation_size = observation_size_marl
        else:
            observation_size = {
                "image": observation_size_image,
                "vector": observation_size_vector,
            }

        agent = factory.create_network(
            observation_size=observation_size, action_num=action_num, config=alg_config
        )
        assert agent is not None, f"{algorithm} was not created successfully"

        if agent.policy_type == "mbrl":
            continue

        agent.save_models(tmp_path, f"{algorithm}")
        agent.load_models(tmp_path, f"{algorithm}")

        # Populate buffer based on algorithm type
        is_discrete = agent.policy_type in ("value", "discrete_policy")

        if alg_config.marl_observation:
            memory_buffer = populate_buffer_marl(
                memory_buffer,
                capacity,
                observation_size,
                action_num,
                discrete=is_discrete,
            )
        else:
            memory_buffer = populate_buffer_sarl(
                memory_buffer,
                capacity,
                observation_size,
                action_num,
                image_state=alg_config.image_observation,
                discrete=is_discrete,
            )

        sample = memory_buffer.sample_uniform(1)

        value = agent._calculate_value(
            sample.experiences[0].observation, sample.experiences[0].action
        )
        assert isinstance(
            value, float
        ), f"{algorithm} did not return a float value for the calculated value"

        training_context = EpisodeContext(
            training_step=1,
            episode=1,
            episode_steps=1,
            episode_reward=10.0,
            episode_done=True,
        )

        info = agent.train_policy(memory_buffer, training_context)
        assert isinstance(
            info, dict
        ), f"{algorithm} did not return a dictionary of training info"

        intrinsic_on = (
            bool(alg_config.intrinsic_on)
            if hasattr(alg_config, "intrinsic_on")
            else False
        )

        if intrinsic_on:
            sample = memory_buffer.sample_uniform(1)

            intrinsic_reward = agent.get_intrinsic_reward(
                sample.experiences[0].observation,
                sample.experiences[0].action,
                sample.experiences[0].next_observation,
            )
