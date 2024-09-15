import inspect
from random import randrange

import numpy as np
import pytest

from cares_reinforcement_learning.memory.memory_factory import MemoryFactory
from cares_reinforcement_learning.util import NetworkFactory, configurations
from cares_reinforcement_learning.util.configurations import AlgorithmConfig


def _policy_buffer(
    memory_buffer,
    capacity,
    observation_size,
    action_num,
    image_state,
    add_log_prob=False,
):
    state = (
        np.random.randint(255, size=observation_size, dtype=np.uint8)
        if image_state
        else list(range(observation_size))
    )
    action = list(range(action_num))
    reward = 10
    next_state = (
        np.random.randint(255, size=observation_size, dtype=np.uint8)
        if image_state
        else list(range(observation_size))
    )
    done = False

    for _ in range(capacity):
        if add_log_prob:
            memory_buffer.add(state, action, reward, next_state, done, 0.0)
        else:
            memory_buffer.add(state, action, reward, next_state, done)

    return memory_buffer


def _value_buffer(memory_buffer, capacity, observation_size, action_num, image_state):
    state = (
        np.random.randint(255, size=observation_size, dtype=np.uint8)
        if image_state
        else list(range(observation_size))
    )
    action = randrange(action_num)
    reward = 10
    next_state = (
        np.random.randint(255, size=observation_size, dtype=np.uint8)
        if image_state
        else list(range(observation_size))
    )
    done = False

    for _ in range(capacity):
        memory_buffer.add(state, action, reward, next_state, done)

    return memory_buffer


# @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_algorithms():
    factory = NetworkFactory()
    memory_factory = MemoryFactory()

    algorithm_configurations = {}
    for name, cls in inspect.getmembers(configurations, inspect.isclass):
        if issubclass(cls, AlgorithmConfig) and cls != AlgorithmConfig:
            name = name.replace("Config", "")
            algorithm_configurations[name] = cls

    capacity = 5
    batch_size = 2

    observation_size_vector = 5

    observation_size_image = (9, 32, 32)

    action_num = 2

    for algorithm, alg_config in algorithm_configurations.items():
        alg_config = alg_config()

        memory_buffer = memory_factory.create_memory(alg_config)

        observation_size = (
            observation_size_image
            if alg_config.image_observation
            else observation_size_vector
        )

        agent = factory.create_network(
            observation_size=observation_size, action_num=action_num, config=alg_config
        )
        assert agent is not None, f"{algorithm} was not created successfully"

        if agent.type == "policy":
            memory_buffer = _policy_buffer(
                memory_buffer,
                capacity,
                observation_size,
                action_num,
                alg_config.image_observation,
                add_log_prob=(alg_config.algorithm == "PPO"),
            )
        elif agent.type == "value" or agent.type == "discrete_policy":
            memory_buffer = _value_buffer(
                memory_buffer,
                capacity,
                observation_size,
                action_num,
                alg_config.image_observation,
            )
        else:
            continue

        info = agent.train_policy(memory_buffer, batch_size)
        assert isinstance(
            info, dict
        ), f"{algorithm} did not return a dictionary of training info"
