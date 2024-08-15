import inspect
import os

import numpy as np
import pytest

from cares_reinforcement_learning.memory.memory_factory import MemoryFactory
from cares_reinforcement_learning.util import NetworkFactory, configurations
from cares_reinforcement_learning.util.configurations import AlgorithmConfig

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


def policy_buffer(memory_buffer, capacity, observation_size, action_num, image_state):
    state = (
        np.random.randint(255, size=observation_size, dtype=np.uint8)
        if image_state
        else list(range(observation_size))
    )
    action = list(range(action_num))
    reward = [10]
    next_state = (
        np.random.randint(255, size=observation_size, dtype=np.uint8)
        if image_state
        else list(range(observation_size))
    )
    done = False

    for _ in range(capacity):
        memory_buffer.add(state, action, reward, next_state, done)

    return memory_buffer


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
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
    im_size = 84

    observation_size_vector = 5

    observation_size_image = (9, 32, 32)

    action_num = 2

    # state = [1, 2, 3, 4, 5]
    # image_state = torch.randn(3, im_size, im_size)
    # action = [0]
    # reward = [10]
    # next_state = [6, 7, 8, 9, 10]
    # done = False
    # log_probs = [0.5]

    # memory_buffer_vector = MemoryBuffer(max_capacity=capacity)
    # for _ in range(capacity):
    #     memory_buffer_vector.add(state, action, reward, next_state, done)

    # memory_buffer_image = MemoryBuffer(max_capacity=capacity)
    # for _ in range(capacity):
    #     memory_buffer_image.add(image_state, action, reward, image_state, done)

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

        if alg_config.algorithm == "PPO":
            continue
        elif agent.type == "policy":
            memory_buffer = policy_buffer(
                memory_buffer,
                capacity,
                observation_size,
                action_num,
                alg_config.image_observation,
            )
        elif agent.type == "discrete_policy":
            continue
        elif agent.type == "value":
            continue
        elif agent.type == "mbrl":
            continue

        agent.train_policy(memory_buffer, batch_size)
