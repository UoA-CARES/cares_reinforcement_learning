import importlib.util
import inspect
import sys
from pathlib import Path
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
    if image_state:
        state_vector = list(range(observation_size["vector"]))
        state_image = np.random.randint(
            255, size=observation_size["image"], dtype=np.uint8
        )
        state = {"image": state_image, "vector": state_vector}
    else:
        state = list(range(observation_size))

    action = list(range(action_num))
    reward = 10

    if image_state:
        next_state_vector = list(range(observation_size["vector"]))
        next_state_image = np.random.randint(
            255, size=observation_size["image"], dtype=np.uint8
        )
        next_state = {"image": next_state_image, "vector": next_state_vector}
    else:
        next_state = list(range(observation_size))

    done = False

    for _ in range(capacity):
        if add_log_prob:
            memory_buffer.add(state, action, reward, next_state, done, 0.0)
        else:
            memory_buffer.add(state, action, reward, next_state, done)

    return memory_buffer


def _value_buffer(memory_buffer, capacity, observation_size, action_num, image_state):

    if image_state:
        state_vector = list(range(observation_size["vector"]))
        state_image = np.random.randint(
            255, size=observation_size["image"], dtype=np.uint8
        )
        state = {"image": state_image, "vector": state_vector}
    else:
        state = list(range(observation_size))

    action = randrange(action_num)
    reward = 10

    if image_state:
        next_state_vector = list(range(observation_size["vector"]))
        next_state_image = np.random.randint(
            255, size=observation_size["image"], dtype=np.uint8
        )
        next_state = {"image": next_state_image, "vector": next_state_vector}
    else:
        next_state = list(range(observation_size))

    done = False

    for _ in range(capacity):
        memory_buffer.add(state, action, reward, next_state, done)

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
    batch_size = 2

    observation_size_vector = 5

    observation_size_image = (9, 32, 32)

    action_num = 2

    for algorithm, alg_config in algorithm_configurations.items():

        alg_config = alg_config()

        memory_buffer = memory_factory.create_memory(alg_config)

        observation_size = (
            {"image": observation_size_image, "vector": observation_size_vector}
            if alg_config.image_observation
            else observation_size_vector
        )

        agent = factory.create_network(
            observation_size=observation_size, action_num=action_num, config=alg_config
        )
        assert agent is not None, f"{algorithm} was not created successfully"

        agent.save_models(tmp_path, f"{algorithm}")
        agent.load_models(tmp_path, f"{algorithm}")

        if agent.policy_type == "policy":
            memory_buffer = _policy_buffer(
                memory_buffer,
                capacity,
                observation_size,
                action_num,
                alg_config.image_observation,
                add_log_prob=(alg_config.algorithm == "PPO"),
            )
        elif agent.policy_type == "value" or agent.type == "discrete_policy":
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

        intrinsic_on = (
            bool(alg_config.intrinsic_on)
            if hasattr(alg_config, "intrinsic_on")
            else False
        )

        if intrinsic_on:
            experiences = memory_buffer.sample_uniform(1)
            states, actions, _, next_states, _, _ = experiences

            intrinsic_reward = agent.get_intrinsic_reward(
                states[0], actions[0], next_states[0]
            )
