import inspect

import pytest

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.util import NetworkFactory, configurations
from cares_reinforcement_learning.util.configurations import AlgorithmConfig


def test_network_factory():
    factory = NetworkFactory()

    algorithm_configurations = {}
    for name, cls in inspect.getmembers(configurations, inspect.isclass):
        if issubclass(cls, AlgorithmConfig) and cls != AlgorithmConfig:
            name = name.replace("Config", "")
            algorithm_configurations[name] = cls

    for algorithm, config in algorithm_configurations.items():
        config = config()
        observation_size = (9, 84, 84) if config.image_observation else 10
        action_num = 2
        network = factory.create_network(
            observation_size=observation_size, action_num=action_num, config=config
        )
        assert network is not None, f"{algorithm} was not created successfully"


def test_denormalize():
    action = 0.5
    max_action_value = 5
    min_action_value = -5
    result = hlp.denormalize(action, max_action_value, min_action_value)
    assert result == 2.5, "Result does not match expected denormalized value"


def test_normalize():
    action = 2.5
    max_action_value = 5
    min_action_value = -5
    result = hlp.normalize(action, max_action_value, min_action_value)
    assert result == 0.5, "Result does not match expected normalized value"
