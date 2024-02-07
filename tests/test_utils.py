import pytest
from cares_reinforcement_learning.algorithm.policy import *
from cares_reinforcement_learning.algorithm.value import *
from cares_reinforcement_learning.util.network_factory import *
from cares_reinforcement_learning.util.helpers import *
from cares_reinforcement_learning.util.configurations import *


def test_create_agents():
    agent = create_DQN(10, 5, DQNConfig())
    assert isinstance(agent, DQN), "Failed to create DQN agent"

    agent = create_DuelingDQN(10, 5, DuelingDQNConfig())
    assert isinstance(agent, DQN), "Failed to create DuelingDQN agent"

    agent = create_DoubleDQN(10, 5, DoubleDQNConfig())
    assert isinstance(agent, DoubleDQN), "Failed to create DDQN agent"

    agent = create_PPO(10, 5, PPOConfig())
    assert isinstance(agent, PPO), "Failed to create PPO agent"

    agent = create_SAC(10, 5, SACConfig())
    assert isinstance(agent, SAC), "Failed to create SAC agent"

    agent = create_DDPG(10, 5, DDPGConfig())
    assert isinstance(agent, DDPG), "Failed to create DDPG agent"

    agent = create_TD3(10, 5, TD3Config())
    assert isinstance(agent, TD3), "Failed to create TD3 agent"


def test_create_network():
    factory = NetworkFactory()

    agent = factory.create_network(10, 5, DQNConfig())
    assert isinstance(agent, DQN), "Failed to create DQN agent"

    agent = factory.create_network(10, 5, DoubleDQNConfig())
    assert isinstance(agent, DoubleDQN), "Failed to create DDQN agent"

    agent = factory.create_network(10, 5, DuelingDQNConfig())
    assert isinstance(agent, DQN), "Failed to create DuelingDQN agent"

    agent = factory.create_network(10, 5, PPOConfig())
    assert isinstance(agent, PPO), "Failed to create PPO agent"

    agent = factory.create_network(10, 5, SACConfig())
    assert isinstance(agent, SAC), "Failed to create SAC agent"

    agent = factory.create_network(10, 5, DDPGConfig())
    assert isinstance(agent, DDPG), "Failed to create DDPG agent"

    agent = factory.create_network(10, 5, TD3Config())
    assert isinstance(agent, TD3), "Failed to create TD3 agent"

    agent = factory.create_network(10, 5, AlgorithmConfig(algorithm="unknown"))
    assert agent is None, f"Unkown failed to return None: returned {agent}"


def test_denormalize():
    action = 0.5
    max_action_value = 5
    min_action_value = -5
    result = denormalize(action, max_action_value, min_action_value)
    assert result == 2.5, "Result does not match expected denormalized value"


def test_normalize():
    action = 2.5
    max_action_value = 5
    min_action_value = -5
    result = normalize(action, max_action_value, min_action_value)
    assert result == 0.5, "Result does not match expected normalized value"
