import pytest
from cares_reinforcement_learning.algorithm.policy import *
from cares_reinforcement_learning.algorithm.value import *
from cares_reinforcement_learning.util.NetworkFactory import *


def test_create_agents():
    args = {
        "observation_size": 10,
        "action_num": 5,
        "lr": 0.001,
        "actor_lr": 0.001,
        "critic_lr": 0.001,
        "gamma": 0.99,
        "tau": 0.01,
        "device": "cpu"
    }

    agent = create_DQN(args)
    assert isinstance(agent, DQN), "Failed to create DQN agent"

    agent = create_DuelingDQN(args)
    assert isinstance(agent, DQN), "Failed to create DuelingDQN agent"

    agent = create_DDQN(args)
    assert isinstance(agent, DoubleDQN), "Failed to create DDQN agent"

    agent = create_PPO(args)
    assert isinstance(agent, PPO), "Failed to create PPO agent"

    agent = create_SAC(args)
    assert isinstance(agent, SAC), "Failed to create SAC agent"

    agent = create_DDPG(args)
    assert isinstance(agent, DDPG), "Failed to create DDPG agent"

    agent = create_TD3(args)
    assert isinstance(agent, TD3), "Failed to create TD3 agent"


def test_create_network():
    factory = NetworkFactory()
    args = {
        "observation_size": 10,
        "action_num": 5,
        "lr": 0.001,
        "actor_lr": 0.001,
        "critic_lr": 0.001,
        "gamma": 0.99,
        "tau": 0.01,
        "device": "cpu"
    }

    agent = factory.create_network("DQN", args)
    assert isinstance(agent, DQN), "Failed to create DQN agent"

    agent = factory.create_network("DDQN", args)
    assert isinstance(agent, DoubleDQN), "Failed to create DDQN agent"

    agent = factory.create_network("DuelingDQN", args)
    assert isinstance(agent, DQN), "Failed to create DuelingDQN agent"

    agent = factory.create_network("PPO", args)
    assert isinstance(agent, PPO), "Failed to create PPO agent"

    agent = factory.create_network("SAC", args)
    assert isinstance(agent, SAC), "Failed to create SAC agent"

    agent = factory.create_network("DDPG", args)
    assert isinstance(agent, DDPG), "Failed to create DDPG agent"

    agent = factory.create_network("TD3", args)
    assert isinstance(agent, TD3), "Failed to create TD3 agent"

    with pytest.raises(ValueError):
        factory.create_network("Unknown", args)
