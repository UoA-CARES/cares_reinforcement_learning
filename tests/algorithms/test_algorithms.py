import inspect
from random import randrange

import numpy as np
import pytest

from cares_reinforcement_learning.memory.memory_factory import MemoryFactory
from cares_reinforcement_learning.types.observation import Observation
from cares_reinforcement_learning.types.training import TrainingContext
from cares_reinforcement_learning.util import configurations
from cares_reinforcement_learning.util.configurations import AlgorithmConfig
from cares_reinforcement_learning.util.network_factory import NetworkFactory


def _policy_buffer(
    memory_buffer,
    capacity,
    observation_size,
    action_num,
    image_state,
    marl_state,
):
    if image_state:
        state = np.arange(observation_size["vector"], dtype=np.float32)
        state_image = np.random.randint(
            255, size=observation_size["image"], dtype=np.uint8
        )
    elif marl_state:
        state = np.arange(observation_size["state"], dtype=np.float32)

        obs = {
            agent_id: np.arange(obs_dim, dtype=np.float32)
            for agent_id, obs_dim in observation_size["obs"].items()
        }

        avail_actions = [
            np.zeros(action_num, dtype=np.float32)
            for _ in range(observation_size["num_agents"])
        ]
    else:
        state = np.arange(observation_size["vector"], dtype=np.float32)

    state = Observation(
        vector_state=state,
        image_state=state_image if image_state else None,
        agent_states=obs if marl_state else None,
        avail_actions=avail_actions if marl_state else None,
    )

    if marl_state:
        action = []
        for i in range(observation_size["num_agents"]):
            action.append(list(range(action_num)))
        action = np.array(action)
    else:
        action = list(range(action_num))

    if marl_state:
        reward = [10] * observation_size["num_agents"]
    else:
        reward = 10

    if marl_state:
        done = [False] * observation_size["num_agents"]
    else:
        done = False

    for _ in range(capacity):
        memory_buffer.add(state, action, reward, state, done)

    return memory_buffer


def _value_buffer(
    memory_buffer, capacity, observation_size, action_num, image_state, marl_state
):

    if image_state:
        state = np.arange(observation_size["vector"], dtype=np.float32)
        state_image = np.random.randint(
            255, size=observation_size["image"], dtype=np.uint8
        )
    elif marl_state:
        state = np.arange(observation_size["state"], dtype=np.float32)

        obs = {}
        avail_actions = []
        for i, agent_id in enumerate(observation_size["obs"].keys()):
            obs[agent_id] = np.arange(
                observation_size["obs"][agent_id], dtype=np.float32
            )
            avail_actions.append(np.ones(action_num, dtype=np.float32))
    else:
        state = np.arange(observation_size["vector"], dtype=np.float32)

    state = Observation(
        vector_state=state,
        image_state=state_image if image_state else None,
        agent_states=obs if marl_state else None,
        avail_actions=avail_actions if marl_state else None,
    )

    if marl_state:
        action = []
        for i in range(observation_size["num_agents"]):
            action.append(randrange(action_num))
    else:
        action = randrange(action_num)

    if marl_state:
        reward = [10] * observation_size["num_agents"]
    else:
        reward = 10

    if marl_state:
        done = [False] * observation_size["num_agents"]
    else:
        done = False

    for _ in range(capacity):
        memory_buffer.add(state, action, reward, state, done)

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

        agent.save_models(tmp_path, f"{algorithm}")
        agent.load_models(tmp_path, f"{algorithm}")

        if agent.policy_type == "policy":
            memory_buffer = _policy_buffer(
                memory_buffer,
                capacity,
                observation_size,
                action_num,
                alg_config.image_observation,
                alg_config.marl_observation,
            )
        elif agent.policy_type == "value" or agent.policy_type == "discrete_policy":
            memory_buffer = _value_buffer(
                memory_buffer,
                capacity,
                observation_size,
                action_num,
                alg_config.image_observation,
                alg_config.marl_observation,
            )
        else:
            continue

        experiences = memory_buffer.sample_uniform(1)
        states, actions, rewards, next_states, dones, _ = experiences

        value = agent._calculate_value(states[0], actions[0])
        assert isinstance(
            value, float
        ), f"{algorithm} did not return a float value for the calculated value"

        training_context = TrainingContext(
            memory=memory_buffer,
            batch_size=batch_size,
            training_step=1,
            episode=1,
            episode_steps=1,
            episode_reward=10.0,
            episode_done=True,
        )

        info = agent.train_policy(training_context)
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
