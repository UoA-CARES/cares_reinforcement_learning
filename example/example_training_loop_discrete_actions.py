"""
Description:
            This is a basic example of the training loop for Off Policy Algorithms,
            We may move this later for each repo/env or keep this in this repo
"""

from cares_reinforcement_learning.algorithm import DQN
from cares_reinforcement_learning.networks.DQN import Network
from cares_reinforcement_learning.util import MemoryBuffer

import gym
import torch
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env    = gym.make('CartPole-v1')  # Pendulum-v1, BipedalWalker-v3


G = 1
GAMMA = 0.99
LR   = 1e-4
BATCH_SIZE = 32

EXPLORATION_MIN   = 0.001
EXPLORATION_DECAY = 0.9999

max_steps_training    = 100_000
SEED                  = 571


def set_seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    env.action_space.seed(SEED)


def plot_reward_curve(data_reward):
    data = pd.DataFrame.from_dict(data_reward)
    data.plot(x='step', y='episode_reward', title="Reward Curve")
    plt.show()


def train(agent, memory, max_action_value, min_action_value):
    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0

    state, _ = env.reset(seed=SEED)

    historical_reward = {"step": [], "episode_reward": []}
    exploration_rate  = 1

    for total_step_counter in range(int(max_steps_training)):
        episode_timesteps += 1

        exploration_rate *= EXPLORATION_DECAY
        exploration_rate = max(EXPLORATION_MIN, exploration_rate)

        if random.random() < exploration_rate:
            action = env.action_space.sample()
        else:
            action = agent.select_action_from_policy(state)

        next_state, reward, done, truncated, _ = env.step(action)
        memory.add(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        if len(memory.buffer) > BATCH_SIZE:
            for _ in range(G):
                experiences = memory.sample(BATCH_SIZE)
                agent.train_policy(experiences)

        if done or truncated:
            logging.info(f"Total T:{total_step_counter+1} Episode {episode_num+1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}")

            historical_reward["step"].append(total_step_counter)
            historical_reward["episode_reward"].append(episode_reward)

            # Reset environment
            state, _ = env.reset()
            episode_reward    = 0
            episode_timesteps = 0
            episode_num += 1

    plot_reward_curve(historical_reward)


def main():
    observation_size = env.observation_space.shape[0]
    action_num       = env.action_space.n

    # max_actions = env.action_space.high[0]
    # min_actions = env.action_space.low[0]

    max_actions = 2
    min_actions = -2

    memory  = MemoryBuffer()
    network = Network(observation_size, action_num, LR)

    agent = DQN(
        network=network,
        gamma=GAMMA,
        device=DEVICE
    )

    set_seed()
    train(agent, memory, max_actions, min_actions)


if __name__ == '__main__':
    main()
