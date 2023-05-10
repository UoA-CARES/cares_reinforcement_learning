
import logging
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import copy
import numpy as np
import gym
import torch
import pandas as pd
import matplotlib.pyplot as plt

from cares_reinforcement_learning.networks.TD3 import Actor
from cares_reinforcement_learning.networks.TD3 import Critic
from cares_reinforcement_learning.util import PrioritizedReplayBuffer
from cares_reinforcement_learning.algorithm import PER_TD3

logging.basicConfig(level=logging.INFO)

env    = gym.make('HalfCheetah-v4')
G          = 10
GAMMA      = 0.99
TAU        = 0.005
ACTOR_LR   = 1e-4  #3e-4  1e-4
CRITIC_LR  = 7e-4  #3e-4  1e-3
BATCH_SIZE = 32 #256 , 32, 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

max_steps_exploration = 1_000
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


def denormalize(action, max_action_value, min_action_value):
    # return action in env range [max_action_value, min_action_value]
    max_range_value = max_action_value
    min_range_value = min_action_value
    max_value_in    = 1
    min_value_in    = -1
    action_denorm = (action - min_value_in) * (max_range_value - min_range_value) / (max_value_in - min_value_in) + min_range_value
    return action_denorm


def normalize(action, max_action_value, min_action_value):
    # return action in algorithm range [-1, +1]
    max_range_value = 1
    min_range_value = -1
    max_value_in = max_action_value
    min_value_in = min_action_value
    action_norm = (action - min_value_in) * (max_range_value - min_range_value) / (max_value_in - min_value_in) + min_range_value
    return action_norm


def train(agent, memory, max_action_value, min_action_value):
    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0

    state,_ = env.reset(seed=SEED)

    historical_reward = {"step": [], "episode_reward": []}

    for total_step_counter in range(int(max_steps_training)):
        episode_timesteps += 1

        if total_step_counter < max_steps_exploration:
            logging.info(f"Running Exploration Steps {total_step_counter}/{max_steps_exploration}")
            action_env = env.action_space.sample() # action range the env uses [e.g. -2 , 2 for pendulum]
            action = normalize(action_env, max_action_value, min_action_value)  # algorithm range [-1, 1]

        else:
            action = agent.select_action_from_policy(state) # algorithm range [-1, 1]
            action_env = denormalize(action, max_action_value, min_action_value)  # mapping to env range [e.g. -2 , 2 for pendulum]

        next_state, reward, done, truncated, _ = env.step(action_env)
        memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

        state = next_state
        episode_reward += reward

        if total_step_counter >= max_steps_exploration:
            for _ in range(G):
                agent.train_policy(memory, BATCH_SIZE)

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
    action_num       = env.action_space.shape[0]

    max_action_value = env.action_space.high[0]
    min_action_value = env.action_space.low[0]

    memory = PrioritizedReplayBuffer(observation_size, action_num)
    actor  = Actor(observation_size, action_num, ACTOR_LR)
    critic = Critic(observation_size, action_num, CRITIC_LR)

    agent = PER_TD3(
        actor_network=actor,
        critic_network=critic,
        gamma=GAMMA,
        tau=TAU,
        action_num=action_num,
        device=DEVICE,
    )


    set_seed()
    train(agent, memory, max_action_value, min_action_value)


if __name__ == '__main__':
    main()