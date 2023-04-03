"""
Description:
            This is a basic example of the training loop for ON Policy Algorithms, PPO

"""
from cares_reinforcement_learning.algorithm import PPO
from cares_reinforcement_learning.networks.PPO import Actor
from cares_reinforcement_learning.networks.PPO import Critic

import gym
import torch
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env    = gym.make('BipedalWalker-v3')  # Pendulum-v1, BipedalWalker-v3

SEED       = 571
GAMMA      = 0.99
ACTOR_LR   = 1e-4  # 3e-4
CRITIC_LR  = 1e-3

max_steps_training  = 1_000_000
max_steps_per_batch = 5000


def set_seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    env.action_space.seed(SEED)

def plot_reward_curve(data_reward):
    data = pd.DataFrame.from_dict(data_reward)
    data.plot(x='step', y='episode_reward', title="Reward Curve")
    plt.show()


def normalize(action, max_action_value, min_action_value):
    # return action in env range [max_action_value, min_action_value]
    max_range_value = max_action_value
    min_range_value = min_action_value
    max_value_in    = 1
    min_value_in    = -1
    action = (action - min_value_in) * (max_range_value - min_range_value) / (max_value_in - min_value_in) + min_range_value
    return action

# "============================================================================================"
# todo move this class to a better place
class RolloutBuffer:
    def __init__(self):
        self.states       = []
        self.actions      = []
        self.log_probs    = []
        self.next_states  = []
        self.rewards      = []
        self.dones        = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.next_states[:]
        del self.rewards[:]
        del self.dones[:]
# "==========================================================================================="

def train(agent, memory, max_action_value, min_action_value):
    episode_timesteps = 0
    episode_num       = 0
    episode_reward    = 0
    time_step         = 0

    state, _ = env.reset(seed=SEED)
    historical_reward = {"step": [], "episode_reward": []}

    for total_step_counter in range(int(max_steps_training)):
        episode_timesteps += 1

        action, log_prob = agent.select_action_from_policy(state)
        action_env = normalize(action, max_action_value, min_action_value)  # mapping the env range [e.g. -2 , 2 for pendulum]

        next_state, reward, done, truncated, _ = env.step(action_env)

        # ------------------------------------------------------------------------------------------------
        # save rollouts in memory, TODO this could be moved in a better place in a better way
        memory.states.append(state)
        memory.next_states.append(next_state)
        memory.actions.append(action)
        memory.log_probs.append(log_prob)
        memory.rewards.append(reward)
        memory.dones.append(done)
        # ------------------------------------------------------------------------------------------------

        state = next_state
        episode_reward += reward

        time_step += 1  # I need this otherwise the next if is true at the first interaction
        if time_step % max_steps_per_batch == 0:
            agent.train_policy(memory)

        if done or truncated:
            logging.info(f"Total T:{total_step_counter + 1} Episode {episode_num + 1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}")

            historical_reward["step"].append(total_step_counter+1)
            historical_reward["episode_reward"].append(episode_reward)

            # Reset environment
            state, _ = env.reset()
            episode_reward    = 0
            episode_timesteps = 0
            episode_num       += 1

    plot_reward_curve(historical_reward)

def main():
    observation_size = env.observation_space.shape[0]
    action_num       = env.action_space.shape[0]

    max_action_value = env.action_space.high[0]
    min_action_value = env.action_space.low[0]

    memory = RolloutBuffer()
    actor  = Actor(observation_size, action_num, ACTOR_LR)
    critic = Critic(observation_size, CRITIC_LR)

    agent = PPO(
        actor_network=actor,
        critic_network=critic,
        gamma=GAMMA,
        action_num=action_num,
        device=DEVICE,
    )

    set_seed()
    train(agent, memory, max_action_value, min_action_value)


if __name__ == '__main__':
    main()
