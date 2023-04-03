"""
Description:
            This is a basic example of the training loop for Off Policy Algorithms,
            We may move this later for each repo/env or keep this in this repo
"""

import argparse

from cares_reinforcement_learning.util import NetworkFactory
from cares_reinforcement_learning.util import MemoryBuffer

import gym
from gym import spaces

import torch
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

def set_seed(env, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.action_space.seed(seed)

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

def evaluate_value_network(env, agent, args):
    evaluation_seed = args["evaluation_seed"]
    max_steps_evaluation = args["max_steps_evaluation"]

    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0

    env = gym.make(env.spec.id, render_mode="human")
    state, _ = env.reset(seed=evaluation_seed)

    historical_reward = {"step": [], "episode_reward": []}
    # exploration_rate  = 1

    for total_step_counter in range(int(max_steps_evaluation)):
        episode_timesteps += 1

        action = agent.select_action_from_policy(state)

        next_state, reward, done, truncated, _ = env.step(action)
        episode_reward += reward

        if done or truncated:
            logging.info(f"Total T:{total_step_counter+1} Episode {episode_num+1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}.")

            historical_reward["step"].append(total_step_counter)
            historical_reward["episode_reward"].append(episode_reward)

            # Reset environment
            state, _ = env.reset()
            episode_reward    = 0
            episode_timesteps = 0
            episode_num += 1

def value_based_train(env, agent, args):
    max_steps_training = args["max_steps_training"]
    exploration_min    = args["exploration_min"]
    exploration_decay  = args["exploration_decay"]
    
    batch_size = args["batch_size"]
    seed = args["seed"]
    G = args["G"]

    memory = MemoryBuffer()

    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0

    state, _ = env.reset(seed=seed)

    historical_reward = {"step": [], "episode_reward": []}
    exploration_rate  = 1

    for total_step_counter in range(int(max_steps_training)):
        episode_timesteps += 1

        exploration_rate *= exploration_decay
        exploration_rate = max(exploration_min, exploration_rate)

        if random.random() < exploration_rate:
            action = env.action_space.sample()
        else:
            action = agent.select_action_from_policy(state)

        next_state, reward, done, truncated, _ = env.step(action)
        memory.add(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        if len(memory.buffer) > batch_size:
            for _ in range(G):
                experiences = memory.sample(batch_size)
                agent.train_policy(experiences)

        if done or truncated:
            logging.info(f"Total T:{total_step_counter+1} Episode {episode_num+1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}. Exploration Rate: {exploration_rate}")

            historical_reward["step"].append(total_step_counter)
            historical_reward["episode_reward"].append(episode_reward)

            # Reset environment
            state, _ = env.reset()
            episode_reward    = 0
            episode_timesteps = 0
            episode_num += 1

    plot_reward_curve(historical_reward)

def evaluate_policy_network(env, agent, args):
    evaluation_seed = args["evaluation_seed"]
    max_steps_evaluation = args["max_steps_evaluation"]

    min_action_value = env.action_space.low[0]
    max_action_value = env.action_space.high[0]

    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0

    env = gym.make(env.spec.id, render_mode="human")
    state, _ = env.reset(seed=evaluation_seed)

    for total_step_counter in range(int(max_steps_evaluation)):
        episode_timesteps += 1
        action = agent.select_action_from_policy(state, evaluation=True)
        action_env = denormalize(action, max_action_value, min_action_value)

        state, reward, done, truncated, _ = env.step(action_env)
        episode_reward += reward

        if done or truncated:
            logging.info(f" Evaluation Episode {episode_num+1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}")
            # Reset environment
            state, _ = env.reset()
            episode_reward    = 0
            episode_timesteps = 0
            episode_num += 1

def policy_based_train(env, agent, args):
    max_steps_training = args["max_steps_training"]
    max_steps_exploration = args["max_steps_exploration"]
    batch_size = args["batch_size"]
    seed = args["seed"]
    G = args["G"]

    min_action_value = env.action_space.low[0]
    max_action_value = env.action_space.high[0]
    
    memory = MemoryBuffer()

    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0

    state, _ = env.reset(seed=seed)

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

        next_state, reward, done, truncated, info = env.step(action_env)
        memory.add(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if total_step_counter >= max_steps_exploration:
            for _ in range(G):
                experiences = memory.sample(batch_size)
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

def parse_args():
    parser = argparse.ArgumentParser()# Add an argument

    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--network', type=str, required=True)

    parser.add_argument('--G', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=32)
    
    parser.add_argument('--max_steps_exploration', type=int, default=10000)
    parser.add_argument('--max_steps_training', type=int, default=50000)
    parser.add_argument('--max_steps_evaluation', type=int, default=5000)

    parser.add_argument('--seed', type=int, default=571)
    parser.add_argument('--evaluation_seed', type=int, default=152)

    parser.add_argument('--actor_lr', type=float, default=1e-4)
    parser.add_argument('--critic_lr', type=float, default=1e-3)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--exploration_min', type=float, default=1e-3)
    parser.add_argument('--exploration_decay', type=float, default=0.9999)
    
    return vars(parser.parse_args())# converts into a dictionary

def main():
    args = parse_args()
    
    logging.info(f"Training on {args['task']}")
    env = gym.make(args["task"])
    
    args["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {args['device']}")

    args["observation_size"] = env.observation_space.shape[0]
    if type(env.action_space) == spaces.Box:
        args["action_num"] = env.action_space.shape[0]
    elif type(env.action_space) == spaces.Discrete:
        args["action_num"] = env.action_space.n
    else:
        raise ValueError(f"Unhandled action space type: {type(env.action_space)}")
    
    set_seed(env, args["seed"])

    # Create the network we are using
    factory = NetworkFactory()
    agent   = factory.create_network(args["network"], args)    
    
    # Train the policy or value based approach
    if agent.type == "policy":
        policy_based_train(env, agent, args)
        evaluate_policy_network(env, agent, args)
    elif agent.type == "value":
        value_based_train(env, agent, args)
        evaluate_value_network(env, agent, args)
    else:
        raise ValueError(f"Agent type is unkown: {agent.type}")

if __name__ == '__main__':
    main()
