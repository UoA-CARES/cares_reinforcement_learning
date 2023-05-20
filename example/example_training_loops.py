"""
Description:
            This is a basic example of the training loop for Off Policy Algorithms,
            We may move this later for each repo/env or keep this in this repo
"""

import argparse

from cares_reinforcement_learning.util import NetworkFactory

import example.policy_example as pbe
import example.value_example as vbe
import ppo_example as ppe

import gym
from gym import spaces

import torch
import random
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

def set_seed(env, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.action_space.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()# Add an argument

    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--algorithm', type=str, required=True)

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

    parser.add_argument('--max_steps_per_batch', type=float, default=5000)
    
    return vars(parser.parse_args())# converts into a dictionary

def main():
    args = parse_args()
    args["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info(f"Training on {args['task']}")
    env = gym.make(args["task"])
    
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
    agent   = factory.create_network(args["algorithm"], args)    
    
    # Train the policy or value based approach
    if args["algorithm"] == "PPO":# PPO has a different approach using a rollout can merge into value based once Rollout is refactored
        ppe.ppo_train(env, agent, args)
    elif agent.type == "policy":
        pbe.policy_based_train(env, agent, args)
    elif agent.type == "value":
        vbe.value_based_train(env, agent, args)
    else:
        raise ValueError(f"Agent type is unkown: {agent.type}")

if __name__ == '__main__':
        main()
