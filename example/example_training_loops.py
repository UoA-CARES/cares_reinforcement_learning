import time
import argparse

from cares_reinforcement_learning.util import NetworkFactory
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.memory.augments import *
from cares_reinforcement_learning.util import Record

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
    parser = argparse.ArgumentParser()  # Add an argument

    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--render', type=str, default="None")
    parser.add_argument('--algorithm', type=str, required=True)
    parser.add_argument('--memory', type=str, default="MemoryBuffer")

    parser.add_argument('--G', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--max_steps_exploration', type=int, default=10000)
    parser.add_argument('--max_steps_training', type=int, default=50000)

    parser.add_argument('--number_steps_per_evaluation', type=int, default=1000)
    parser.add_argument('--number_eval_episodes', type=int, default=10)

    parser.add_argument('--seed', type=int, default=571)
    parser.add_argument('--evaluation_seed', type=int, default=152)

    parser.add_argument('--actor_lr', type=float, default=1e-4)
    parser.add_argument('--critic_lr', type=float, default=1e-3)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--exploration_min', type=float, default=1e-3)
    parser.add_argument('--exploration_decay', type=float, default=0.95)

    parser.add_argument('--max_steps_per_batch', type=float, default=5000)

    parser.add_argument('--plot_frequency', type=int, default=10)
    
    parser.add_argument('--display', type=str, default=True)

    return vars(parser.parse_args())  # converts into a dictionary

def main():
    args = parse_args()
    args["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f"Training on {args['task']}")
    env = gym.make(args["task"], render_mode=(None if args['render'] == "None" else args['render']))

    logging.info(f"Device: {args['device']}")

    args["observation_size"] = env.observation_space.shape[0]
    logging.info(f"Observation Size: {args['observation_size']}")

    if type(env.action_space) == spaces.Box:
        args["action_num"] = env.action_space.shape[0]
    elif type(env.action_space) == spaces.Discrete:
        args["action_num"] = env.action_space.n
    else:
        raise ValueError(f"Unhandled action space type: {type(env.action_space)}")
    logging.info(f"Action Num: {args['action_num']}")

    logging.info(f"Seed: {args['seed']}")
    set_seed(env, args["seed"])

    # Create the network we are using
    factory = NetworkFactory()
    logging.info(f"Algorithm: {args['algorithm']}")
    agent = factory.create_network(args["algorithm"], args)
    logging.info(f"Algorithm: {args['algorithm']}")

    # TODO move to memory factory as we add new PER
    if args["memory"] == "MemoryBuffer":
        memory = MemoryBuffer()
    elif args["memory"] == "PER":
        memory = MemoryBuffer(augment=td_error)
    else:
        error_message = f"Unkown memory type: {args['memory']}"
        logging.error(error_message)
        raise ValueError(error_message)
    
    logging.info(f"Memory: {args['memory']}")

    # Train the policy or value based approach
    record = Record(network=agent, config={'args': args})
    if args["algorithm"] == "PPO":
        #create the record class
        ppe.ppo_train(env, agent, record, args)
        env = gym.make(env.spec.id, render_mode="human")
        ppe.evaluate_ppo_network(env, agent, args)
    elif agent.type == "policy":
        pbe.policy_based_train(env, agent, memory, record, args)
        env = gym.make(env.spec.id, render_mode="human")
        pbe.evaluate_policy_network(env, agent, args)
    elif agent.type == "value":
        vbe.value_based_train(env, agent, memory, record, args)
        env = gym.make(env.spec.id, render_mode="human")
        vbe.evaluate_value_network(env, agent, args)
    else:
        raise ValueError(f"Agent type is unkown: {agent.type}")
    
    record.save()

if __name__ == '__main__':
    main()

