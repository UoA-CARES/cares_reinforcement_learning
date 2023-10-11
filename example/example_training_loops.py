import time
import argparse
import logging
logging.basicConfig(level=logging.INFO)

from cares_reinforcement_learning.util import NetworkFactory
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.memory.augments import *
from cares_reinforcement_learning.util import Record
from cares_reinforcement_learning.util import EnvironmentFactory

import example.policy_example as pbe
import example.value_example as vbe
import ppo_example as ppe

import gym
from gym import spaces

import torch
import random
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def environment_args(parent_parser):
    env_parser = argparse.ArgumentParser()
    env_parsers = env_parser.add_subparsers(help='sub-command help', dest='gym_environment', required=True)

    # create the parser for the DMCS sub-command
    parser_dmcs = env_parsers.add_parser('dmcs', help='DMCS', parents=[parent_parser])
    parser_dmcs.add_argument('--domain', type=str, required=True)
    parser_dmcs.add_argument('--task', type=str, required=True)
    
    # create the parser for the OpenAI sub-command
    parser_openai = env_parsers.add_parser('openai', help='openai', parents=[parent_parser])
    parser_openai.add_argument('--task', type=str, required=True)
    return env_parser

def parse_args():
    parser = argparse.ArgumentParser(add_help=False)  # Add an argument
    
    parser.add_argument('--algorithm', type=str, required=True)
    parser.add_argument('--memory', type=str, default="MemoryBuffer")
    parser.add_argument('--image_observation', type=bool, default=False)

    parser.add_argument('--G', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--max_steps_exploration', type=int, default=10000)
    parser.add_argument('--max_steps_training', type=int, default=100000)

    parser.add_argument('--number_steps_per_evaluation', type=int, default=10000)
    parser.add_argument('--number_eval_episodes', type=int, default=10)

    parser.add_argument('--seed', type=int, default=571)
    parser.add_argument('--evaluation_seed', type=int, default=152)

    parser.add_argument('--actor_lr', type=float, default=1e-4)
    parser.add_argument('--critic_lr', type=float, default=1e-3)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--exploration_min', type=float, default=1e-3)
    parser.add_argument('--exploration_decay', type=float, default=0.95)

    parser.add_argument('--max_steps_per_batch', type=float, default=5000)

    parser.add_argument('--plot_frequency', type=int, default=100)
    parser.add_argument('--checkpoint_frequency', type=int, default=100)

    parser = environment_args(parent_parser=parser) # NOTE this has to go after the rest of parser is created

    return vars(parser.parse_args())  # converts into a dictionary

def main():
    args = parse_args()
    args["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {args['device']}")

    logging.info(f"Training on {args['task']}")
    env_factory = EnvironmentFactory()
    
    gym_environment = args['gym_environment']
    env = env_factory.create_environment(gym_environment=gym_environment, args=args)

    args["observation_size"] = env.observation_space
    logging.info(f"Observation Size: {args['observation_size']}")

    args['action_num'] = env.action_num
    logging.info(f"Action Num: {args['action_num']}")

    logging.info(f"Seed: {args['seed']}")
    set_seed(args["seed"])

    # Create the network we are using
    factory = NetworkFactory()
    logging.info(f"Algorithm: {args['algorithm']}")
    agent = factory.create_network(args["algorithm"], args)

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

    #create the record class - standardised results tracking
    record = Record(network=agent, config={'args': args})
    # Train the policy or value based approach
    if args["algorithm"] == "PPO":
        ppe.ppo_train(env, agent, record, args)
    elif agent.type == "policy":
        pbe.policy_based_train(env, agent, memory, record, args)
    elif agent.type == "value":
        vbe.value_based_train(env, agent, memory, record, args)
    else:
        raise ValueError(f"Agent type is unkown: {agent.type}")
    
    record.save()

if __name__ == '__main__':
    main()

