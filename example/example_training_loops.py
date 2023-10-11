import time
import argparse
import logging
logging.basicConfig(level=logging.INFO)

from cares_reinforcement_learning.util import NetworkFactory
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.memory.augments import *
from cares_reinforcement_learning.util import Record
from cares_reinforcement_learning.util import EnvironmentFactory
from cares_reinforcement_learning.util import arguement_parser as ap

import example.policy_example as pbe
import example.value_example as vbe
import ppo_example as ppe

import gym
from gym import spaces

import torch
import random
import numpy as np
from pathlib import Path

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    args = ap.parse_args()

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

    seed = args['seed']

    training_iterations = args['number_training_iterations']
    for training_iteration in range(0, training_iterations):
        logging.info(f"Training iteration {training_iteration+1}/{training_iterations} with Seed: {seed}")
        set_seed(seed)
        env.set_seed(seed)

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
        seed += 10
    
    record.save()

if __name__ == '__main__':
    main()

