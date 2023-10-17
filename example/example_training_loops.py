import time
import argparse
import logging
logging.basicConfig(level=logging.INFO)

from cares_reinforcement_learning.util import NetworkFactory
from cares_reinforcement_learning.util import MemoryFactory
from cares_reinforcement_learning.util import Record
from cares_reinforcement_learning.util import EnvironmentFactory
from cares_reinforcement_learning.util import arguement_parser as ap
from cares_reinforcement_learning.util import helpers as hlp

import cares_reinforcement_learning.util.configurations as configurations
from cares_reinforcement_learning.util.configurations import TrainingConfig, AlgorithmConfig, EnvironmentConfig

import cares_reinforcement_learning.train_loops.policy_loop as pbe
import cares_reinforcement_learning.train_loops.value_loop as vbe
import cares_reinforcement_learning.train_loops.ppo_loop as ppe

import gym
from gym import spaces

import json
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(add_help=False)  # Add an argument
    parser.add_argument('-c', '--configuration_files', action='store_true', default=False)
    args, rest = parser.parse_known_args()
    args = ap.parse_args(args, rest)
    
    env_config = EnvironmentConfig.model_validate(args)
    training_config = TrainingConfig.model_validate(args)
    alg_config = configurations.create_algorithm_config(args)

    env_factory = EnvironmentFactory()
    network_factory = NetworkFactory()
    memory_factory = MemoryFactory()
    
    env = env_factory.create_environment(env_config)

    iterations_folder = f"{alg_config.algorithm}-{env_config.task}-{datetime.now().strftime('%y_%m_%d_%H:%M:%S')}"
    glob_log_dir = f'{Path.home()}/cares_rl_logs/{iterations_folder}'

    training_iterations = training_config.number_training_iterations

    seed = training_config.seed
    for training_iteration in range(0, training_iterations):
        logging.info(f"Training iteration {training_iteration+1}/{training_iterations} with Seed: {seed}")
        hlp.set_seed(seed)
        env.set_seed(seed)

        logging.info(f"Algorithm: {alg_config.algorithm}")
        agent = network_factory.create_network(env.observation_space, env.action_num, alg_config)
        if agent == None:
            raise ValueError(f"Unkown agent for default algorithms {alg_config.algorithm}")

        # TODO manage arguements for future memory types
        memory = memory_factory.create_memory(alg_config.memory, args=[])
        logging.info(f"Memory: {alg_config.memory}")

        #create the record class - standardised results tracking
        log_dir = f"{seed}"
        record = Record(glob_log_dir=glob_log_dir, log_dir=log_dir, network=agent, plot_frequency=training_config.plot_frequency, checkpoint_frequency=training_config.checkpoint_frequency)
        record.save_config(env_config, "env_config")
        record.save_config(training_config, "train_config")
        record.save_config(alg_config, "alg_config")
    
        # Train the policy or value based approach
        if alg_config.algorithm == "PPO":
            ppe.ppo_train(env, agent, record, training_config)
        elif agent.type == "policy":
            pbe.policy_based_train(env, agent, memory, record, training_config)
        elif agent.type == "value":
            vbe.value_based_train(env, agent, memory, record, training_config)
        else:
            raise ValueError(f"Agent type is unkown: {agent.type}")
        
        record.save()
        
        seed += 10

if __name__ == '__main__':
    main()

