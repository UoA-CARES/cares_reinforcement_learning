import sys
import time
import argparse
import logging
logging.basicConfig(level=logging.INFO)

from cares_reinforcement_learning.util import NetworkFactory
from cares_reinforcement_learning.util import MemoryFactory
from cares_reinforcement_learning.util import Record
from cares_reinforcement_learning.util import EnvironmentFactory
from cares_reinforcement_learning.util import RLParser
from cares_reinforcement_learning.util import helpers as hlp

import cares_reinforcement_learning.util.configurations as configurations
from cares_reinforcement_learning.util.configurations import TrainingConfig, AlgorithmConfig, GymEnvironmentConfig

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
    parser = RLParser()
    
    configurations = parser.parse_args()
    env_config = configurations["env_config"] 
    training_config = configurations["training_config"]
    alg_config = configurations["algorithm_config"]
    
    env_factory = EnvironmentFactory()
    network_factory = NetworkFactory()
    memory_factory = MemoryFactory()
    
    env = env_factory.create_environment(env_config)

    iterations_folder = f"{alg_config.algorithm}-{env_config.task}-{datetime.now().strftime('%y_%m_%d_%H:%M:%S')}"
    glob_log_dir = f'{Path.home()}/cares_rl_logs/{iterations_folder}'

    for training_iteration, seed in enumerate(training_config.seeds):
        logging.info(f"Training iteration {training_iteration+1}/{len(training_config.seeds)} with Seed: {seed}")
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
        record = Record(glob_log_dir=glob_log_dir, 
                        log_dir=log_dir, 
                        algorithm=alg_config.algorithm, 
                        task=env_config.task, 
                        network=agent, 
                        plot_frequency=training_config.plot_frequency, 
                        checkpoint_frequency=training_config.checkpoint_frequency)
        
        record.save_config(env_config, "env_config")
        record.save_config(training_config, "train_config")
        record.save_config(alg_config, "alg_config")
    
        # Train the policy or value based approach
        if alg_config.algorithm == "PPO":
            ppe.ppo_train(env, agent, record, training_config, alg_config)
        elif agent.type == "policy":
            pbe.policy_based_train(env, agent, memory, record, training_config, alg_config)
        elif agent.type == "value":
            vbe.value_based_train(env, agent, memory, record, training_config, alg_config)
        else:
            raise ValueError(f"Agent type is unkown: {agent.type}")
        
        record.save()

if __name__ == '__main__':
    main()

