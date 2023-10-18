"""
Example of using sub-parser, sub-commands and sub-sub-commands :-)
"""
import os
import sys
import argparse
from argparse import Namespace
import logging

import cares_reinforcement_learning.util.configurations as configurations
from cares_reinforcement_learning.util.configurations import TrainingConfig, AlgorithmConfig, EnvironmentConfig
import json

import rich

import pydantic

import importlib
import inspect

def factory(classname, args):
    import cares_reinforcement_learning.util.configurations as configurations
    cls = getattr(configurations, classname)
    return cls(**vars(args))

def add_model(parser, model):
    "Add Pydantic model to an ArgumentParser"
    fields = model.__fields__
    for name, field in fields.items():
        parser.add_argument(
            f"--{name}", 
            dest=name, 
            type=field.type_, 
            default=field.default,
            help=field.field_info.description,
            required=field.required
        )

def get_algorithm_parser():
    alg_parser = argparse.ArgumentParser()
    alg_parsers = alg_parser.add_subparsers(help='Select which RL algorith you want to use', dest='algorithm', required=True)

    # create the parser for TD3 with default parameters
    parser_TD3 = alg_parsers.add_parser('TD3', help='TD3')
    add_model(parser_TD3, configurations.TD3Config)
    
    # create the parser for DDPG with default parameters
    parser_DDPG = alg_parsers.add_parser('DDPG', help='DDPG')
    add_model(parser_DDPG, configurations.DDPGConfig)

    # create the parser for SAC with default parameters
    parser_SAC = alg_parsers.add_parser('SAC', help='SAC')
    add_model(parser_SAC, configurations.SACConfig)

    # create the parser for PPO with default parameters
    parser_PPO = alg_parsers.add_parser('PPO', help='PPO')
    add_model(parser_PPO, configurations.PPOConfig)

    # create the parser for DQN with default parameters
    parser_DQN = alg_parsers.add_parser('DQN', help='DQN')
    add_model(parser_DQN, configurations.DQNConfig)

    # create the parser for DuelingDQN with default parameters
    parser_DuelingDQN = alg_parsers.add_parser('DuelingDQN', help='DuelingDQN')
    add_model(parser_DuelingDQN, configurations.DuelingDQNConfig)

    # create the parser for DoubleDQN with default parameters
    parser_DoubleDQN = alg_parsers.add_parser('DoubleDQN', help='DoubleDQN')
    add_model(parser_DoubleDQN, configurations.DoubleDQNConfig)

    return alg_parser, alg_parsers

class RLParser:
    def __init__(self) -> None:
        self.algorithm_parser, self.algorithm_parsers = get_algorithm_parser()

        self.algorithm_configs = {}
        for name, cls in inspect.getmembers(configurations, inspect.isclass):
            self.algorithm_configs[name] = cls
    
    def add_algorithm(self, name, algorithm_model):
        parser = self.algorithm_parsers.add_parser(f"{name}", help=f"{name}")
        add_model(parser, algorithm_model)
        self.algorithm_configs[algorithm_model.__name__] = algorithm_model

    def parse_args(self):
        parser = argparse.ArgumentParser(usage="<command> [<args>]")
        # Add an argument
        parser.add_argument('command', choices=["config", "run"], help="Commands to run this package")
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        cmd_arg = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, f"_{cmd_arg.command}"):
            logging.error(f"Unrecognized command: {cmd_arg.command}")
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        self.args = getattr(self, f"_{cmd_arg.command}")()

        env_config = EnvironmentConfig(**self.args)
        training_config = TrainingConfig(**self.args)
        algorithm_config = self.algorithm_configs[f"{self.args['algorithm']}Config"](**self.args)
        return env_config, training_config, algorithm_config
        
    def _config(self):
        parser = argparse.ArgumentParser()
        required = parser.add_argument_group('required arguments')
        required.add_argument("--env_config", type=str, required=True, help='Configuration path for the environment')
        required.add_argument("--training_config", type=str, required=True, help='Configuration path that defines the training parameters')
        required.add_argument("--algorithm_config", type=str, required=True, help='Configuration path that defines the algorithm and its learning parameters')

        config_args = parser.parse_args(sys.argv[2:])
        args = {}
        with open(config_args.env_config) as f:
            env_args = json.load(f)
        
        with open(config_args.training_config) as f:
            training_config = json.load(f)

        with open(config_args.algorithm_config) as f:
            algorithm_config = json.load(f)

        args.update(env_args)
        args.update(training_config)
        args.update(algorithm_config)
        return args

    def _run(self):
        parser = argparse.ArgumentParser()

        add_model(parser, EnvironmentConfig)
        add_model(parser, TrainingConfig)
        firt_args, rest = parser.parse_known_args(sys.argv[2:])

        alg_args, rest = self.algorithm_parser.parse_known_args(rest)

        args = Namespace(**vars(firt_args), **vars(alg_args))
        return vars(args)

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
class LMAOConfig(AlgorithmConfig):
    algorithm: str = Field("LMAO", Literal=True)
    lr: Optional[float] = 1e-3
    gamma: Optional[float] = 0.99
    memory: Optional[str] = "MemoryBuffer"

    exploration_min: Optional[float] = 1e-3
    exploration_decay: Optional[float] = 0.95

if __name__ == '__main__':
    parser = RLParser()
    parser.add_algorithm("LMAO", LMAOConfig)
    parser.parse_args()