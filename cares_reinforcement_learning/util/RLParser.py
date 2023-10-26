"""
Example of using sub-parser, sub-commands and sub-sub-commands :-)
"""
import os
import sys
import argparse
from argparse import Namespace
import logging

import cares_reinforcement_learning.util.configurations as configurations
from cares_reinforcement_learning.util.configurations import TrainingConfig, AlgorithmConfig, GymEnvironmentConfig, EnvironmentConfig, SubscriptableClass
import json

import pydantic

import importlib
import inspect
from typing import get_origin

class RLParser:
    def __init__(self, EnvironmentConfig = GymEnvironmentConfig) -> None:
        self.configurations = {}
        
        self.algorithm_parser, self.algorithm_parsers = self.get_algorithm_parser()

        self.algorithm_configurations = {}
        for name, cls in inspect.getmembers(configurations, inspect.isclass):
            if issubclass(cls, AlgorithmConfig) and cls != AlgorithmConfig:
                self.algorithm_configurations[name] = cls

        self.add_configuration("env_config", EnvironmentConfig)
        self.add_configuration("training_config", TrainingConfig)

    def add_model(self, parser, model):
        fields = model.__fields__
        for name, field in fields.items():
            nargs = '+' if get_origin(field.annotation) is list else None
            parser.add_argument(
                f"--{name}", 
                dest=name, 
                type=field.type_, 
                default=field.default,
                help=field.field_info.description,
                required=field.required,
                nargs=nargs
            )

    def get_algorithm_parser(self):
        alg_parser = argparse.ArgumentParser()
        alg_parsers = alg_parser.add_subparsers(help='Select which RL algorith you want to use', dest='algorithm', required=True)

        for name, cls in inspect.getmembers(configurations, inspect.isclass):
            if issubclass(cls, AlgorithmConfig) and cls != AlgorithmConfig:
                name = name.replace('Config', '')
                cls_parser = alg_parsers.add_parser(name, help=name)
                self.add_model(cls_parser, cls)

        return alg_parser, alg_parsers

    def add_algorithm_config(self, algorithm_model):
        name = algorithm_model.__name__.replace('Config', '')
        parser = self.algorithm_parsers.add_parser(f"{name}", help=f"{name}")
        self.add_model(parser, algorithm_model)
        self.algorithm_configurations[algorithm_model.__name__] = algorithm_model

    def add_configuration(self, name, Configuration):
        self.configurations[name] = Configuration

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
        print(self.args)
        
        configurations = {}

        for name, Configuration in self.configurations.items():
            configuration = Configuration(**self.args)
            configurations[name] = configuration

        algorithm_config = self.algorithm_configurations[f"{self.args['algorithm']}Config"](**self.args)
        configurations['algorithm_config'] = algorithm_config

        return configurations

    def _config(self):
        parser = argparse.ArgumentParser()
        required = parser.add_argument_group('required arguments')
        required.add_argument("--algorithm_config", type=str, required=True, help='Configuration path that defines the algorithm and its learning parameters')

        for name, configuration in self.configurations.items():
            required.add_argument(f"--{name}", required=True, help=f"Configuration path for {name}")

        config_args = parser.parse_args(sys.argv[2:])

        args = {}

        with open(config_args.algorithm_config) as f:
            algorithm_config = json.load(f)

        args.update(algorithm_config)

        config_args = vars(config_args)
        for name, configuration in self.configurations.items():
            with open(config_args[name]) as f:
                config = json.load(f)
                args.update(config)

        return args

    def _run(self):
        parser = argparse.ArgumentParser()

        for _, Configuration in self.configurations.items():
            self.add_model(parser, Configuration)

        firt_args, rest = parser.parse_known_args(sys.argv[2:])

        alg_args, rest = self.algorithm_parser.parse_known_args(rest)

        if len(rest) > 0:
            logging.warn(f"Arugements not being passed properly and have been left over: {rest}")

        args = Namespace(**vars(firt_args), **vars(alg_args))
        return vars(args)

## Example of how to use the RLParser for custom environments - in this case the LAMO task
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
class LMAOConfig(AlgorithmConfig):
    algorithm: str = Field("LMAO", Literal=True)
    lr: Optional[float] = 1e-3
    gamma: Optional[float] = 0.99
    memory: Optional[str] = "MemoryBuffer"

    exploration_min: Optional[float] = 1e-3
    exploration_decay: Optional[float] = 0.95

class LMAOEnvironmentConfig(EnvironmentConfig):
    gym: str = Field("LMAO-Gym", Literal=True)
    task: str
    domain: Optional[str] = None
    image_observation: Optional[bool] = False

class LMAOHardwareConfig(SubscriptableClass):
    value: str = "rofl-copter"

if __name__ == '__main__':
    parser = RLParser(LMAOEnvironmentConfig)
    parser.add_configuration("lmao_config",LMAOHardwareConfig)
    parser.add_algorithm_config(LMAOConfig)
    configurations = parser.parse_args()
    print(configurations)