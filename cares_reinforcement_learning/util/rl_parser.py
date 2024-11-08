import argparse
import inspect
import json
import logging
import sys
from argparse import Namespace
from typing import get_origin, Any

from pydantic import Field

from cares_reinforcement_learning.util import configurations
from cares_reinforcement_learning.util.configurations import (
    AlgorithmConfig,
    EnvironmentConfig,
    SubscriptableClass,
    TrainingConfig,
)


class RLParser:
    def __init__(self, environment_config: type[EnvironmentConfig]) -> None:
        self.configurations: dict[str, Any] = {}

        self.algorithm_parser, self.sub_algorithm_parsers = self._get_algorithm_parser()

        self.algorithm_configurations = {}
        for name, cls in inspect.getmembers(configurations, inspect.isclass):
            if issubclass(cls, AlgorithmConfig) and cls != AlgorithmConfig:
                self.algorithm_configurations[name] = cls

        self.args: dict[str, Any] = {}

        self.add_configuration("env_config", environment_config)
        self.add_configuration("train_config", TrainingConfig)

    def add_model(
        self, parser: argparse.ArgumentParser, model: type[AlgorithmConfig]
    ) -> None:
        fields = model.__fields__
        for name, field in fields.items():
            nargs = "+" if get_origin(field.annotation) is list else None
            parser.add_argument(
                f"--{name}",
                dest=name,
                type=field.type_,
                default=field.default,
                help=field.field_info.description,
                required=field.required,
                nargs=nargs,
            )

    def _get_algorithm_parser(
        self,
    ) -> tuple[argparse.ArgumentParser, argparse._SubParsersAction]:
        alg_parser = argparse.ArgumentParser()
        sub_alg_parsers = alg_parser.add_subparsers(
            help="Select which RL algorith you want to use",
            dest="algorithm",
            required=True,
        )

        for name, cls in inspect.getmembers(configurations, inspect.isclass):
            if issubclass(cls, AlgorithmConfig) and cls != AlgorithmConfig:
                name = name.replace("Config", "")
                cls_parser = sub_alg_parsers.add_parser(name, help=name)
                self.add_model(cls_parser, cls)

        return alg_parser, sub_alg_parsers

    def add_algorithm_config(self, algorithm_config: type[AlgorithmConfig]) -> None:
        name = algorithm_config.__name__.replace("Config", "")
        parser = self.sub_algorithm_parsers.add_parser(f"{name}", help=f"{name}")
        self.add_model(parser, algorithm_config)
        self.algorithm_configurations[algorithm_config.__name__] = algorithm_config

    def add_configuration(
        self, name: str, configuration: type[SubscriptableClass]
    ) -> None:
        self.configurations[name] = configuration

    def parse_args(self) -> dict:
        parser = argparse.ArgumentParser(usage="<command> [<args>]")
        # Add an argument
        parser.add_argument(
            "command",
            choices=["train", "evaluate"],
            help="Commands to run this package",
        )

        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        cmd_arg = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, f"_{cmd_arg.command}"):
            logging.error(f"Unrecognized command: {cmd_arg.command}")
            parser.print_help()
            sys.exit(1)

        # use dispatch pattern to invoke method with same name
        self.args = getattr(self, f"_{cmd_arg.command}")()
        logging.debug(self.args)

        configs = {}
        configs["command"] = cmd_arg.command

        for name, configuration in self.configurations.items():
            configuration = configuration(**self.args)
            configs[name] = configuration

        algorithm_config = self.algorithm_configurations[
            f"{self.args['algorithm']}Config"
        ](**self.args)
        configs["algorithm_config"] = algorithm_config

        return configs

    def _cli(self, initial_index) -> dict:
        parser = argparse.ArgumentParser()

        for _, configuration in self.configurations.items():
            self.add_model(parser, configuration)

        first_args, rest = parser.parse_known_args(sys.argv[initial_index:])

        alg_args, rest = self.algorithm_parser.parse_known_args(rest)

        if len(rest) > 0:
            logging.warning(
                f"Arugements not being passed properly and have been left over: {rest}"
            )

        args = Namespace(**vars(first_args), **vars(alg_args))

        return vars(args)

    def _config(self, initial_index) -> dict:
        parser = argparse.ArgumentParser()

        required = parser.add_argument_group("required arguments")

        required.add_argument(
            "--data_path",
            type=str,
            required=True,
            help="Path to training configuration files - e.g. alg_config.json, env_config.json, train_config.json",
        )

        config_args = parser.parse_args(sys.argv[initial_index:])

        args = {}

        data_path = config_args.data_path

        with open(f"{data_path}/alg_config.json", encoding="utf-8") as f:
            algorithm_config = json.load(f)

        args.update(algorithm_config)

        for name, _ in self.configurations.items():
            with open(f"{data_path}/{name}.json", encoding="utf-8") as f:
                config = json.load(f)
                args.update(config)

        return args

    def _evaluate(self) -> dict:
        return self._config(initial_index=2)

    def _train(self) -> dict:
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "load",
            choices=["cli", "config"],
            help="Set training configuration from CLI or config files",
        )

        load_arg = parser.parse_args(sys.argv[2:3])
        if load_arg.load == "cli":
            args = self._cli(initial_index=3)
        else:
            args = self._config(initial_index=3)

        return args


## Example of how to use the RLParser for custom environments -
#  in this case the LAMO envrionment and task with LAMO algorithm


class ExampleConfig(AlgorithmConfig):
    algorithm: str = Field("LMAO", Literal=True)
    lr: float = 1e-3
    gamma: float = 0.99
    memory: str = "MemoryBuffer"

    exploration_min: float = 1e-3
    exploration_decay: float = 0.95


class ExampleEnvironmentConfig(EnvironmentConfig):
    gym: str = Field("LMAO-Gym", Literal=True)
    task: str
    domain: str | None = None
    image_observation: int | None = 0


class ExampleHardwareConfig(SubscriptableClass):
    value: str = "rofl-copter"


def main():
    parser = RLParser(ExampleEnvironmentConfig)
    parser.add_configuration("lmao_config", ExampleHardwareConfig)
    parser.add_algorithm_config(ExampleConfig)
    configs = parser.parse_args()
    print(configs)


if __name__ == "__main__":
    main()
