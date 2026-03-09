import argparse
import ast
import inspect
import json
import logging
import sys
from argparse import Namespace
from typing import Any, Union, get_args, get_origin

from pydantic_core import PydanticUndefined

import cares_reinforcement_learning.envs.configurations as cfg
from cares_reinforcement_learning.algorithm import configurations
from cares_reinforcement_learning.algorithm.configurations import (
    AlgorithmConfig,
    SubscriptableClass,
    TrainingConfig,
)


# TODO command specific args
class RunConfig(SubscriptableClass):
    command: str
    data_path: str | None = None

    eval_seed: int | None = None
    episodes: int | None = None


def annotation_to_argparse_type(annotation):
    """
    Convert a Pydantic v2 annotation into an argparse-compatible type function.
    """
    origin = get_origin(annotation)

    # Handle Optional[T] or Union[T, None]
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return annotation_to_argparse_type(args[0])
        else:
            # Multi-type unions cannot map to a single type
            return str

    # Handle list inputs
    if origin is list:
        elem_type = get_args(annotation)[0]
        return annotation_to_argparse_type(elem_type)

    # Handle dict / tuple by parsing literal input
    if origin in (dict, tuple):
        return ast.literal_eval

    # Basic built-in types
    if annotation in (int, float, str, bool):
        return annotation

    # Fallback
    return str


class RLParser:
    def __init__(self) -> None:
        self.configurations: dict[str, Any] = {}

        self.environment_parser, self.sub_environment_parsers = (
            self._get_environment_parser()
        )

        self.environment_configurations = {}
        for class_name, cls in inspect.getmembers(cfg, inspect.isclass):
            if (
                issubclass(cls, cfg.GymEnvironmentConfig)
                and cls != cfg.GymEnvironmentConfig
            ):
                self.environment_configurations[cls.gym] = cls

        self.algorithm_parser, self.sub_algorithm_parsers = self._get_algorithm_parser()

        self.algorithm_configurations = {}
        for class_name, cls in inspect.getmembers(configurations, inspect.isclass):
            if issubclass(cls, AlgorithmConfig) and cls != AlgorithmConfig:
                self.algorithm_configurations[class_name] = cls

        self.args: dict[str, Any] = {}

        self.add_configuration("train_config", TrainingConfig)

    def _add_model(
        self,
        parser: argparse.ArgumentParser,
        model: type[AlgorithmConfig] | type[cfg.GymEnvironmentConfig],
    ) -> None:
        fields = model.model_fields
        for name, field in fields.items():
            # Check for list type (or other iterable types)
            nargs = "+" if get_origin(field.annotation) is list else None

            # Default value handling
            if field.default is not PydanticUndefined:
                default_value = field.default
            elif field.default_factory is not None:
                default_value = field.default_factory()  # type: ignore
            else:
                default_value = None

            # argparse type resolution
            arg_type = annotation_to_argparse_type(field.annotation)

            parser.add_argument(
                f"--{name}",
                dest=name,
                type=arg_type,
                default=default_value,
                help=field.description,
                required=field.is_required(),  # type: ignore[arg-type]
                nargs=nargs,
            )

    def _get_environment_parser(
        self,
    ) -> tuple[argparse.ArgumentParser, argparse._SubParsersAction]:
        env_parser = argparse.ArgumentParser()

        sub_env_parsers = env_parser.add_subparsers(
            help="Select which gym you want to use",
            dest="gym",
            required=True,
        )

        for name, cls in inspect.getmembers(cfg, inspect.isclass):
            if (
                issubclass(cls, cfg.GymEnvironmentConfig)
                and cls != cfg.GymEnvironmentConfig
            ):
                name = cls.gym
                cls_parser = sub_env_parsers.add_parser(name, help=name)
                self._add_model(cls_parser, cls)

        return env_parser, sub_env_parsers

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
                self._add_model(cls_parser, cls)

        return alg_parser, sub_alg_parsers

    def add_algorithm_config(self, algorithm_config: type[AlgorithmConfig]) -> None:
        name = algorithm_config.__name__.replace("Config", "")
        parser = self.sub_algorithm_parsers.add_parser(f"{name}", help=f"{name}")
        self._add_model(parser, algorithm_config)
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
            choices=["train", "evaluate", "test", "resume"],
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
        run_args, self.args = getattr(self, f"_{cmd_arg.command}")()
        logging.debug(self.args)

        configs: dict[str, SubscriptableClass] = {}

        configs["run_config"] = RunConfig(command=cmd_arg.command, **run_args)

        configs["env_config"] = self.environment_configurations[f"{self.args['gym']}"](
            **self.args
        )

        for name, configuration in self.configurations.items():
            configuration = configuration(**self.args)
            configs[name] = configuration

        algorithm_config = self.algorithm_configurations[
            f"{self.args['algorithm']}Config"
        ](**self.args)
        configs["alg_config"] = algorithm_config

        return configs

    def _cli(self, initial_index) -> dict:
        parser = argparse.ArgumentParser()

        for _, configuration in self.configurations.items():
            self._add_model(parser, configuration)

        first_args, rest = parser.parse_known_args(sys.argv[initial_index:])

        env_args, rest = self.environment_parser.parse_known_args(rest)
        alg_args, rest = self.algorithm_parser.parse_known_args(rest)

        if len(rest) > 0:
            logging.warning(
                f"Arugements not being passed properly and have been left over: {rest}"
            )

        args = Namespace(**vars(first_args), **vars(env_args), **vars(alg_args))

        return vars(args)

    def _load_args_from_configs(self, data_path: str) -> dict:
        args = {}

        with open(f"{data_path}/alg_config.json", encoding="utf-8") as f:
            algorithm_config = json.load(f)

        args.update(algorithm_config)

        with open(f"{data_path}/env_config.json", encoding="utf-8") as f:
            environment_config = json.load(f)

        args.update(environment_config)

        for name, _ in self.configurations.items():
            with open(f"{data_path}/{name}.json", encoding="utf-8") as f:
                config = json.load(f)
                args.update(config)

        print(f"Loaded args from {data_path}: {args}")

        return args

    def _config(self, initial_index) -> tuple[str, dict]:
        parser = argparse.ArgumentParser()

        required = parser.add_argument_group("required arguments")

        required.add_argument(
            "--data_path",
            type=str,
            required=True,
            help="Path to training configuration files - e.g. alg_config.json, env_config.json, train_config.json",
        )

        config_args = parser.parse_args(sys.argv[initial_index:])

        data_path = config_args.data_path

        return data_path, self._load_args_from_configs(data_path)

    def _test(self) -> tuple[dict, dict]:
        parser = argparse.ArgumentParser()

        required = parser.add_argument_group("required arguments")

        required.add_argument(
            "--data_path",
            type=str,
            required=True,
            help="Path to testing configuration files - e.g. alg_config.json, env_config.json, train_config.json",
        )

        required.add_argument(
            "--eval_seed",
            type=int,
            required=True,
            help="Seed to use for testing trained models against",
        )

        required.add_argument(
            "--episodes",
            type=int,
            required=True,
            help="Number of evaluation episodes to run",
        )

        run_args = parser.parse_args(sys.argv[2:])

        model_args = self._load_args_from_configs(run_args.data_path)

        return vars(run_args), model_args

    def _evaluate(self) -> tuple[dict, dict]:
        data_path, model_args = self._config(initial_index=2)

        run_args = {"data_path": data_path}

        return run_args, model_args

    def _train(self) -> tuple[dict, dict]:
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "load",
            choices=["cli", "config"],
            help="Set training configuration from CLI or config files",
        )

        run_args = {}

        load_arg = parser.parse_args(sys.argv[2:3])
        if load_arg.load == "cli":
            args = self._cli(initial_index=3)
        else:
            data_path, args = self._config(initial_index=3)
            run_args = {"data_path": data_path}

        return run_args, args

    def _resume(self) -> tuple[dict, dict]:
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--data_path",
            type=str,
            required=True,
            help="Path to training files - e.g. alg_config.json, env_config.json, train_config.json",
        )

        run_args = parser.parse_args(sys.argv[2:])

        model_args = self._load_args_from_configs(run_args.data_path)

        return vars(run_args), model_args


## Example of how to use the RLParser for custom environments -
#  in this case the Example environment and task with Example algorithm


class ExampleHardwareConfig(SubscriptableClass):
    value: str = "rofl-copter"


def main():
    parser = RLParser()
    parser.add_configuration("hardware_config", ExampleHardwareConfig)
    configs = parser.parse_args()
    print(configs.keys())
    print(configs["env_config"])


if __name__ == "__main__":
    main()
