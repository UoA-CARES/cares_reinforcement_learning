import importlib.util
import inspect
import os
from pathlib import Path

from torch import nn

from cares_reinforcement_learning.networks.common import MLP
from cares_reinforcement_learning.util import configurations
from cares_reinforcement_learning.util.configurations import AlgorithmConfig


def _extract_sequential(model: nn.Module) -> nn.Sequential | None:
    if isinstance(model, nn.Sequential):
        return model

    for _, layer in model.named_children():
        model = _extract_sequential(layer)
        if model is not None:
            return model

    return None


def _compare_layers(layer1: nn.Module, layer2: nn.Module) -> bool:
    # Compare layer parameters
    params1 = dict(layer1.named_parameters())
    params2 = dict(layer2.named_parameters())
    if len(params1) != len(params2):
        print(f"Parameter lengths do not match {len(params1)=} {len(params2)=}")
        return False

    for param_name in params1:
        if param_name not in params2:
            print(f"Parameter {param_name} not found in second model")
            return False

    named_modules1 = list(layer1.named_modules())
    named_modules2 = list(layer2.named_modules())

    for (name1, layer1), (name2, layer2) in zip(named_modules1, named_modules2):
        # Ensure both are nn.Module instances
        if isinstance(layer1, nn.Module) and isinstance(layer2, nn.Module):
            # Compare parameter shapes
            params1 = [p.shape for p in layer1.parameters()]
            params2 = [p.shape for p in layer2.parameters()]

            if params1 != params2:
                print(f"Mismatch in layer: {name1} vs {name2}")
                print(f"Shapes: {params1} vs {params2}")
                return False

    return True


def _compare_sequential_layers(seq1: nn.Sequential, seq2: nn.Sequential) -> bool:
    """Compares the layers in two nn.Sequential modules."""
    layers1 = list(seq1.children())
    layers2 = list(seq2.children())

    if len(layers1) != len(layers2):
        print(f"Layer lengths do not match {len(layers1)=} {len(layers2)=}")
        return False

    for layer1, layer2 in zip(layers1, layers2):
        # Compare layer types
        if type(layer1) != type(layer2):
            print(f"Layer types do not match {type(layer1)=} {type(layer2)=}")
            return False

        # Compare layer parameters
        equal = _compare_layers(layer1, layer2)
        if not equal:
            print(f"Layers do not match {layer1=} vs {layer2=}")
            return False

    return True


def _compare_networks(model1: nn.Module, model2: nn.Module) -> bool:
    """Compares two PyTorch models."""
    # print(f"{model1=}")
    name_modules1 = dict(model1.named_children())

    # print(f"{model2=}")
    name_modules2 = dict(model2.named_children())

    if f"{type(model1).__name__}" == f"Default{type(model2).__name__}":
        print(f"Networks do not match {model1=} vs {model2=}")
        return False

    for name, module1 in name_modules1.items():
        if name not in name_modules2:
            print(f"Module {name} not found in second model")
            return False

        module2 = name_modules2[name]

        seq1 = _extract_sequential(module1)
        seq2 = _extract_sequential(module2)

        if seq1 is not None and seq2 is not None:

            equal = _compare_sequential_layers(seq1, seq2)
            if not equal:
                print(f"Sequential layers do not match {name=} {seq1=} vs {seq2=}")
                return False

            # skip the other checks
            continue

        if type(module1) != type(module2):
            print(f"Module types do not match {type(module1)=} vs {type(module2)=}")
            return False
        elif not _compare_layers(module1, module2):
            print(f"Layers do not match {module1=} vs {module2=}")
            return False

    return True


def _import_modules_from_folder(folder_path: str):
    modules = {}

    # Iterate through files in the folder
    for file_name in os.listdir(folder_path):
        # Check for Python files (excluding __init__.py)
        if file_name.endswith(".py") and file_name != "__init__.py":
            module_name = file_name[:-3]  # Strip ".py" extension
            file_path = os.path.join(folder_path, file_name)

            # Dynamically load the module
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # Execute the module
            modules[module_name] = module

    return modules


def _split_dict_by_default_keys(input_dict: dict):
    default_group = {}
    non_default_group = {}

    for key, value in input_dict.items():
        if key.startswith("default_"):
            # Remove the "default_" prefix for uniformity
            stripped_key = key[len("default_") :]
            default_group[stripped_key] = value
        else:
            non_default_group[key] = value

    return default_group, non_default_group


def _get_defined_classes(module):
    classes = inspect.getmembers(module, inspect.isclass)
    module_name = module.__name__

    defined_classes = {}
    for name, cls in classes:
        if issubclass(cls, AlgorithmConfig) or cls == MLP:
            continue

        if "Default" in name:
            name = "default"
        elif "Base" in name:
            name = "base"
        else:
            name = "custom"

        if name in defined_classes and cls.__module__ == module_name:
            defined_classes[name] = cls
        elif name not in defined_classes:
            defined_classes[name] = cls

    return defined_classes


def test_actor_critics():

    algorithm_configurations = {}
    for name, cls in inspect.getmembers(configurations, inspect.isclass):
        if issubclass(cls, AlgorithmConfig) and cls != AlgorithmConfig:
            name = name.replace("Config", "")
            algorithm_configurations[name] = cls

    observation_size_vector = 12

    observation_size_image = (9, 32, 32)

    action_num = 4

    for algorithm, alg_config in algorithm_configurations.items():
        alg_config = alg_config()

        observation_size = (
            {"image": observation_size_image, "vector": observation_size_vector}
            if alg_config.image_observation
            else observation_size_vector
        )

        configuration_path = Path(configurations.__file__).parent
        network_path = f"{configuration_path.parent}/networks/{algorithm}"

        modules = _import_modules_from_folder(network_path)

        model_equal = True

        for name, module in modules.items():

            module_equal = False

            defined_classes = _get_defined_classes(module)

            if "default" in defined_classes:
                cls = defined_classes["custom"]
                default_cls = defined_classes["default"]

                init_method = getattr(cls, "__init__", None)

                init_signature = inspect.signature(init_method)
                number_of_parameters = len(init_signature.parameters)

                if number_of_parameters == 4:

                    try:
                        network = cls(observation_size, action_num, alg_config)
                        default_network = default_cls(observation_size, action_num)
                    except TypeError:
                        network = cls(
                            observation_size["vector"], action_num, alg_config
                        )
                        default_network = default_cls(
                            observation_size["vector"], action_num
                        )

                elif number_of_parameters == 3:
                    network = cls(observation_size, alg_config)
                    default_network = default_cls(observation_size)
                else:
                    raise ValueError(
                        f"Unexpected number of parameters {number_of_parameters} {algorithm} {name}"
                    )

                module_equal = _compare_networks(network, default_network)

            model_equal = model_equal and module_equal

            if not module_equal:
                assert (
                    module_equal
                ), f"{algorithm} {name} model doesn't match default with custom"

        assert model_equal, f"{algorithm} models do not match"
