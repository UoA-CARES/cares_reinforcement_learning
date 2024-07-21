import random
import os
from datetime import datetime

import numpy as np
import torch


def create_path_from_format_string(
    format_str: str,
    algorithm: str,
    domain: str,
    task: str,
    gym: str,
    seed: int,
    run_name: str,
) -> str:
    """
    Create a path from a format string
    :param format_str: The format string to use
    :param domain: The domain of the environment
    :param task: The task of the environment
    :param gym: The gym environment
    :param seed: The seed used
    :param run_name: The name of the run
    :return: The path
    """

    base_dir = os.environ.get("CARES_LOG_DIR", f"{Path.home()}/cares_rl_logs")

    domain_with_hyphen_or_empty = f"{domain}-" if domain != "" else ""
    domain_task = domain_with_hyphen_or_empty + task

    date = datetime.now().strftime("%y_%m_%d_%H-%M-%S")

    run_name_else_date = run_name if run_name != "" else date

    log_dir = format_str.format(
        algorithm=algorithm,
        domain=domain,
        task=task,
        gym=gym,
        run_name=run_name,
        run_name_else_date=run_name_else_date,
        seed=seed,
        domain_task=domain_task,
        date=date,
    )
    return f"{base_dir}/{log_dir}"


def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The random seed value.

    Returns:
        None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)


def soft_update_params(net, target_net, tau):
    """
    Soft updates the parameters of a neural network by blending them with the parameters of a target network.

    Args:
        net (torch.nn.Module): The neural network whose parameters will be updated.
        target_net (torch.nn.Module): The target neural network whose parameters will be blended with the `net` parameters.
        tau (float): The blending factor. The updated parameters will be a weighted average of the `net` parameters and the `target_net` parameters.

    Returns:
        None
    """
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def weight_init(module: torch.nn.Module) -> None:
    """
    Custom weight init for Conv2D and Linear layers

    delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
    """
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.orthogonal_(module.weight.data)
        module.bias.data.fill_(0.0)

    elif isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        assert module.weight.size(2) == module.weight.size(3)
        module.weight.data.fill_(0.0)
        module.bias.data.fill_(0.0)
        mid = module.weight.size(2) // 2
        gain = torch.nn.init.calculate_gain("relu")
        torch.nn.init.orthogonal_(module.weight.data[:, :, mid, mid], gain)


def normalize_observation(observation: torch.Tensor, statistics: dict) -> torch.Tensor:
    """
    This normalization is applied to world models inputs.
    Normalize the states based on the statistics from experience replay.

    Args:
        observation (torch.Tensor): The observation tensor to be normalized.
        statistics (dict): The statistics from experience replay.

    Returns:
        torch.Tensor: The normalized observation tensor.

    """
    return (observation - statistics["observation_mean"]) / statistics[
        "observation_std"
    ]


def denormalize_observation_delta(
    normalized_delta: torch.Tensor, statistics: dict
) -> torch.Tensor:
    """
    This denormlizing is applied to world models predicitons to restore the range of state difference between next and
    current.

    Args:
        normalized_delta (torch.Tensor): The normalized delta tensor to be denormalized.
        statistics (dict): The statistics from experience replay.

    Returns:
        torch.Tensor: The denormalized delta tensor.

    """
    return (normalized_delta * statistics["delta_std"]) + statistics["delta_mean"]


def normalize_observation_delta(delta: torch.Tensor, statistics: dict) -> torch.Tensor:
    """
    This normalization is applied to world models' target lables. The world model is predicting the difference between
    current states and next states. This normalization is applied to the deltas.

    Args:
        delta (torch.Tensor): The delta tensor to be normalized.
        statistics (dict): The statistics from experience replay.

    Returns:
        torch.Tensor: The normalized delta tensor.

    """
    return (delta - statistics["delta_mean"]) / statistics["delta_std"]


def denormalize(
    action: float, max_action_value: float, min_action_value: float
) -> float:
    """
    Denormalize the given action value within the specified range.

    Args:
        action (float): The action value to be denormalized.
        max_action_value (float): The maximum value of the action range.
        min_action_value (float): The minimum value of the action range.

    Returns:
        float: The denormalized action value.

    """
    max_range_value: float = max_action_value
    min_range_value: float = min_action_value
    max_value_in: float = 1
    min_value_in: float = -1
    action_denorm: float = (action - min_value_in) * (
        max_range_value - min_range_value
    ) / (max_value_in - min_value_in) + min_range_value
    return action_denorm


def normalize(action: float, max_action_value: float, min_action_value: float) -> float:
    """
    Normalize the given action value within the specified range.

    Args:
        action (float): The action value to be normalized.
        max_action_value (float): The maximum value of the action range.
        min_action_value (float): The minimum value of the action range.

    Returns:
        float: The normalized action value.

    """
    max_range_value: float = 1
    min_range_value: float = -1
    max_value_in: float = max_action_value
    min_value_in: float = min_action_value
    action_norm: float = (action - min_value_in) * (
        max_range_value - min_range_value
    ) / (max_value_in - min_value_in) + min_range_value
    return action_norm


def compare_models(model_1: torch.nn.Module, model_2: torch.nn.Module) -> bool:
    """
    Compare two PyTorch models and check if their state dictionaries are equal.

    Args:
        model_1 (torch.nn.Module): The first PyTorch model.
        model_2 (torch.nn.Module): The second PyTorch model.

    Returns:
        bool: True if the models have equal state dictionaries, False otherwise.

    Raises:
        ValueError: If the models have different state dictionaries.

    """
    models_differ = 0
    for key_item_1, key_item_2 in zip(
        model_1.state_dict().items(), model_2.state_dict().items()
    ):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if key_item_1[0] == key_item_2[0]:
                print("Mismatch found at", key_item_1[0])
            else:
                raise ValueError(
                    f"Models are not equal. {key_item_1[0]} is not equal to {key_item_2[0]}"
                )
    return models_differ == 0


def prioritized_approximate_loss(
    x: torch.Tensor, min_priority: float, alpha: float
) -> torch.Tensor:
    """
    Calculates the prioritized approximate loss.

    Args:
        x (torch.Tensor): The input tensor.
        min_priority (float): The minimum priority value.
        alpha (float): The alpha value.

    Returns:
        torch.Tensor: The calculated prioritized approximate loss.
    """
    return torch.where(
        x.abs() < min_priority,
        (min_priority**alpha) * 0.5 * x.pow(2),
        min_priority * x.abs().pow(1.0 + alpha) / (1.0 + alpha),
    ).mean()


def huber(x: torch.Tensor, min_priority: float) -> torch.Tensor:
    """
    Computes the Huber loss function.

    Args:
        x (torch.Tensor): The input tensor.
        min_priority (float): The minimum priority value.

    Returns:
        torch.Tensor: The computed Huber loss.

    """
    return torch.where(x < min_priority, 0.5 * x.pow(2), min_priority * x).mean()


def quantile_huber_loss_f(
    quantiles: torch.Tensor, samples: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the quantile Huber loss for a given set of quantiles and samples.

    Args:
        quantiles (torch.Tensor): A tensor of shape (batch_size, num_nets, num_quantiles) representing the quantiles.
        samples (torch.Tensor): A tensor of shape (batch_size, num_samples) representing the samples.

    Returns:
        torch.Tensor: The quantile Huber loss.

    """
    pairwise_delta = (
        samples[:, None, None, :] - quantiles[:, :, :, None]
    )  # batch x nets x quantiles x samples
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(
        abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta**2 * 0.5
    )

    n_quantiles = quantiles.shape[2]

    tau = (
        torch.arange(n_quantiles, device=pairwise_delta.get_device()).float()
        / n_quantiles
        + 1 / 2 / n_quantiles
    )
    loss = (
        torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss
    ).mean()
    return loss


def flatten(w: int, k: int = 3, s: int = 1, p: int = 0, m: bool = True) -> int:
    """
    Returns the right size of the flattened tensor after convolutional transformation
    :param w: width of image
    :param k: kernel size
    :param s: stride
    :param p: padding
    :param m: max pooling (bool)
    :return: proper shape and params: use x * x * previous_out_channels

    Example:
    r = flatten(*flatten(*flatten(w=100, k=3, s=1, p=0, m=True)))[0]
    self.fc1 = nn.Linear(r*r*128, 1024)
    """
    return int((np.floor((w - k + 2 * p) / s) + 1) if m else 1)
