import random
from contextlib import contextmanager

import numpy as np
import torch


class EpsilonScheduler:
    def __init__(self, start_epsilon: float, end_epsilon: float, decay_steps: int):
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_steps = decay_steps
        self.epsilon = start_epsilon

    def get_epsilon(self, step: int) -> float:
        if step < self.decay_steps:
            self.epsilon = self.start_epsilon - (
                self.start_epsilon - self.end_epsilon
            ) * (step / self.decay_steps)
        else:
            self.epsilon = self.end_epsilon
        return self.epsilon


def get_device() -> torch.device:
    device = torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    return device


@contextmanager
def evaluating(model):
    """Context manager for temporarily setting a model to eval mode."""
    try:
        model.eval()
        yield model
    finally:
        model.train()


def image_state_dict_to_tensor(
    state: dict[str, np.ndarray], device: torch.device
) -> dict[str, torch.Tensor]:
    vector_tensor = torch.FloatTensor(state["vector"]).to(device)
    vector_tensor = vector_tensor.unsqueeze(0)

    image_tensor = torch.FloatTensor(state["image"]).to(device)
    image_tensor = image_tensor.unsqueeze(0)

    # Normalise states - image portion
    # This because the states are [0-255] and the predictions are [0-1]
    image_tensor = image_tensor / 255

    return {"image": image_tensor, "vector": vector_tensor}


def image_states_dict_to_tensor(
    states: list[dict[str, np.ndarray]], device: torch.device
) -> dict[str, torch.Tensor]:
    states_images = [state["image"] for state in states]
    states_vector = [state["vector"] for state in states]

    # Convert into tensors - torch.fromy_numpy saves copying the image reducing memory overhead
    states_images_tensor = (
        torch.from_numpy(np.asarray(states_images)).float().to(device)
    )
    states_vector_tensor = (
        torch.from_numpy(np.asarray(states_vector)).float().to(device)
    )

    # Normalise states and next_states - image portion
    # This because the states are [0-255] and the predictions are [0-1]
    states_images_tensor = states_images_tensor / 255

    return {"image": states_images_tensor, "vector": states_vector_tensor}


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
        net (torch.nn.Module): The neural network whose parameters which will be used to update the target network.
        target_net (torch.nn.Module): The target neural network whose parameters will be blended with the `net` parameters.
        tau (float): The blending factor. The updated parameters will be a weighted average of the `net` parameters and the `target_net` parameters.

    Returns:
        None
    """
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    # Hard update the statistics of the target network - specifically for BacthNorm layers
    for param, target_param in zip(net.buffers(), target_net.buffers()):
        target_param.data.copy_(param.data)


def hard_update_params(net, target_net):
    """
    Hard updates the parameters of a target neural network by directly copying the parameters from the source network.

    Args:
        net (torch.nn.Module): The neural network whose parameters will be copied to the target network.
        target_net (torch.nn.Module): The target neural network whose parameters will be replaced.

    Returns:
        None
    """
    soft_update_params(net, target_net, 1.0)


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
        if module.bias is not None:
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
    sample: torch.Tensor, min_priority: float, alpha: float
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
        sample.abs() < min_priority,
        (min_priority**alpha) * 0.5 * sample.pow(2),
        min_priority * sample.abs().pow(1.0 + alpha) / (1.0 + alpha),
    ).mean()


def calculate_huber_loss(
    sample: torch.Tensor,
    kappa: float,
    use_mean_reduction: bool = True,
    use_quadratic_smoothing: bool = True,
) -> torch.Tensor:
    """
    Computes the Huber loss function.

    Args:
        x (torch.Tensor): The input tensor.
        kappa (float): The threshold value for huber calculation
        use_mean_reduction (bool): If True, reduces the loss by taking the mean. If False, returns the loss without reduction.
        use_quadratic_smoothing (bool): If True, applies quadratic smoothing to the Huber loss. If False, applies linear smoothing.

    Returns:
        torch.Tensor: The computed Huber loss.

    """

    # Smoothing factor for quadratic smoothing
    smoothing_factor = 0.0  # linear smoothing
    if use_quadratic_smoothing:
        smoothing_factor = 0.5  # quadratic smoothing

    element_wise_loss = torch.where(
        sample.abs() <= kappa,
        0.5 * sample.pow(2),
        kappa * (sample.abs() - smoothing_factor * kappa),
    )

    return element_wise_loss.mean() if use_mean_reduction else element_wise_loss


def calculate_quantile_huber_loss(
    quantiles: torch.Tensor,
    target_values: torch.Tensor,
    quantile_taus: torch.Tensor,
    kappa: float = 1.0,
    use_pairwise_loss: bool = True,
    use_mean_reduction: bool = True,
    use_quadratic_smoothing: bool = True,
) -> torch.Tensor:
    """
    Calculates the quantile Huber loss for a given set of quantiles and target_values.

    Args:
        quantiles (torch.Tensor): A tensor of shape (batch_size, num_critics, num_quantiles) representing the quantiles.
        target_values (torch.Tensor): A tensor of shape (batch_size, num_samples) representing the samples.
        quantile_taus (torch.Tensor): A tensor of shape (num_quantiles) representing the quantile levels.
        kappa (float): The threshold value for Huber calculation.
        use_pairwise_loss (bool): If True, uses pairwise delta (TQC). If False, uses direct element-wise loss (QR-DQN).
        use_mean_reduction (bool): If True, reduces the loss by taking the mean. If False, returns the loss without reduction.
        use_quadratic_smoothing (bool): If True, applies quadratic smoothing to the Huber loss. If False, applies linear smoothing.

    Returns:
        torch.Tensor: The quantile Huber loss.

    """

    # batch x nets x quantiles x samples
    if use_pairwise_loss:
        # TQC-style: Compute pairwise differences (batch x nets x quantiles x samples)
        pairwise_delta = target_values[:, None, None, :] - quantiles[:, :, :, None]

        element_wise_huber_loss = calculate_huber_loss(
            pairwise_delta,
            kappa=kappa,
            use_mean_reduction=False,
            use_quadratic_smoothing=use_quadratic_smoothing,
        )

        element_wise_loss = (
            torch.abs(quantile_taus[None, None, :, None] - (pairwise_delta < 0).float())
            * element_wise_huber_loss
            / kappa
        )
    else:
        # QR-DQN-style: Compute element-wise TD error loss directly
        td_errors = target_values.unsqueeze(1) - quantiles

        element_wise_huber_loss = calculate_huber_loss(
            td_errors,
            kappa=kappa,
            use_mean_reduction=False,
            use_quadratic_smoothing=use_quadratic_smoothing,
        )

        element_wise_loss = (
            torch.abs(quantile_taus - (td_errors.detach() < 0).float())
            * element_wise_huber_loss
            / kappa
        )

    return element_wise_loss.mean() if use_mean_reduction else element_wise_loss


# TODO rename this function to something more descriptive
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


def compute_discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    """
    Compute discounted returns G_t from a list of rewards.
    Args:
        rewards (list or np.ndarray): Rewards [r_0, r_1, ..., r_T]
        gamma (float): Discount factor (0 <= gamma <= 1)
    Returns:
        list: Discounted returns [G_0, G_1, ..., G_T]
    """
    returns: list[float] = [0.0] * len(rewards)
    G = 0.0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns


def avg_l1_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)
