import random
from contextlib import contextmanager
from typing import overload

import cv2
import numpy as np
import torch


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


@overload
def denormalize(
    action: np.ndarray,
    max_action_value: np.ndarray,
    min_action_value: np.ndarray,
) -> np.ndarray: ...


@overload
def denormalize(
    action: list[np.ndarray],
    max_action_value: list[np.ndarray],
    min_action_value: list[np.ndarray],
) -> list[np.ndarray]: ...


def denormalize(
    action,
    max_action_value,
    min_action_value,
):
    """
    Denormalize the given action value within the specified range.

    Args:
        action (np.ndarray): The action value to be denormalized.
        max_action_value (np.ndarray): The maximum value of the action range.
        min_action_value (np.ndarray): The minimum value of the action range.

    Returns:
        np.ndarray: The denormalized action value.

    """

    def _denormalize(
        action: np.ndarray,
        max_action_value: np.ndarray,
        min_action_value: np.ndarray,
    ) -> np.ndarray:
        max_range_value: np.ndarray = max_action_value
        min_range_value: np.ndarray = min_action_value
        max_value_in: float = 1
        min_value_in: float = -1
        action_denorm: np.ndarray = (action - min_value_in) * (
            max_range_value - min_range_value
        ) / (max_value_in - min_value_in) + min_range_value

        action_denorm = np.clip(action_denorm, min_action_value, max_action_value)

        return action_denorm

    if isinstance(action, list):
        denormalized_actions: list[np.ndarray] = []
        for a, max_a, min_a in zip(action, max_action_value, min_action_value):
            denormalized_actions.append(_denormalize(a, max_a, min_a))
        return denormalized_actions

    return _denormalize(action, max_action_value, min_action_value)


@overload
def normalize(
    action: np.ndarray,
    max_action_value: np.ndarray,
    min_action_value: np.ndarray,
) -> np.ndarray: ...


@overload
def normalize(
    action: list[np.ndarray],
    max_action_value: list[np.ndarray],
    min_action_value: list[np.ndarray],
) -> list[np.ndarray]: ...


def normalize(
    action,
    max_action_value,
    min_action_value,
):
    """
    Normalize the given action value within the specified range.

    Args:
        action (np.ndarray): The action value to be normalized.
        max_action_value (np.ndarray): The maximum value of the action range.
        min_action_value (np.ndarray): The minimum value of the action range.

    Returns:
        np.ndarray: The normalized action value.

    """

    def _normalize(
        action: np.ndarray,
        max_action_value: np.ndarray,
        min_action_value: np.ndarray,
    ) -> np.ndarray:
        max_range_value: float = 1
        min_range_value: float = -1
        max_value_in: np.ndarray = max_action_value
        min_value_in: np.ndarray = min_action_value
        action_norm: np.ndarray = (action - min_value_in) * (
            max_range_value - min_range_value
        ) / (max_value_in - min_value_in) + min_range_value
        return action_norm

    if isinstance(action, list):
        normalized_actions: list[np.ndarray] = []
        for a, max_a, min_a in zip(action, max_action_value, min_action_value):
            normalized_actions.append(_normalize(a, max_a, min_a))
        return normalized_actions

    return _normalize(action, max_action_value, min_action_value)


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


def dump_tensor(x):
    return (
        x.dtype,
        x.device,
        tuple(x.shape),
        x.is_contiguous(),
        x.stride(),
        float(x.min()),
        float(x.max()),
        float(x.mean()),
        float(x.std()),
    )


def overlay_info(image: np.ndarray, **kwargs):
    """Overlay key-value information on an image."""
    output_image = image.copy()

    # Define text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    color = (0, 0, 255)
    thickness = 1

    # Define initial position for the text
    text_x, text_y = 10, 30

    # Get text height for proper spacing
    (_, h), baseline = cv2.getTextSize("A", font, font_scale, thickness)
    line_height = h + baseline

    # Iterate over key-value pairs and overlay text
    for i, (key, value) in enumerate(kwargs.items()):
        text = f"{key}: {value}"
        cv2.putText(
            output_image,
            text,
            (text_x, text_y + i * line_height),
            font,
            font_scale,
            color,
            thickness,
        )

    return output_image
