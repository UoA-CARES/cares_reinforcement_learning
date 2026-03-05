"""
Functional utilities for neural networks.

This module contains standalone tensor operations and network utilities
that don't maintain state, similar to torch.nn.functional.
"""

from contextlib import contextmanager

import torch


@contextmanager
def evaluating(model):
    """
    Context manager for temporarily setting a model to eval mode.

    Automatically restores the model to train mode after the context exits,
    even if an exception occurs. Useful for performing evaluation or inference
    within training loops without affecting the training state.

    Args:
        model (torch.nn.Module): The model to temporarily set to eval mode.

    Yields:
        torch.nn.Module: The same model in eval mode.

    Example:
        >>> with evaluating(model):
        ...     output = model(input)  # model is in eval mode
        ... # model is back in train mode
    """
    try:
        model.eval()
        yield model
    finally:
        model.train()


def avg_l1_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize a tensor by its average L1 norm along the last dimension.

    Args:
        x (torch.Tensor): Input tensor to normalize.
        eps (float): Small constant to prevent division by zero. Default: 1e-8.

    Returns:
        torch.Tensor: Normalized tensor where each element is divided by the mean
                     absolute value along the last dimension.

    Example:
        >>> x = torch.randn(32, 256)
        >>> normalized = avg_l1_norm(x)
    """
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


def weight_init(module: torch.nn.Module) -> None:
    """
    Custom weight initialization for Conv2D and Linear layers.

    Uses delta-orthogonal initialization from https://arxiv.org/pdf/1806.05393.pdf

    Args:
        module (torch.nn.Module): The module to initialize.

    Returns:
        None

    Note:
        - Linear layers: Orthogonal weight initialization, zero bias.
        - Conv2d/ConvTranspose2d: Delta-orthogonal initialization with zero weights
          except for the center of the kernel, which is initialized orthogonally.

    Example:
        >>> network = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
        >>> network.apply(weight_init)
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
