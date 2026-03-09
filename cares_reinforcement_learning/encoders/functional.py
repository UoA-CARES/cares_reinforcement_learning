"""
Functional utilities for encoders.

This module contains utilities specific to encoder architectures,
particularly for computing output dimensions of convolutional layers.
"""

import numpy as np


def flatten(w: int, k: int = 3, s: int = 1, p: int = 0, m: bool = True) -> int:
    """
    Calculate the output size of a convolutional layer after transformation.

    This function computes the spatial dimension of the output tensor after
    applying a convolutional layer with specified parameters.

    Args:
        w (int): Width (or height) of the input image.
        k (int): Kernel size. Default: 3.
        s (int): Stride. Default: 1.
        p (int): Padding. Default: 0.
        m (bool): Whether max pooling is applied (affects calculation). Default: True.

    Returns:
        int: The output dimension after the convolutional transformation.

    Example:
        Calculate output size after 3 conv layers with 100x100 input:
        >>> r = flatten(flatten(flatten(w=100, k=3, s=1, p=0, m=True)))
        >>> # Use for determining Linear layer input size:
        >>> # self.fc1 = nn.Linear(r * r * num_channels, 1024)

    Note:
        The formula used is: floor((w - k + 2*p) / s) + 1
        Returns 1 if m=False (typically used for final pooling layers).
    """
    return int((np.floor((w - k + 2 * p) / s) + 1) if m else 1)
