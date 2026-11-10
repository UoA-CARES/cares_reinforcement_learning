import torch


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
    use_huber_offset: bool = True,
) -> torch.Tensor:
    """
    Computes the Huber loss function.

    Args:
        x (torch.Tensor): The input tensor.
        kappa (float): The threshold value for huber calculation
        use_mean_reduction (bool): If True, reduces the loss by taking the mean. If False, returns the loss without reduction.
        use_huber_offset (bool): If True, applies quadratic smoothing to the Huber loss. If False, applies linear smoothing.

    Returns:
        torch.Tensor: The computed Huber loss.

    """

    # Smoothing factor for quadratic smoothing
    smoothing_factor = 0.0  # linear smoothing
    if use_huber_offset:
        smoothing_factor = 0.5  # quadratic smoothing

    element_wise_loss = torch.where(
        sample.abs() <= kappa,
        0.5 * sample.pow(2),
        kappa * (sample.abs() - smoothing_factor * kappa),
    )

    return element_wise_loss.mean() if use_mean_reduction else element_wise_loss
