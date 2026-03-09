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
