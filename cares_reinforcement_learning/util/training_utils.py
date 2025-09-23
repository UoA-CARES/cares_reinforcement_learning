"""
Training utilities for reinforcement learning algorithms.

Common functions used across different RL algorithms to reduce code duplication
while maintaining readability for students.
"""

from typing import Tuple

import numpy as np
import torch

from cares_reinforcement_learning.memory import MemoryBuffer


def sample_and_prepare_batch(
    memory: MemoryBuffer,
    batch_size: int,
    device: torch.device,
    use_per_buffer: int = 0,
    per_sampling_strategy: str = "uniform",
    per_weight_normalisation: str = "none",
    states_dtype: torch.dtype = torch.float32,
    action_dtype: torch.dtype = torch.float32,
    rewards_dtype: torch.dtype = torch.float32,
    next_states_dtype: torch.dtype = torch.float32,
    dones_dtype: torch.dtype = torch.long,
    weights_dtype: torch.dtype = torch.float32,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    np.ndarray,
]:
    """
    Sample a batch from memory and convert to tensors ready for training.

    This function handles both uniform and priority sampling, converts numpy arrays
    to PyTorch tensors, and reshapes them appropriately for training.

    Args:
        memory: The replay buffer to sample from
        batch_size: Number of experiences to sample
        device: PyTorch device to place tensors on
        use_per_buffer: Whether to use Prioritized Experience Replay
        per_sampling_strategy: Strategy for PER sampling
        per_weight_normalisation: Weight normalization strategy for PER
        states_dtype: Dtype for states tensor (default: torch.float32)
        action_dtype: Dtype for actions tensor (default: torch.float32)
        rewards_dtype: Dtype for rewards tensor (default: torch.float32)
        next_states_dtype: Dtype for next_states tensor (default: torch.float32)
        dones_dtype: Dtype for dones tensor (default: torch.long)
        weights_dtype: Dtype for weights tensor (default: torch.float32)

    Returns:
        Tuple of (states_tensor, actions_tensor, rewards_tensor, next_states_tensor,
                 dones_tensor, weights_tensor, indices)
    """

    # Sample from memory buffer
    if use_per_buffer:
        experiences = memory.sample_priority(
            batch_size,
            sampling_stratagy=per_sampling_strategy,
            weight_normalisation=per_weight_normalisation,
        )
        states, actions, rewards, next_states, dones, indices, weights = experiences
    else:
        experiences = memory.sample_uniform(batch_size)
        states, actions, rewards, next_states, dones, indices = experiences
        weights = [1.0] * batch_size

    batch_size = len(states)

    # Convert to PyTorch tensors with specified dtypes
    states_tensor = torch.tensor(np.asarray(states), dtype=states_dtype, device=device)
    actions_tensor = torch.tensor(
        np.asarray(actions), dtype=action_dtype, device=device
    )
    rewards_tensor = torch.tensor(
        np.asarray(rewards), dtype=rewards_dtype, device=device
    )
    next_states_tensor = torch.tensor(
        np.asarray(next_states), dtype=next_states_dtype, device=device
    )
    dones_tensor = torch.tensor(np.asarray(dones), dtype=dones_dtype, device=device)
    weights_tensor = torch.tensor(
        np.asarray(weights), dtype=weights_dtype, device=device
    )

    # Reshape tensors to proper dimensions
    rewards_tensor = rewards_tensor.reshape(batch_size, 1)
    dones_tensor = dones_tensor.reshape(batch_size, 1)
    weights_tensor = weights_tensor.reshape(batch_size, 1)

    return (
        states_tensor,
        actions_tensor,
        rewards_tensor,
        next_states_tensor,
        dones_tensor,
        weights_tensor,
        indices,
    )


def sample_and_prepare_consecutive_batch(
    memory: MemoryBuffer,
    batch_size: int,
    device: torch.device,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Sample consecutive experiences and prepare tensors (used by SDAR).

    Args:
        memory: The replay buffer to sample from
        batch_size: Number of consecutive experience pairs to sample
        device: PyTorch device to place tensors on

    Returns:
        Tuple of tensors ready for training
    """
    # Sample consecutive experiences for SDAR
    experiences = memory.sample_consecutive(batch_size)
    (
        _,
        prev_actions,
        _,
        _,
        _,
        states,
        actions,
        rewards,
        next_states,
        dones,
        _,
    ) = experiences

    batch_size = len(states)

    # Convert to PyTorch tensors
    states_tensor = torch.FloatTensor(np.asarray(states)).to(device)
    actions_tensor = torch.FloatTensor(np.asarray(actions)).to(device)
    rewards_tensor = torch.FloatTensor(np.asarray(rewards)).to(device)
    next_states_tensor = torch.FloatTensor(np.asarray(next_states)).to(device)
    dones_tensor = torch.LongTensor(np.asarray(dones)).to(device)
    prev_actions_tensor = torch.FloatTensor(np.asarray(prev_actions)).to(device)

    # Reshape tensors
    rewards_tensor = rewards_tensor.reshape(batch_size, 1)
    dones_tensor = dones_tensor.reshape(batch_size, 1)

    return (
        states_tensor,
        actions_tensor,
        rewards_tensor,
        next_states_tensor,
        dones_tensor,
        prev_actions_tensor,
        torch.FloatTensor([1.0] * batch_size)
        .reshape(batch_size, 1)
        .to(device),  # weights
    )
