"""
Original Paper:
"""

import torch

from cares_reinforcement_learning.algorithm.value import DQN
from cares_reinforcement_learning.networks.IQN.network import Network
from cares_reinforcement_learning.util import helpers as hlp
from cares_reinforcement_learning.util.configurations import IQNConfig


class IQN(DQN):
    def __init__(
        self,
        network: Network,
        config: IQNConfig,
        device: torch.device,
    ):
        super().__init__(network=network, config=config, device=device)

        # IQN
        self.num_quantiles = config.quantiles
        self.samples_per_quantile = config.samples_per_quantile
        self.embedding_dim = config.embedding_dim
        self.cosine_basis_functions = config.cosine_basis_functions
        self.kappa = config.kappa

    def _compute_loss(
        self,
        states_tensor: torch.Tensor,
        actions_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        next_states_tensor: torch.Tensor,
        dones_tensor: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:

        # Calculate features of states.
        state_embeddings = self.network.calculate_state_embeddings(states_tensor)

        # Sample fractions.
        taus = torch.rand(
            batch_size,
            self.num_quantiles,
            dtype=state_embeddings.dtype,
            device=state_embeddings.device,
        )

        # Predicted Q-value quantiles for current state
        current_quantile_values = self.network.calculate_quantiles(states_tensor, taus)

        # Gather Q-value quantiles of actions actually taken
        current_action_q_values = hlp.evaluate_quantile_at_action(
            current_quantile_values, actions_tensor
        )

        with torch.no_grad():
            if self.use_double_dqn:
                # Sample the noise of online network to decorrelate between
                # the action selection and the quantile calculation.
                next_state_q_values = self.network.calculate_quantiles(
                    next_states_tensor, taus
                ).mean(dim=-1)
            else:
                next_state_q_values = self.target_network.calculate_quantiles(
                    next_states_tensor
                ).mean(dim=-1)

        return None
