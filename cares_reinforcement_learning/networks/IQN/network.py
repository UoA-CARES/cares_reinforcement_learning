import numpy as np
import torch
from torch import nn

from cares_reinforcement_learning.networks.common import MLP
from cares_reinforcement_learning.util.configurations import IQNConfig


class BaseNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_actions: int,
        quantiles: int,
        num_cosines: int,
        feature_network: nn.Module | nn.Sequential,
        cosine_network: nn.Module | nn.Sequential,
        quantile_network: nn.Module | nn.Sequential,
    ):
        super().__init__()

        self.input_size = input_size
        self.num_actions = num_actions

        self.quantiles = quantiles
        self.num_cosines = num_cosines

        self.feature_network = feature_network
        self.cosine_network = cosine_network
        self.quantile_network = quantile_network

    def calculate_state_embeddings(self, states: torch.Tensor) -> torch.Tensor:
        return self.feature_network(states)

    def _calculate_tau_embeddings(self, taus: torch.Tensor) -> torch.Tensor:
        batch_size = taus.shape[0]
        N = taus.shape[1]

        # Calculate i * \pi (i=1,...,N).
        i_pi = np.pi * torch.arange(
            start=1, end=self.num_cosines + 1, dtype=taus.dtype, device=taus.device
        ).view(1, 1, self.num_cosines)

        # Calculate cos(i * \pi * \tau).
        cosines = torch.cos(taus.view(batch_size, N, 1) * i_pi).view(
            batch_size * N, self.num_cosines
        )

        # Calculate embeddings of taus.
        tau_embeddings = self.net(cosines).view(batch_size, N, embedding_dim)

        return tau_embeddings

    def _calculate_quantile_embeddings(
        self, state_beddings: torch.Tensor, tau_embeddings: torch.Tensor
    ) -> torch.Tensor:
        # NOTE: Because variable taus correspond to either \tau or \hat \tau
        # in the paper, N isn't neccesarily the same as fqf.N.
        batch_size = state_embeddings.shape[0]
        N = tau_embeddings.shape[1]

        # Reshape into (batch_size, 1, embedding_dim).
        state_embeddings = state_embeddings.view(batch_size, 1, embedding_dim)

        # Calculate embeddings of states and taus.
        embeddings = (state_embeddings * tau_embeddings).view(
            batch_size * N, embedding_dim
        )

        # Calculate quantile values.
        quantiles = self.quantile_network(embeddings)

        return quantiles.view(batch_size, N, self.num_actions)

    def calculate_quantiles(
        self, state: torch.Tensor, taus: torch.Tensor
    ) -> torch.Tensor:
        state_embeddings = self.calculate_state_embeddings(state)

        tau_embeddings = self._calculate_tau_embeddings(taus)

        return self._calculate_quantile_embeddings(state_embeddings, tau_embeddings)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state_embeddings = self.calculate_state_embeddings(state)
        batch_size = state_embeddings.shape[0]

        taus = torch.rand(
            batch_size,
            self.K,
            dtype=state_embeddings.dtype,
            device=state_embeddings.device,
        )

        quantiles = self.calculate_quantiles(state, taus)

        return quantiles.mean(dim=-1)


# This is the default base network for DQN for reference and testing of default network configurations
class DefaultNetwork(BaseNetwork):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
    ):
        quantiles = 64
        cosine_basis_functions = 64
        hidden_sizes = [256]

        feature_network = nn.Sequential(
            nn.Linear(observation_size, hidden_sizes[0]),
            nn.ReLU(),
        )

        cosine_network = nn.Sequential(
            nn.Linear(cosine_basis_functions, hidden_sizes[0]),
            nn.ReLU(),
        )

        quantile_network = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], num_actions),
        )

        super().__init__(
            input_size=observation_size,
            num_actions=num_actions,
            quantiles=quantiles,
            num_cosines=cosine_basis_functions,
            feature_network=feature_network,
            cosine_network=cosine_network,
            quantile_network=quantile_network,
        )


class Network(BaseNetwork):
    def __init__(self, observation_size: int, num_actions: int, config: IQNConfig):

        network = MLP(
            input_size=observation_size,
            output_size=num_actions * config.quantiles,
            config=config.network_config,
        )
        super().__init__(
            input_size=observation_size,
            num_actions=num_actions,
            quantiles=config.quantiles,
            num_cosines=0,
            feature_network=network,
            cosine_network=network,
            quantile_network=network,
        )
