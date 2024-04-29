import logging
import math
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils
from torch import optim

from cares_reinforcement_learning.networks.world_models.simple_dynamics import (
    SimpleDynamics,
)
from cares_reinforcement_learning.networks.world_models.simple_rewards import (
    SimpleReward,
)
from cares_reinforcement_learning.util.helpers import normalize_observation_delta


class EnsembleWorldAndOneReward:
    def __init__(
            self,
            observation_size: int,
            num_actions: int,
            num_models: int,
            lr: float,
            device: str,
            hidden_size: int = 128,
    ):
        self.num_models = num_models
        self.observation_size = observation_size
        self.num_actions = num_actions

        self.reward_network = SimpleReward(
            observation_size=observation_size,
            num_actions=num_actions,
            hidden_size=hidden_size,
        )
        self.reward_optimizer = optim.Adam(self.reward_network.parameters(), lr=lr)

        self.models = [
            SimpleDynamics(
                observation_size=observation_size,
                num_actions=num_actions,
                hidden_size=hidden_size,
            )
            for _ in range(self.num_models)
        ]

        self.optimizers = [optim.Adam(self.models[i].parameters(), lr=lr) for i in range(self.num_models)]

        self.statistics = {}

        # Bring all reward prediction and dynamic rediction networks to device.
        self.device = device
        self.reward_network.to(self.device)
        for model in self.models:
            model.to(device)

    def set_statistics(self, statistics: dict) -> None:
        """
        Update all statistics for normalization for all world models and the
        ensemble itself.

        :param (Dictionary) statistics:
        """
        for key, value in statistics.items():
            if isinstance(value, np.ndarray):
                statistics[key] = torch.FloatTensor(statistics[key]).to(self.device)

        self.statistics = statistics
        for model in self.models:
            model.statistics = statistics

    def pred_rewards(self, observation: torch.Tensor):
        pred_rewards = self.reward_network(observation)
        return pred_rewards

    def pred_next_states(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (
                observation.shape[1] + actions.shape[1]
                == self.observation_size + self.num_actions
        )
        means = []
        norm_means = []
        norm_vars = []
        # Iterate over the neural networks and get the predictions
        for model in self.models:
            # Predict delta
            mean, n_mean, n_var = model.forward(observation, actions)
            means.append(mean)
            norm_means.append(n_mean)
            norm_vars.append(n_var)
        # Normalized
        predictions_means = torch.stack(means)
        predictions_norm_means = torch.stack(norm_means)
        predictions_vars = torch.stack(norm_vars)
        # Get rid of the nans
        not_nans = []
        for i in range(self.num_models):
            if not torch.any(torch.isnan(predictions_means[i])):
                not_nans.append(i)
        if len(not_nans) == 0:
            logging.info("Predicting all Nans")
            sys.exit()
        # Random Take next state.
        rand_ind = random.randint(0, len(not_nans) - 1)
        prediction = predictions_means[not_nans[rand_ind]]
        # next = current + delta
        prediction += observation
        all_predictions = torch.stack(means)
        for j in range(all_predictions.shape[0]):
            all_predictions[j] += observation
        return prediction, all_predictions, predictions_norm_means, predictions_vars

    def train_world(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
    ) -> None:

        assert len(states.shape) >= 2
        assert len(actions.shape) == 2
        assert (
                states.shape[1] + actions.shape[1]
                == self.num_actions + self.observation_size
        )
        # For each model, train with different data.
        mini_batch_size = int(math.floor(states.shape[0] / self.num_models))

        for i in range(self.num_models):
            sub_states = states[i * mini_batch_size: (i + 1) * mini_batch_size]
            sub_actions = actions[i * mini_batch_size: (i + 1) * mini_batch_size]
            sub_next_states = next_states[i * mini_batch_size: (i + 1) * mini_batch_size]
            sub_target = sub_next_states - sub_states

            delta_targets_normalized = normalize_observation_delta(sub_target, self.statistics)
            _, n_mean, n_var = self.models[i].forward(sub_states, sub_actions)
            model_loss = F.gaussian_nll_loss(input=n_mean, target=delta_targets_normalized, var=n_var).mean()

            self.optimizers[i].zero_grad()
            model_loss.backward()
            self.optimizers[i].step()

    def train_reward(
            self,
            next_states: torch.Tensor,
            rewards: torch.Tensor,
    ) -> None:
        assert len(next_states.shape) >= 2
        # assert len(actions.shape) == 2
        # assert (
        #         next_states.shape[1] + actions.shape[1]
        #         == self.num_actions + self.observation_size
        # )
        self.reward_optimizer.zero_grad()
        rwd_mean = self.reward_network.forward(next_states)
        reward_loss = F.mse_loss(rwd_mean, rewards)
        reward_loss.backward()
        self.reward_optimizer.step()


