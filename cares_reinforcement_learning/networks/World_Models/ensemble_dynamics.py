import math
import random
import sys
import logging
import torch
import torch.utils
import numpy as np
from torch import optim
import torch.nn.functional as F
from cares_reinforcement_learning.util.helpers import normalize_obs_deltas
from cares_reinforcement_learning.networks.World_Models.simple_dynamics import (
    Simple_Dynamics,
)


class Ensemble_Dynamics:
    """
    Ensemble of dynamic models. It works like a group of
    experts. The predicted results can be used to estimate the uncertainty.

    :param (int) observation_size -- dimension of states
    :param (int) num_actions -- dimension of actions
    :param (int) num_models -- number of world models in this ensemble.
    :param (int) hidden_size -- size of neurons in hidden layers.
    """

    def __init__(
            self, observation_size, num_actions, num_models, hidden_size=128, lr=0.001
    ):
        self.device = None
        self.num_models = num_models
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.models = [
            Simple_Dynamics(
                observation_size=observation_size,
                num_actions=num_actions,
                hidden_size=hidden_size,
            )
            for _ in range(self.num_models)
        ]
        self.optimizers = [
            optim.Adam(self.models[i].parameters(), lr=lr)
            for i in range(self.num_models)
        ]
        self.statistics = {}

    def to(self, device):
        """
        A function that take all networks to a designate device.
        """
        self.device = device
        for model in self.models:
            model.dyna_network.to(device)

    def set_statistics(self, statistics):
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

    def pred_next_states(self, obs, actions):
        """
        Predict the next state based on current state and action, using an
        ensemble of world models. The world model is probablisitic. It is
        trained with Gaussian NLL loss.

        :param (Tensors) obs -- dimension of states
        :param (Tensors) actions -- dimension of actions

        :return (Tensors) random picked next state predicitons
        :return (Tensors) all next state predicitons
        :return (Tensors) all normalized delta' means for uncertainty
        :return (Tensors) all normalized delta' vars for uncertainty
        """
        assert (
            obs.shape[1] + actions.shape[1] == self.observation_size + self.num_actions
        )
        means = []
        norm_means = []
        norm_vars = []
        # Iterate over the neural networks and get the predictions
        for model in self.models:
            # Predict delta
            mean, n_mean, n_var = model.dyna_network.forward(obs, actions)
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
        rand_ind = random.randint(0, len(not_nans) - 1)
        prediction = predictions_means[not_nans[rand_ind]]
        # NOTE: Already Done: next = current + delta
        prediction += obs
        all_predictions = torch.stack(means)
        for j in range(all_predictions.shape[0]):
            all_predictions[j] += obs
        return prediction, all_predictions, predictions_norm_means, predictions_vars

    def train_world(self, states, actions, next_states):
        """
        This function decides how to train dynamic. Different models in an
        ensemble is trained with different data.

        :param (Tensors) input states:
        :param (Tensors) input actions:
        :param (Tensors) input next_states:
        """
        assert len(states.shape) >= 2
        assert len(actions.shape) == 2
        assert (
            states.shape[1] + actions.shape[1]
            == self.num_actions + self.observation_size
        )
        # For each model, train with different data.
        mini_batch_size = int(math.floor(states.shape[0] / self.num_models))
        for i in range(self.num_models):
            sub_states = states[i * mini_batch_size : (i + 1) * mini_batch_size]
            sub_actions = actions[i * mini_batch_size : (i + 1) * mini_batch_size]
            sub_n_states = next_states[i * mini_batch_size : (i + 1) * mini_batch_size]

            target = sub_n_states - sub_states
            delta_targets_normalized = normalize_obs_deltas(target, self.statistics)
            _, n_mean, n_var = self.models[i].forward(sub_states, sub_actions)

            model_loss = F.gaussian_nll_loss(
                input=n_mean, target=delta_targets_normalized, var=n_var
            ).mean()

            self.optimizers[i].zero_grad()
            model_loss.backward()
            self.optimizers[i].step()

