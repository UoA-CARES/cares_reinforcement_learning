import math
import random
import sys
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils
from torch import optim
import numpy as np
from .ensemble_integrated import IntegratedWorldModel
from cares_reinforcement_learning.util.helpers import normalize_obs_deltas
from torch.autograd import Variable


class Discriminator(nn.Module):
    def __init__(self, observation_size):
        super().__init__()
        self.linear1 = nn.Linear(observation_size, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 1)

    def forward(self, state):
        """
        Forward
        :param state:
        :return:
        """
        x_1 = self.linear1(state)
        x_1 = F.leaky_relu(x_1, negative_slope=0.1, inplace=True)
        x_1 = self.linear2(x_1)
        x_1 = F.leaky_relu(x_1, negative_slope=0.1, inplace=True)
        x_1 = self.linear3(x_1)
        x_1 = F.sigmoid(x_1)
        return x_1


class Ensemble_World_Reward_GAN:
    """
    Ensemble the integrated dynamic reward models. It works like a group of
    experts. The predicted results can be used to estimate the uncertainty.

    :param (int) observation_size -- dimension of states
    :param (int) num_actions -- dimension of actions
    :param (int) num_models -- number of world models in this ensemble.
    :param (int) hidden_size -- size of neurons in hidden layers.
    """

    def __init__(self, observation_size, num_actions, num_models, hidden_size=128):
        self.device = None
        self.num_models = num_models
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.models = [
            IntegratedWorldModel(
                observation_size=observation_size,
                num_actions=num_actions,
                hidden_size=hidden_size,
            )
            for _ in range(self.num_models)
        ]
        self.statistics = {}
        self.discriminator = Discriminator(observation_size=observation_size)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002)

    def to(self, device):
        """
        A function that take all networks to a designate device.
        """
        self.device = device
        self.discriminator.to(device)
        for model in self.models:
            model.dyna_network.to(device)
            model.reward_network.to(device)

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
            model.dyna_network.statistics = statistics

    def pred_rewards(self, obs, actions):
        """
        Make a prediciton of rewards based on current state and actions. Take
        the mean of rewards as final for now.

        :param (Tensors) obs -- dimension of states
        :param (Tensors) actions -- dimension of actions

        :return (Tensors) reward -- predicted mean rewards.
        :return (List) rewards -- A list of predicted rewards. For STEVE use.
        """
        rewards = []
        for model in self.models:
            pred_rewards = model.reward_network.forward(obs, actions)
            rewards.append(pred_rewards)
        # Use average
        rewards = torch.stack(rewards)
        reward = torch.min(rewards, dim=0).values  # Pessimetic
        return reward, rewards

    def pred_next_states(self, obs, actions):
        """
        Predict the next state based on current state and action, using an
        ensemble of world models. The world model is probablisitic. It is
        trained with Gaussian NLL loss.

        :param (Tensors) obs -- dimension of states
        :param (Tensors) actions -- dimension of actions

        :return (Tensors) random picked next state predicitons
        :return (Tensors) all next state predicitons
        :return (Tensors) all normalized delta' means
        :return (Tensors) all normalized delta' vars
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
            print("Predicting all Nans")
            sys.exit()
        rand_ind = random.randint(0, len(not_nans) - 1)
        prediction = predictions_means[not_nans[rand_ind]]
        # next = current + delta
        prediction += obs
        all_predictions = torch.stack(means)
        for j in range(all_predictions.shape[0]):
            all_predictions[j] += obs
        return (prediction, all_predictions, predictions_norm_means, predictions_vars)

    def train_world(
        self, states, actions, rewards, next_states, next_actions, next_rewards
    ):
        """
        This function decides how to train both reward prediciton and dynamic
        prediction.

        :param (Tensors) input states:
        :param (Tensors) input actions:
        :param (Tensors) input rewards:
        :param (Tensors) input next_states:
        :param (Tensors) input next_actions:
        :param (Tensors) input next_rewards:
        """
        assert len(states.shape) >= 2
        assert len(actions.shape) == 2
        assert len(rewards.shape) == 2
        assert (
            states.shape[1] + actions.shape[1]
            == self.num_actions + self.observation_size
        )
        # For each model, train with different data.
        mini_batch_size = int(math.floor(states.shape[0] / self.num_models))

        for i in range(self.num_models):
            sub_states = states[i*mini_batch_size:(i+1)*mini_batch_size]
            sub_actions = actions[i*mini_batch_size:(i+1)*mini_batch_size]
            sub_next_states = next_states[i*mini_batch_size:(i+1)*mini_batch_size]

            target = sub_next_states - sub_states
            delta_targets_normalized = normalize_obs_deltas(target,self.statistics)
            # Get the world model error.
            delta_state, n_mean, n_var = self.models[i].dyna_network.forward(
                sub_states, sub_actions
            )
            gen_states = delta_state + sub_states
            model_loss = F.gaussian_nll_loss(
                input=n_mean, target=delta_targets_normalized, var=n_var
            ).mean()
            self.models[i].dyna_optimizer.zero_grad()
            model_loss.backward()
            self.models[i].dyna_optimizer.step()

            adv_loss = torch.nn.BCELoss()
            valid = Variable(
                torch.FloatTensor(sub_states.size(0), 1).fill_(1.0),
                requires_grad=False,
            ).to(self.device)
            fake = Variable(
                torch.FloatTensor(sub_states.size(0), 1).fill_(0.0),
                requires_grad=False,
            ).to(self.device)

            # Train Discriminator
            self.optimizer_D.zero_grad()
            real_loss = adv_loss(self.discriminator(sub_next_states), valid)
            fake_loss = adv_loss(self.discriminator(gen_states.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            self.optimizer_D.step()

            self.models[i].train_overall(
                states[i * mini_batch_size : (i + 1) * mini_batch_size],
                actions[i * mini_batch_size : (i + 1) * mini_batch_size],
                next_states[i * mini_batch_size : (i + 1) * mini_batch_size],
                next_actions[i * mini_batch_size : (i + 1) * mini_batch_size],
                next_rewards[i * mini_batch_size : (i + 1) * mini_batch_size],
            )
