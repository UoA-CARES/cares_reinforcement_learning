import math
import random
import sys
import torch
import torch.nn.functional as F
import torch.utils
from torch import optim
import numpy as np
from cares_reinforcement_learning.util.helpers import normalize_obs_deltas
from cares_reinforcement_learning.networks.World_Models.simple_dynamics import (Simple_Dynamics)
from cares_reinforcement_learning.networks.World_Models.simple_rewards import (Simple_Reward)


class IntegratedWorldModel:
    """
    A integrated world model aims to train the reward prediciton and next state
    prediciton together.

    :param (int) observation_size -- dimension of states
    :param (int) num_actions -- dimension of actions
    :param (int) hidden_size -- size of neurons in hidden layers.
    """
    def __init__(self, observation_size, num_actions, hidden_size):
        self.dyna_network = Simple_Dynamics(observation_size=observation_size,
                                            num_actions=num_actions,
                                            hidden_size=hidden_size)
        self.reward_network = Simple_Reward(observation_size=observation_size,
                                            num_actions=num_actions,
                                            hidden_size=hidden_size)
        self.reward_optimizer = optim.Adam(self.reward_network.parameters(),
                                           lr=0.001)
        self.dyna_optimizer = optim.Adam(self.dyna_network.parameters(),
                                         lr=0.001)
        self.all_optimizer = optim.Adam(list(self.reward_network.parameters())
                                        + list(self.dyna_network.parameters()),
                                        lr=0.001)
        self.statistics = {}

    def train_dynamics(self, states, actions, next_states):
        """
        Train the dynamics (next state prediciton) alone. Predicting the delta
        rather than the next state.

        :param (Tensor) states -- states input
        :param (Tensor) actions -- actions input
        :param (Tensor) next_states -- target label.
        """
        target = (next_states - states)
        delta_targets_normalized = normalize_obs_deltas(target, self.statistics)
        _, n_mean, n_var = self.dyna_network.forward(states, actions)
        model_loss = F.gaussian_nll_loss(input=n_mean,
                                         target=delta_targets_normalized,
                                         var=n_var).mean()

        self.dyna_optimizer.zero_grad()
        model_loss.backward()
        self.dyna_optimizer.step()

    def train_overall(self, states, actions, next_states, next_actions,
                      next_rewards):
        """
        Do one step preidiciton, train both network together. Add Two loss
        functions.

        :param (Tensor) states:
        :param (Tensor) actions:
        :param (Tensor) next_states:
        :param (Tensor) next_actions:
        :param (Tensor) next_rewards:
        """
        # Get the dynamics training losses first
        mean_deltas, normalized_mean, normalized_var = (
            self.dyna_network.forward(states, actions))
        # Always denormalized delta
        pred_next_state = mean_deltas + states
        target = (next_states - states)
        delta_targets_normalized = normalize_obs_deltas(target, self.statistics)
        model_loss = F.gaussian_nll_loss(input=normalized_mean,
                                         target=delta_targets_normalized,
                                         var=normalized_var).mean()
        pred_rewards = self.reward_network.forward(pred_next_state,
                                                   next_actions)
        all_loss = F.mse_loss(pred_rewards, next_rewards) + model_loss.mean()
        # Update
        self.all_optimizer.zero_grad()
        all_loss.backward()
        self.all_optimizer.step()


class Ensemble_World_Reward:
    """
    Ensemble the integrated dynamic reward models. It works like a group of
    experts. The predicted results can be used to estimate the uncertainty.

    :param (int) observation_size -- dimension of states
    :param (int) num_actions -- dimension of actions
    :param (int) num_models -- number of world models in this ensemble.
    :param (int) hidden_size -- size of neurons in hidden layers.
    """

    def __init__(self, observation_size, num_actions, num_models,
                 hidden_size=128):
        self.device = None
        self.num_models = num_models
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.models = [IntegratedWorldModel(observation_size=observation_size,
                                            num_actions=num_actions,
                                            hidden_size=hidden_size)
                       for _ in range(self.num_models)]
        self.statistics = {}

    def to(self, device):
        """
        A function that take all networks to a designate device.
        """
        self.device = device
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
        assert (obs.shape[1] + actions.shape[1] == self.observation_size +
                self.num_actions)
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
        return (prediction, all_predictions, predictions_norm_means,
                predictions_vars)

    def train_world(self, states, actions, rewards, next_states, next_actions,
                    next_rewards):
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
        assert (states.shape[1] + actions.shape[1] == self.num_actions +
                self.observation_size)
        # For each model, train with different data.
        mini_batch_size = int(math.floor(states.shape[0] / self.num_models))
        for i in range(self.num_models):
            self.models[i].train_dynamics(states[i*mini_batch_size:(i+1)*mini_batch_size],
                                          actions[i*mini_batch_size:(i+1)*mini_batch_size],
                                          next_states[i*mini_batch_size:(i+1)*mini_batch_size])
            self.models[i].train_overall(states[i*mini_batch_size:(i+1)*mini_batch_size],
                                         actions[i*mini_batch_size:(i+1)*mini_batch_size],
                                         next_states[i*mini_batch_size:(i+1)*mini_batch_size],
                                         next_actions[i*mini_batch_size:(i+1)*mini_batch_size],
                                         next_rewards[i*mini_batch_size:(i+1)*mini_batch_size])
