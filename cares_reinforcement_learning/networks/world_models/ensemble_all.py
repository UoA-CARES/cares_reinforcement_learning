import logging
import math
import random
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils
from torch import optim

from cares_reinforcement_learning.networks.world_models.probabilistic_dynamics import (
    ProbabilisticDynamics,
)
from cares_reinforcement_learning.networks.world_models.probabilistic_sas_reward import (
    Probabilistic_SAS_Reward,
)
from cares_reinforcement_learning.networks.world_models.simple_sas_done import (
    SASDone,
)
from cares_reinforcement_learning.util.helpers import normalize_observation_delta


class EnsembleWorldRewardDone:
    """
    This class consist of an ensemble of all components for critic update.
    Q_label = REWARD + gamma * (1 - DONES) * Q(NEXT_STATES).

    """

    def __init__(
            self,
            observation_size: int,
            num_actions: int,
            num_world_models: int,
            num_reward_models: int,
            lr: float,
            device: str,
            hidden_size: int = 128,
    ):
        self.num_reward_models = num_reward_models
        self.num_world_models = num_world_models

        self.observation_size = observation_size
        self.num_actions = num_actions
        self.device = device

        self.world_models = [ProbabilisticDynamics(observation_size=observation_size, num_actions=num_actions,
                                                   hidden_size=hidden_size) for _ in range(self.num_world_models)]
        self.reward_models = [Probabilistic_SAS_Reward(observation_size=observation_size, num_actions=num_actions,
                                                       hidden_size=hidden_size) for _ in range(self.num_reward_models)]
        self.world_optimizers = [optim.Adam(self.world_models[i].parameters(), lr=lr) for i in
                                 range(self.num_world_models)]
        self.reward_optimizers = [optim.Adam(self.reward_models[i].parameters(), lr=lr) for i in
                                  range(self.num_reward_models)]

        # Bring all reward prediction and dynamic rediction networks to device.
        for reward_model in self.world_models:
            reward_model.to(self.device)
        for world_model in self.world_models:
            world_model.to(self.device)

        self.done_model = SASDone(observation_size=observation_size, num_actions=num_actions,
                                  hidden_size=hidden_size)
        self.done_optimizers = optim.Adam(self.done_model.parameters(), lr=lr)
        self.done_model.to(self.device)
        self.statistics = {}

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
        for model in self.world_models:
            model.statistics = statistics

    def pred_multiple_rewards(self, observation: torch.Tensor, action: torch.Tensor, next_observation: torch.Tensor):
        """
        predict reward based on current observation and action and next state
        """
        assert len(next_observation.shape) == 3
        pred_reward_means = []
        pred_reward_vars = []
        # 5
        for j in range(next_observation.shape[0]):
            next_obs = next_observation[j]
            # 5
            for i in range(self.num_reward_models):
                pred_reward, reward_var = self.reward_models[i].forward(observation, action, next_obs)
                pred_reward_means.append(pred_reward)
                pred_reward_vars.append(reward_var)
        pred_reward_means = torch.stack(pred_reward_means)
        pred_reward_vars = torch.stack(pred_reward_vars)
        return pred_reward_means, pred_reward_vars

    def pred_rewards(self, observation: torch.Tensor,
                     action: torch.Tensor, next_observation: torch.Tensor):
        """
        predict reward based on current observation and action and next state
        """
        pred_reward_means = []
        pred_reward_vars = []
        for i in range(self.num_reward_models):
            pred_reward, reward_var = self.reward_models[i].forward(observation, action, next_observation)
            pred_reward_means.append(pred_reward)
            pred_reward_vars.append(reward_var)
        pred_reward_means = torch.stack(pred_reward_means)
        pred_reward_vars = torch.stack(pred_reward_vars)
        pred_rewards = torch.mean(pred_reward_means, dim=0)

        return pred_rewards, pred_reward_means, pred_reward_vars

    def pred_next_states(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict the next state based on the current state and action.

        The output is
        Args:
            observation:
            actions:

        Returns:
            prediction: Single prediction, probably mean.
            all_predictions: all means from different model.
            predictions_norm_means: normalized means.
            predictions_vars: normalized vars.
        """
        assert (
                observation.shape[1] + actions.shape[1]
                == self.observation_size + self.num_actions
        )
        means = []
        norm_means = []
        norm_vars = []
        # Iterate over the neural networks and get the predictions
        for model in self.world_models:
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
        for i in range(self.num_world_models):
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
        """
        Train the world with S, A, SN. Different sub-batch.

        Args:
            states:
            actions:
            next_states:
        """
        assert len(states.shape) >= 2
        assert len(actions.shape) == 2
        assert (
                states.shape[1] + actions.shape[1]
                == self.num_actions + self.observation_size
        )
        # For each model, train with different data.
        mini_batch_size = int(math.floor(states.shape[0] / self.num_world_models))

        for i in range(self.num_world_models):
            sub_states = states[i * mini_batch_size: (i + 1) * mini_batch_size]
            sub_actions = actions[i * mini_batch_size: (i + 1) * mini_batch_size]
            sub_next_states = next_states[i * mini_batch_size: (i + 1) * mini_batch_size]
            sub_target = sub_next_states - sub_states
            delta_targets_normalized = normalize_observation_delta(sub_target, self.statistics)
            _, n_mean, n_var = self.world_models[i].forward(sub_states, sub_actions)
            model_loss = F.gaussian_nll_loss(input=n_mean, target=delta_targets_normalized, var=n_var).mean()
            self.world_optimizers[i].zero_grad()
            model_loss.backward()
            self.world_optimizers[i].step()

    def train_reward(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
            rewards: torch.Tensor,
    ) -> None:
        """
        Train the reward with S, A, SN to eliminate difference between them.

        Args:
            states:
            actions:
            next_states:
            rewards:
        """
        mini_batch_size = int(math.floor(states.shape[0] / self.num_reward_models))
        for i in range(self.num_reward_models):
            sub_states = states[i * mini_batch_size: (i + 1) * mini_batch_size]
            sub_actions = actions[i * mini_batch_size: (i + 1) * mini_batch_size]
            sub_next_states = next_states[i * mini_batch_size: (i + 1) * mini_batch_size]
            sub_rewards = rewards[i * mini_batch_size: (i + 1) * mini_batch_size]
            self.reward_optimizers[i].zero_grad()
            rwd_mean, rwd_var = self.reward_models[i].forward(sub_states, sub_actions, sub_next_states)
            # reward_loss = F.mse_loss(rwd_mean, sub_rewards)
            reward_loss = F.gaussian_nll_loss(input=rwd_mean, target=sub_rewards, var=rwd_var).mean()
            reward_loss.backward()
            self.reward_optimizers[i].step()

    # def train_done(
    #         self,
    #         states: torch.Tensor,
    #         actions: torch.Tensor,
    #         dones: torch.Tensor,
    # ) -> None:
    #     self.reward_optimizer.zero_grad()
    #     prob_dones = self.reward_network.forward(states, actions)
    #     reward_loss = F.binary_cross_entropy(prob_dones, dones)
    #     reward_loss.backward()
    #     self.reward_optimizer.step()
