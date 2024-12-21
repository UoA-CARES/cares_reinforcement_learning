import logging
import torch
import numpy as np
from cares_reinforcement_learning.networks.world_models.simple import Probabilistic_SAS_Reward, Probabilistic_NS_Reward
from cares_reinforcement_learning.networks.world_models.simple import Simple_SAS_Reward, Simple_NS_Reward
import torch.nn.functional as F
import torch.utils
from torch import optim


class World_Model:
    """
    World Model
    """

    def __init__(
            self,
            observation_size: int,
            num_actions: int,
            l_r: float,
            device: str,
            hidden_size=None,
            sas: bool = True,
            prob_rwd: bool = False,
    ):
        if hidden_size is None:
            hidden_size = [128, 128]
        self.sas = None
        self.prob_rwd = None
        self.statistics = {}
        self.device = device
        self.sas = sas
        self.prob_rwd = prob_rwd
        self.statistics = {}
        if prob_rwd:
            if sas:
                self.reward_network = Probabilistic_SAS_Reward(
                    observation_size=observation_size,
                    num_actions=num_actions,
                    hidden_size=hidden_size,
                    normalize=False
                )
            else:
                self.reward_network = Probabilistic_NS_Reward(
                    observation_size=observation_size,
                    num_actions=num_actions,
                    hidden_size=hidden_size,
                    normalize=False
                )
        else:
            if sas:
                self.reward_network = Simple_SAS_Reward(
                    observation_size=observation_size,
                    num_actions=num_actions,
                    hidden_size=hidden_size,
                    normalize=False
                )
            else:
                self.reward_network = Simple_NS_Reward(
                    observation_size=observation_size,
                    num_actions=num_actions,
                    hidden_size=hidden_size,
                    normalize=False
                )
        self.reward_network.to(self.device)
        self.reward_optimizer = optim.Adam(self.reward_network.parameters(), lr=l_r)

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

    def train_world(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
    ) -> None:
        """
        Train the dynamic of world model.
        :param states:
        :param actions:
        :param next_states:
        """
        logging.info(" Train world Not Implemented")

    def pred_next_states(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make a prediction of next state.
        :param observation:
        :param actions:
        :return: Next_state Prediction, Next_state Means, Next_State Variance.
        """
        logging.info("Predict Next Not Implemented")
        return torch.zeros(observation.shape), torch.zeros(observation.shape), torch.zeros(observation.shape)

    def train_reward(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
            rewards: torch.Tensor,
    ) -> None:
        """
        Train the reward prediction with or without world model dynamics.

        :param states:
        :param actions:
        :param next_states:
        :param rewards:
        """
        self.reward_optimizer.zero_grad()
        if self.prob_rwd:
            if self.sas:
                rwd_mean, rwd_var = self.reward_network(states, actions, next_states)
            else:
                rwd_mean, rwd_var = self.reward_network(next_states, actions)
            reward_loss = F.gaussian_nll_loss(input=rwd_mean, target=rewards, var=rwd_var)
        else:
            if self.sas:
                rwd_mean = self.reward_network(states, actions, next_states)
            else:
                rwd_mean = self.reward_network(next_states, actions)
            reward_loss = F.mse_loss(rwd_mean, rewards)
        reward_loss.backward()
        self.reward_optimizer.step()

    def pred_rewards(self, observation: torch.Tensor, action: torch.Tensor, next_observation: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict reward based on SAS
        :param observation:
        :param action:
        :param next_observation:
        :return: Predicted rewards, Means of rewards, Variances of rewards
        """

        if self.prob_rwd:
            if self.sas:
                pred_rewards, rwd_var = self.reward_network(observation, action, next_observation)
            else:
                pred_rewards, rwd_var = self.reward_network(next_observation)
            return pred_rewards, rwd_var
        else:
            if self.sas:
                pred_rewards = self.reward_network(observation, action, next_observation)
            else:
                pred_rewards = self.reward_network(next_observation)
            return pred_rewards, None

    def estimate_uncertainty(
            self, observation: torch.Tensor, actions: torch.Tensor, train_reward:bool,
    ) -> tuple[float, float, torch.Tensor]:
        """
        Estimate next state uncertainty and reward uncertainty.

        :param observation:
        :param actions:
        :return: Dynamic Uncertainty, Reward Uncertainty
        """
        logging.info("Estimating Uncertainty Not Implemented")
        return 0.0, 0.0, None

    def train_together(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, ):
        logging.info("Train Together Not Implemented")
