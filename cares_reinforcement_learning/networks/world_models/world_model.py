import logging
import torch
import numpy as np
from cares_reinforcement_learning.networks.world_models.simple import (
    Probabilistic_SAS_Reward,
    Probabilistic_NS_Reward,
)
from cares_reinforcement_learning.networks.world_models.simple import (
    Simple_SAS_Reward,
    Simple_NS_Reward,
)
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
        num_rwd_model: int = 5,
    ):
        logging.info(f"Num of Reward models: {num_rwd_model}")
        if hidden_size is None:
            hidden_size = [128, 128]
        self.sas = None
        self.prob_rwd = None
        self.statistics = {}
        self.device = device
        self.sas = sas
        self.prob_rwd = prob_rwd
        self.statistics = {}
        self.counter = 0
        self.num_rwd_model = num_rwd_model

        self.rwd_models = []
        self.rwd_model_optimizers = []
        for i in range(self.num_rwd_model):
            if prob_rwd:
                if sas:
                    reward_network = Probabilistic_SAS_Reward(
                        observation_size=observation_size,
                        num_actions=num_actions,
                        hidden_size=hidden_size,
                        normalize=False,
                    )
                else:
                    reward_network = Probabilistic_NS_Reward(
                        observation_size=observation_size,
                        num_actions=num_actions,
                        hidden_size=hidden_size,
                        normalize=False,
                    )
            else:
                if sas:
                    reward_network = Simple_SAS_Reward(
                        observation_size=observation_size,
                        num_actions=num_actions,
                        hidden_size=hidden_size,
                        normalize=False,
                    )
                else:
                    reward_network = Simple_NS_Reward(
                        observation_size=observation_size,
                        num_actions=num_actions,
                        hidden_size=hidden_size,
                        normalize=False,
                    )
            reward_network.to(self.device)
            self.rwd_models.append(reward_network)
            reward_optimizer = optim.Adam(reward_network.parameters(), lr=l_r)
            self.rwd_model_optimizers.append(reward_optimizer)

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
        return (
            torch.zeros(observation.shape),
            torch.zeros(observation.shape),
            torch.zeros(observation.shape),
        )

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
        indice = self.counter % self.num_rwd_model
        self.rwd_model_optimizers[indice].zero_grad()
        if self.prob_rwd:
            if self.sas:
                rwd_mean, rwd_var = self.rwd_models[indice](
                    states, actions, next_states
                )
            else:
                rwd_mean, rwd_var = self.rwd_models[indice](next_states)
            reward_loss = F.gaussian_nll_loss(
                input=rwd_mean, target=rewards, var=rwd_var
            )
        else:
            if self.sas:
                rwd_mean = self.rwd_models[indice](states, actions, next_states)
            else:
                rwd_mean = self.rwd_models[indice](next_states)
            reward_loss = F.mse_loss(rwd_mean, rewards)
        reward_loss.backward()
        self.rwd_model_optimizers[indice].step()
        self.counter += 1

    def pred_rewards(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        next_observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict reward based on SAS
        :param observation:
        :param action:
        :param next_observation:
        :return: Predicted rewards, Means of rewards, Variances of rewards
        """
        preds = []
        preds_vars = []
        for i in range(self.num_rwd_model):
            if self.prob_rwd:
                if self.sas:
                    pred_rewards, rwd_var = self.rwd_models[i](
                        observation, action, next_observation
                    )
                else:
                    pred_rewards, rwd_var = self.rwd_models[i](next_observation)
            else:
                if self.sas:
                    pred_rewards = self.rwd_models[i](
                        observation, action, next_observation
                    )
                else:
                    pred_rewards = self.rwd_models[i](next_observation)
                rwd_var = None
            preds.append(pred_rewards)
            preds_vars.append(rwd_var)
        preds = torch.stack(preds)
        total_unc = 0.0
        if self.num_rwd_model > 1:
            epistemic_uncert = torch.var(preds, dim=0) ** 0.5
            aleatoric_uncert = torch.zeros(epistemic_uncert.shape)
            if rwd_var is None:
                rwd_var = torch.zeros(preds.shape)
            else:
                rwd_var = torch.stack(preds_vars)
                aleatoric_uncert = torch.mean(rwd_var**2, dim=0) ** 0.5
            total_unc = (aleatoric_uncert**2 + epistemic_uncert**2) ** 0.5

        if preds.shape[0] > 1:
            preds = torch.mean(preds, dim=0)
        else:
            preds = preds[0]

        return preds, total_unc

    def pred_all_rewards(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        next_observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict reward based on SAS
        :param observation:
        :param action:
        :param next_observation:
        :return: Predicted rewards, Means of rewards, Variances of rewards
        """
        preds = []
        preds_vars = []
        for j in range(next_observation.shape[0]):
            for i in range(self.num_rwd_model):
                if self.prob_rwd:
                    if self.sas:
                        pred_rewards, rwd_var = self.rwd_models[i](
                            observation, action, next_observation[j]
                        )
                    else:
                        pred_rewards, rwd_var = self.rwd_models[i](next_observation[j])
                else:
                    if self.sas:
                        pred_rewards = self.rwd_models[i](
                            observation, action, next_observation[j]
                        )
                    else:
                        pred_rewards = self.rwd_models[i](next_observation[j])
                    rwd_var = None
                preds.append(pred_rewards)
                preds_vars.append(rwd_var)
        preds = torch.stack(preds)
        if rwd_var is None:
            preds_vars = torch.zeros(preds.shape)
        else:
            preds_vars = torch.stack(preds_vars)

        return preds, preds_vars

    def estimate_uncertainty(
        self,
        observation: torch.Tensor,
        actions: torch.Tensor,
        train_reward: bool,
    ) -> tuple[float, float, torch.Tensor]:
        """
        Estimate next state uncertainty and reward uncertainty.

        :param observation:
        :param actions:
        :return: Dynamic Uncertainty, Reward Uncertainty
        """
        logging.info("Estimating Uncertainty Not Implemented")
        return 0.0, 0.0, None

    def train_together(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
    ):
        logging.info("Train Together Not Implemented")
