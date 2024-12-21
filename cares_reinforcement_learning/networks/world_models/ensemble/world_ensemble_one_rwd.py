import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils
from torch import optim
from cares_reinforcement_learning.networks.world_models.simple import Probabilistic_Dynamics
from cares_reinforcement_learning.networks.world_models import World_Model
from cares_reinforcement_learning.util.helpers import normalize_observation_delta
from cares_reinforcement_learning.util import denormalize_observation_delta, normalize_observation

def sig(x):
    """
    Sigmoid
    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


class Ensemble_Dyna_One_Reward(World_Model):
    """
    World Model
    """
    def __init__(self,
                 observation_size: int,
                 num_actions: int,
                 device: str,
                 num_models: int = 5,
                 l_r: float = 0.001,
                 boost_inter: int = 3,
                 hidden_size=None,
                 sas: bool = True,
                 prob_rwd: bool = True):
        super().__init__(observation_size, num_actions, l_r, device, hidden_size, sas, prob_rwd)
        if hidden_size is None:
            hidden_size = [128, 128]
        self.num_models = num_models
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.l_r = l_r
        self.curr_losses = np.ones((self.num_models,)) * 5
        self.world_models = [
            Probabilistic_Dynamics(
                observation_size=observation_size,
                num_actions=num_actions,
                hidden_size=hidden_size,
            )
            for _ in range(self.num_models)
        ]
        self.optimizers = [optim.Adam(self.world_models[i].parameters(), lr=l_r) for i in range(self.num_models)]
        self.statistics = {}
        # Bring all reward prediction and dynamic rediction networks to device.
        self.device = device
        for model in self.world_models:
            model.to(device)
        self.boost_inter = boost_inter
        self.update_counter = 0

    def pred_next_states(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (
                observation.shape[1] + actions.shape[1]
                == self.observation_size + self.num_actions
        )
        norm_means = []
        norm_vars = []
        normalized_observation = normalize_observation(observation, self.statistics)
        # Iterate over the neural networks and get the predictions
        for model in self.world_models:
            # Predict delta
            n_mean, n_var = model.forward(normalized_observation, actions)
            norm_means.append(n_mean)
            norm_vars.append(n_var)
        predictions_vars = torch.stack(norm_vars)
        predictions_norm_means = torch.stack(norm_means)
        # Normalized
        predictions_means = denormalize_observation_delta(predictions_norm_means, self.statistics)
        all_predictions = predictions_means + observation
        denorm_avg = torch.mean(predictions_means, dim=0)
        prediction = denorm_avg + observation
        return prediction, all_predictions, predictions_norm_means, predictions_vars

    def train_world(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
    ) -> None:
        # This boosting part is useless, cause inaccuracy.
        # weights = 1.5 - sig(self.curr_losses)
        # weights /= np.max(weights)
        assert len(states.shape) >= 2
        assert len(actions.shape) == 2
        assert (
                states.shape[1] + actions.shape[1]
                == self.num_actions + self.observation_size
        )
        # min_ = np.min(self.curr_losses)
        # max_ = np.max(self.curr_losses)
        # delta = max_ - min_
        # if delta == 0:
        #     delta = 0.1
        # temp = (self.curr_losses - min_) / delta * 5.0
        # temp = sig(temp)
        # temp[index] *
        index = int(math.floor(self.update_counter / self.boost_inter))
        target = next_states - states
        delta_targets_normalized = normalize_observation_delta(target, self.statistics)
        normalized_state = normalize_observation(states, self.statistics)
        n_mean, n_var = self.world_models[index].forward(normalized_state, actions)
        model_loss = F.gaussian_nll_loss(input=n_mean, target=delta_targets_normalized, var=n_var).mean()
        self.optimizers[index].zero_grad()
        model_loss.backward()
        self.optimizers[index].step()
        self.curr_losses[index] = model_loss.item()
        self.update_counter += 1
        self.update_counter %= self.boost_inter * self.num_models

    def estimate_uncertainty(
            self, observation: torch.Tensor, actions: torch.Tensor, train_reward:bool
    ) -> tuple[float, float, torch.Tensor]:
        """
        Estimate uncertainty.

        :param observation:
        :param actions:
        """
        next_state_samples = None
        uncert_rwd = 0.0
        means = []
        vars_s = []
        normalized_state = normalize_observation(observation, self.statistics)
        for model in self.world_models:
            mean, var = model.forward(normalized_state, actions)
            means.append(mean)
            vars_s.append(var)
        vars_s = torch.stack(vars_s).squeeze()
        noises = vars_s.cpu().detach().numpy()
        aleatoric = (noises ** 2).mean(axis=0) ** 0.5
        all_means = torch.stack(means).squeeze()
        epistemic = all_means.cpu().detach().numpy()
        epistemic = epistemic.var(axis=0) ** 0.5
        aleatoric = np.minimum(aleatoric, 10e3)
        epistemic = np.minimum(epistemic, 10e3)
        total_unc = (aleatoric ** 2 + epistemic ** 2) ** 0.5
        uncert = np.mean(total_unc)
        if train_reward:
            # Reward Uncertainty
            sample_times = 20
            means = torch.vstack(means)
            dist = torch.distributions.Normal(means, vars_s)
            samples = dist.sample([sample_times])
            samples = torch.reshape(samples, (sample_times * self.num_models, self.observation_size))
            samples = denormalize_observation_delta(samples, self.statistics)
            observationss = torch.repeat_interleave(observation, repeats=sample_times * self.num_models, dim=0)
            actionss = torch.repeat_interleave(actions, repeats=sample_times * self.num_models, dim=0)
            samples += observationss

            if self.sas:
                if self.prob_rwd:
                    rewards, rwd_var = self.reward_network(observationss, actionss, samples)
                    epis_uncert = torch.var(rewards, dim=0).item()
                    rwd_var = rwd_var.squeeze().detach().cpu().numpy().mean()
                    alea_uncert = rwd_var
                    epis_uncert = np.minimum(epis_uncert, 10e3)
                    alea_uncert = np.minimum(alea_uncert, 10e3)
                    uncert_rwd = ((epis_uncert ** 2) + (alea_uncert ** 2)) ** 0.5
                else:
                    rewards = self.reward_network(observationss, actionss, samples)
                    uncert_rwd = torch.var(rewards, dim=0).item()
            else:
                if self.prob_rwd:
                    rewards, rwd_var = self.reward_network(samples, actionss)
                    epis_uncert = torch.var(rewards, dim=0).item()
                    rwd_var = rwd_var.squeeze().detach().cpu().numpy().mean()
                    alea_uncert = rwd_var
                    epis_uncert = np.minimum(epis_uncert, 10e3)
                    alea_uncert = np.minimum(alea_uncert, 10e3)
                    uncert_rwd = ((epis_uncert ** 2) + (alea_uncert ** 2)) ** 0.5
                else:
                    rewards = self.reward_network(samples, actionss)
                    uncert_rwd = torch.var(rewards, dim=0).item()
        else:
            dist = torch.distributions.Normal(all_means, vars_s)
            next_state_samples = dist.sample([20])
            next_state_samples = next_state_samples.reshape((self.num_models * 20, self.observation_size))
            next_state_samples = denormalize_observation_delta(next_state_samples, self.statistics)
            next_state_samples += observation
        return uncert, uncert_rwd, next_state_samples

    def train_together(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor):
        sample_times = 20
        normalized_state = normalize_observation(states, self.statistics)
        mean_s = []
        var_s = []
        act_s = []
        state_s = []
        rwd_s = []
        for i in range(self.num_models):
            mean, var = self.world_models[i].forward(normalized_state, actions)
            mean_s.append(mean)
            var_s.append(var)
            act_s.append(actions)
            state_s.append(states)
            rwd_s.append(rewards)

        mean_s = torch.vstack(mean_s)
        var_s = torch.vstack(var_s)
        act_s = torch.vstack(act_s)
        state_s = torch.vstack(state_s)
        rwd_s = torch.vstack(rwd_s)

        dist = torch.distributions.Normal(mean_s, var_s)
        samples = (dist.sample([sample_times]))

        actions = torch.repeat_interleave(act_s.unsqueeze(dim=0), repeats=sample_times, dim=0)
        states = torch.repeat_interleave(state_s.unsqueeze(dim=0), repeats=sample_times,dim=0)
        rwd_s = torch.repeat_interleave(rwd_s.unsqueeze(dim=0), repeats=sample_times, dim=0)

        samples = torch.reshape(samples, (samples.shape[0] * samples.shape[1], self.observation_size))
        states = torch.reshape(states, (states.shape[0] * states.shape[1], states.shape[2]))
        actions = torch.reshape(actions, (actions.shape[0] * actions.shape[1], actions.shape[2]))
        rwd_s = torch.reshape(rwd_s, (rwd_s.shape[0] * rwd_s.shape[1], rwd_s.shape[2]))

        samples = denormalize_observation_delta(samples, self.statistics)
        samples += states


        if self.prob_rwd:
            if self.sas:
                rwd_mean, rwd_var = self.reward_network(states, actions, samples)
            else:
                rwd_mean, rwd_var = self.reward_network(samples, actions)
            rwd_loss = F.gaussian_nll_loss(rwd_mean, rwd_s, rwd_var)
        else:
            if self.sas:
                rwd_mean = self.reward_network(states, actions, samples)
            else:
                rwd_mean = self.reward_network(samples, actions)
            rwd_loss = F.mse_loss(rwd_mean, rwd_s)
        self.reward_optimizer.zero_grad()
        rwd_loss.backward()
        self.reward_optimizer.step()
