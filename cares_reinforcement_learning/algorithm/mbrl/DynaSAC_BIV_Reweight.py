"""
Sutton, Richard S. "Dyna, an integrated architecture for learning, planning, and reacting."

Original Paper: https://dl.acm.org/doi/abs/10.1145/122344.122377

This code runs automatic entropy tuning
"""

import copy
import logging
import os
from scipy.optimize import minimize
import numpy as np
import torch
from cares_reinforcement_learning.memory import PrioritizedReplayBuffer
import torch.nn.functional as F

from cares_reinforcement_learning.networks.world_models.ensemble_ns_world import (
    EnsembleWorldAndOneReward,
)


class DynaSAC_BIVReweight:
    """
    Max as ?
    """

    def __init__(
            self,
            actor_network: torch.nn.Module,
            critic_network: torch.nn.Module,
            world_network: EnsembleWorldAndOneReward,
            gamma: float,
            tau: float,
            action_num: int,
            actor_lr: float,
            critic_lr: float,
            alpha_lr: float,
            num_samples: int,
            horizon: int,
            threshold_scale: float,
            reweight_critic: bool,
            reweight_actor: bool,
            mode: int,
            sample_times: int,
            device: torch.device,
    ):
        self.type = "mbrl"
        self.device = device
        self.reweight_critic = reweight_critic
        self.reweight_actor = reweight_actor
        # this may be called policy_net in other implementations
        self.actor_net = actor_network.to(self.device)
        # this may be called soft_q_net in other implementations
        self.critic_net = critic_network.to(self.device)
        self.target_critic_net = copy.deepcopy(self.critic_net)

        self.gamma = gamma
        self.tau = tau

        self.num_samples = num_samples
        self.horizon = horizon
        self.action_num = action_num

        self.learn_counter = 0
        self.policy_update_freq = 1

        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=actor_lr
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=critic_lr
        )

        # Set to initial alpha to 1.0 according to other baselines.
        self.log_alpha = torch.tensor(np.log(1.0)).to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -action_num
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # World model
        self.world_model = world_network
        # Parameter
        self.threshold_scale = threshold_scale
        self.mode = mode
        self.sample_times = sample_times

    @property
    def _alpha(self) -> float:
        return self.log_alpha.exp()

    # pylint: disable-next=unused-argument to keep the same interface
    def select_action_from_policy(
            self, state: np.ndarray, evaluation: bool = False, noise_scale: float = 0
    ) -> np.ndarray:
        # note that when evaluating this algorithm we need to select mu as
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if evaluation is False:
                (action, _, _) = self.actor_net(state_tensor)
            else:
                (_, _, action) = self.actor_net(state_tensor)
            action = action.cpu().data.numpy().flatten()
        self.actor_net.train()
        return action

    def _train_policy(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            next_states: torch.Tensor,
            dones: torch.Tensor,
            weights: torch.Tensor,
    ) -> None:
        ##################     Update the Critic First     ####################
        # Have more target values?
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor_net(next_states)
            target_q_one, target_q_two = self.target_critic_net(
                next_states, next_actions
            )
            target_q_values = (
                    torch.minimum(target_q_one, target_q_two) - self._alpha * next_log_pi
            )
            q_target = rewards + self.gamma * (1 - dones) * target_q_values

        q_values_one, q_values_two = self.critic_net(states, actions)

        if self.reweight_critic:
            # Reweighted loss function. weight not participant in training.
            l2_loss_one = (q_values_one - q_target).pow(2)
            l2_loss_two = (q_values_two - q_target).pow(2)

            weights = weights.detach()
            disc_l2_loss_one = l2_loss_one * weights
            disc_l2_loss_two = l2_loss_two * weights
            # A ratio to scale the loss back to original loss scale.

            ratio_1 = torch.mean(l2_loss_one) / torch.mean(disc_l2_loss_one)
            ratio_1 = ratio_1.detach()
            ratio_2 = torch.mean(l2_loss_two) / torch.mean(disc_l2_loss_two)
            ratio_2 = ratio_2.detach()

            critic_loss_one = disc_l2_loss_one.mean() * ratio_1
            critic_loss_two = disc_l2_loss_two.mean() * ratio_2

            critic_loss_total = critic_loss_one + critic_loss_two
        else:
            critic_loss_one = F.mse_loss(q_values_one, q_target)
            critic_loss_two = F.mse_loss(q_values_two, q_target)
            critic_loss_total = critic_loss_one + critic_loss_two

        # Update the Critic
        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        ##################     Update the Actor Second     ####################
        pi, first_log_p, _ = self.actor_net(states)
        qf1_pi, qf2_pi = self.critic_net(states, pi)
        min_qf_pi = torch.minimum(qf1_pi, qf2_pi)

        if self.reweight_actor:
            weights = weights.detach()
            a_loss = (self._alpha * first_log_p) - min_qf_pi
            disc_actor_loss = a_loss * weights
            ratio = torch.mean(a_loss) / torch.mean(disc_actor_loss)
            ratio = ratio.detach()
            actor_loss = ratio * torch.mean(disc_actor_loss)
        else:
            actor_loss = ((self._alpha * first_log_p) - min_qf_pi).mean()

        # Update the Actor
        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        # Update the temperature
        alpha_loss = -(
                self.log_alpha * (first_log_p + self.target_entropy).detach()
        ).mean()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        if self.learn_counter % self.policy_update_freq == 0:
            for target_param, param in zip(
                    self.target_critic_net.parameters(), self.critic_net.parameters()
            ):
                target_param.data.copy_(
                    param.data * self.tau + target_param.data * (1.0 - self.tau)
                )

    def train_world_model(
            self, memory: PrioritizedReplayBuffer, batch_size: int
    ) -> None:
        experiences = memory.sample_uniform(batch_size)
        states, actions, rewards, next_states, _, _ = experiences

        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)

        self.world_model.train_world(
            states=states,
            actions=actions,
            next_states=next_states,
        )
        self.world_model.train_reward(
            next_states=next_states,
            rewards=rewards,
        )

    def train_policy(self, memory: PrioritizedReplayBuffer, batch_size: int) -> None:
        self.learn_counter += 1

        experiences = memory.sample_uniform(batch_size)
        states, actions, rewards, next_states, dones, _ = experiences

        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.LongTensor(np.asarray(dones)).to(self.device).unsqueeze(1)
        full_weights = torch.ones(rewards.shape).to(self.device)
        # Step 2 train as usual
        self._train_policy(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            weights=full_weights,
        )
        # # # Step 3 Dyna add more data
        self._dyna_generate_and_train(next_states=next_states)

    def _dyna_generate_and_train(self, next_states):
        """
        Only off-policy Dyna will work.
        :param next_states:
        """
        pred_states = []
        pred_actions = []
        pred_rs = []
        pred_n_states = []
        pred_uncerts = []
        with torch.no_grad():
            pred_state = next_states
            for _ in range(self.horizon):
                pred_state = torch.repeat_interleave(pred_state, self.num_samples, dim=0)
                # This part is controversial. But random actions is empirically better.
                rand_acts = np.random.uniform(-1, 1, (pred_state.shape[0], self.action_num))
                pred_acts = torch.FloatTensor(rand_acts).to(self.device)

                pred_next_state, _, pred_mean, pred_var = self.world_model.pred_next_states(
                    pred_state, pred_acts
                )
                uncert = self.sampling(pred_means=pred_mean, pred_vars=pred_var)
                uncert = uncert.unsqueeze(dim=1).to(self.device)
                pred_uncerts.append(uncert)

                pred_reward = self.world_model.pred_rewards(pred_next_state)
                pred_states.append(pred_state)
                pred_actions.append(pred_acts.detach())
                pred_rs.append(pred_reward.detach())
                pred_n_states.append(pred_next_state.detach())
                pred_state = pred_next_state.detach()
            pred_states = torch.vstack(pred_states)
            pred_actions = torch.vstack(pred_actions)
            pred_rs = torch.vstack(pred_rs)
            pred_n_states = torch.vstack(pred_n_states)
            pred_weights = torch.vstack(pred_uncerts)
            # Pay attention to here! It is dones in the Cares RL Code!
            pred_dones = torch.FloatTensor(np.zeros(pred_rs.shape)).to(self.device)
            # states, actions, rewards, next_states, not_dones
        self._train_policy(
            pred_states, pred_actions, pred_rs, pred_n_states, pred_dones, pred_weights
        )

    def sampling(self, pred_means, pred_vars):
        """
        High std means low uncertainty. Therefore, divided by 1

        :param pred_means:
        :param pred_vars:
        :return:
        """
        with torch.no_grad():
            # 5 models. Each predict 10 next_states.
            sample1 = torch.distributions.Normal(pred_means[0], pred_vars[0]).sample(
                [self.sample_times])
            sample2 = torch.distributions.Normal(pred_means[1], pred_vars[1]).sample(
                [self.sample_times])
            sample3 = torch.distributions.Normal(pred_means[2], pred_vars[2]).sample(
                [self.sample_times])
            sample4 = torch.distributions.Normal(pred_means[3], pred_vars[3]).sample(
                [self.sample_times])
            sample5 = torch.distributions.Normal(pred_means[4], pred_vars[4]).sample(
                [self.sample_times])
            rs = []
            acts = []
            qs = []
            # Varying the next_state's distribution.
            for i in range(self.sample_times):
                # 5 models, each sampled 10 times = 50,
                pred_rwd1 = self.world_model.pred_rewards(sample1[i])
                pred_rwd2 = self.world_model.pred_rewards(sample2[i])
                pred_rwd3 = self.world_model.pred_rewards(sample3[i])
                pred_rwd4 = self.world_model.pred_rewards(sample4[i])
                pred_rwd5 = self.world_model.pred_rewards(sample5[i])
                rs.append(pred_rwd1)
                rs.append(pred_rwd2)
                rs.append(pred_rwd3)
                rs.append(pred_rwd4)
                rs.append(pred_rwd5)
                # Each times, 5 models predict different actions.
                # [2560, 17]
                pred_act1, log_pi1, _ = self.actor_net(sample1[i])
                pred_act2, log_pi2, _ = self.actor_net(sample2[i])
                pred_act3, log_pi3, _ = self.actor_net(sample3[i])
                pred_act4, log_pi4, _ = self.actor_net(sample4[i])
                pred_act5, log_pi5, _ = self.actor_net(sample5[i])
                acts.append(log_pi1)
                acts.append(log_pi2)
                acts.append(log_pi3)
                acts.append(log_pi4)
                acts.append(log_pi5)
                # How to become the same next state, different action.
                # Now: sample1 sample2... same next state, different model.
                # Pred_act1 pred_act2 same next_state, different actions.
                # 5[] * 10[var of state]
                qa1, qa2 = self.target_critic_net(sample1[i], pred_act1)
                qa = torch.minimum(qa1, qa2)
                qb1, qb2 = self.target_critic_net(sample2[i], pred_act2)
                qb = torch.minimum(qb1, qb2)
                qc1, qc2 = self.target_critic_net(sample3[i], pred_act3)
                qc = torch.minimum(qc1, qc2)
                qd1, qd2 = self.target_critic_net(sample4[i], pred_act4)
                qd = torch.minimum(qd1, qd2)
                qe1, qe2 = self.target_critic_net(sample5[i], pred_act5)
                qe = torch.minimum(qe1, qe2)
                qs.append(qa)
                qs.append(qb)
                qs.append(qc)
                qs.append(qd)
                qs.append(qe)

            rs = torch.stack(rs)
            acts = torch.stack(acts)
            qs = torch.stack(qs)

            var_r = torch.var(rs, dim=0)

            if self.mode < 3:
                var_a = torch.var(acts, dim=0)
                var_q = torch.var(qs, dim=0)

            # Computing covariance.
            if self.mode < 2:
                mean_a = torch.mean(acts, dim=0, keepdim=True)
                mean_q = torch.mean(qs, dim=0, keepdim=True)
                diff_a = acts - mean_a
                diff_q = qs - mean_q
                cov_aq = torch.mean(diff_a * diff_q, dim=0)

            if self.mode < 1:
                mean_r = torch.mean(rs, dim=0, keepdim=True)
                diff_r = rs - mean_r
                cov_rq = torch.mean(diff_r * diff_q, dim=0)

                cov_ra = torch.mean(diff_r * diff_a, dim=0)

            gamma_sq = self.gamma * self.gamma
            # Ablation
            if self.mode == 0:
                total_var = var_r + gamma_sq * var_a + gamma_sq * var_q + gamma_sq * 2 * cov_aq + \
                            gamma_sq * 2 * cov_rq + gamma_sq * 2 * cov_ra
            if self.mode == 1:
                total_var = var_r + gamma_sq * var_a + gamma_sq * var_q + gamma_sq * 2 * cov_aq
            if self.mode == 2:
                total_var = var_r + gamma_sq * var_a + gamma_sq * var_q
            if self.mode == 3:
                total_var = var_r

            xi = self.get_optimal_xi(total_var.detach().cpu().squeeze().numpy())
            total_var += xi
            # Weight = inverse of sum of weights * inverse of varaince.
            weights = 1.0 / total_var
            ratio = 1.0 / torch.sum(weights)
            total_stds = ratio * weights
        return total_stds.detach()


    def get_optimal_xi(self, variances):
        minimal_size = self.threshold_scale
        if self.compute_eff_bs(self.get_iv_weights(variances)) >= minimal_size:
            return 0
        fn = lambda x: np.abs(self.compute_eff_bs(self.get_iv_weights(variances + np.abs(x))) - minimal_size)
        epsilon = minimize(fn, 0, method='Nelder-Mead', options={'fatol': 1.0, 'maxiter': 100})
        xi = np.abs(epsilon.x[0])
        xi = 0 if xi is None else xi
        return xi

    def get_iv_weights(self, variances):
        '''
        Returns Inverse Variance weights
        Params
        ======
            variances (numpy array): variance of the targets
        '''
        weights = 1 / variances
        weights = weights / np.sum(weights)
        return weights

    def compute_eff_bs(self, weights):
        # Compute original effective mini-batch size
        eff_bs = 1 / np.sum(np.square(weights))
        eff_bs = eff_bs / np.shape(weights)[0]
        return eff_bs



    def set_statistics(self, stats: dict) -> None:
        self.world_model.set_statistics(stats)

    def save_models(self, filename: str, filepath: str = "models") -> None:
        path = f"{filepath}/models" if filepath != "models" else filepath
        dir_exists = os.path.exists(path)
        if not dir_exists:
            os.makedirs(path)
        torch.save(self.actor_net.state_dict(), f"{path}/{filename}_actor.pth")
        torch.save(self.critic_net.state_dict(), f"{path}/{filename}_critic.pth")
        logging.info("models has been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        path = f"{filepath}/models" if filepath != "models" else filepath
        self.actor_net.load_state_dict(torch.load(f"{path}/{filename}_actor.pth"))
        self.critic_net.load_state_dict(torch.load(f"{path}/{filename}_critic.pth"))
        logging.info("models has been loaded...")
