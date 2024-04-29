"""
Sutton, Richard S. "Dyna, an integrated architecture for learning, planning, and reacting."

Original Paper: https://dl.acm.org/doi/abs/10.1145/122344.122377

This code runs automatic entropy tuning
"""

import copy
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from cares_reinforcement_learning.memory import PrioritizedReplayBuffer
from cares_reinforcement_learning.util import sampling

from cares_reinforcement_learning.networks.world_models.ensemble_integrated import (
    EnsembleWorldReward,
)
from cares_reinforcement_learning.networks.world_models.ensemble_world import (
    EnsembleWorldAndOneReward,
)


class DynaSAC_Reweight:
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
        device: torch.device,
    ):
        self.type = "mbrl"
        self.device = device

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

        l1_loss_one = q_values_one - q_target
        l1_loss_two = q_values_two - q_target

        reweighted_l1_loss_one = weights * l1_loss_one
        reweighted_l1_loss_two = weights * l1_loss_two

        total_loss_after_one = torch.sum(reweighted_l1_loss_one)
        total_loss_after_two = torch.sum(reweighted_l1_loss_two)

        total_loss_one = torch.sum(l1_loss_one)
        total_loss_two = torch.sum(l1_loss_two)

        ratio_one = total_loss_one / total_loss_after_one
        ratio_two = total_loss_two / total_loss_after_two

        reweighted_l1_loss_one = reweighted_l1_loss_one * ratio_one
        reweighted_l1_loss_two = reweighted_l1_loss_two * ratio_two

        critic_loss_one = 0.5 * (reweighted_l1_loss_one.pow(2)).mean()
        critic_loss_two = 0.5 * (reweighted_l1_loss_two.pow(2)).mean()

        # critic_loss_one = F.mse_loss(q_values_one, q_target)
        # critic_loss_two = F.mse_loss(q_values_two, q_target)
        critic_loss_total = critic_loss_one + critic_loss_two

        # Update the Critic
        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        ##################     Update the Actor Second     ####################
        pi, first_log_p, _ = self.actor_net(states)
        qf1_pi, qf2_pi = self.critic_net(states, pi)
        min_qf_pi = torch.minimum(qf1_pi, qf2_pi)
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

        # experiences = memory.sample_consecutive(batch_size)
        #
        # (
        #     states,
        #     actions,
        #     rewards,
        #     next_states,
        #     _,
        #     _,
        #     next_actions,
        #     next_rewards,
        #     _,
        #     _,
        #     _,
        # ) = experiences

        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        # next_rewards = (
        #     torch.FloatTensor(np.asarray(next_rewards)).to(self.device).unsqueeze(1)
        # )
        # next_actions = torch.FloatTensor(np.asarray(next_actions)).to(self.device)

        # # Step 1 train the world model.
        # self.world_model.train_world(
        #     states=states,
        #     actions=actions,
        #     rewards=rewards,
        #     next_states=next_states,
        #     next_actions=next_actions,
        #     next_rewards=next_rewards,
        # )

        self.world_model.train_world(
            states=states,
            actions=actions,
            next_states=next_states,
        )
        self.world_model.train_reward(
            next_states=next_states,
            actions=actions,
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
            uncert = 1.5 - uncert
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
        # 5 models, each sampled 10 times = 50,
        sample1 = torch.distributions.Normal(pred_means[0], pred_vars[0]).sample(
            [10])
        rwd_sample1 = self.world_model.pred_rewards(sample1)

        sample2 = torch.distributions.Normal(pred_means[1], pred_vars[1]).sample(
            [10])
        rwd_sample2 =  self.world_model.pred_rewards(sample2)

        sample3 = torch.distributions.Normal(pred_means[2], pred_vars[2]).sample(
            [10])
        rwd_sample3 = self.world_model.pred_rewards(sample3)

        sample4 = torch.distributions.Normal(pred_means[3], pred_vars[3]).sample(
            [10])
        rwd_sample4 = self.world_model.pred_rewards(sample4)

        sample5 = torch.distributions.Normal(pred_means[4], pred_vars[4]).sample(
            [10])
        rwd_sample5 = self.world_model.pred_rewards(sample5)

        samples = torch.cat((rwd_sample1, rwd_sample2, rwd_sample3, rwd_sample4, rwd_sample5))
        # samples = torch.cat((sample1, sample2, sample3, sample4, sample5))
        # Samples = [5 * 10, 10 predictions, 11 state dims]
        # print(samples.shape)
        stds = torch.var(samples, dim=0)
        # print(stds.shape)
        # [10 predictions, 11 state dims]
        total_stds = torch.mean(stds, dim=1)
        # Clip for sigmoid
        # total_stds[total_stds < 0.2] = 0.0
        # total_stds[total_stds > 4.0] = 4.0

        total_stds = F.sigmoid(total_stds)  # 0.5 - 1.0
        # total_stds = 1 / total_stds
        # total_stds = total_stds / torch.mean(total_stds)  # if very uncertain,
        # high std, encouraged.
        # total_stds = total_stds - torch.min(total_stds)
        return total_stds.detach()

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
