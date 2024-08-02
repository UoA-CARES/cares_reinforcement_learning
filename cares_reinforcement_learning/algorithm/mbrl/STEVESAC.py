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

from cares_reinforcement_learning.memory import PrioritizedReplayBuffer

from cares_reinforcement_learning.networks.world_models.ensemble_all import (
    EnsembleWorldRewardDone,
)


class STEVE:
    def __init__(
            self,
            actor_network: torch.nn.Module,
            critic_network: torch.nn.Module,
            world_network: EnsembleWorldRewardDone,
            gamma: float,
            tau: float,
            action_num: int,
            actor_lr: float,
            critic_lr: float,
            alpha_lr: float,
            horizon: int,
            L: int,
            device: torch.device,
    ):
        self.L = L
        self.horizon = horizon
        self.type = "mbrl"
        self.device = device
        # this may be called policy_net in other implementations
        self.actor_net = actor_network.to(self.device)
        # this may be called soft_q_net in other implementations
        self.critic_net = critic_network.to(self.device)
        self.target_critic_net = copy.deepcopy(self.critic_net)
        self.gamma = gamma
        self.tau = tau

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
    ) -> None:
        ##################     Update the Critic First     ####################
        with torch.no_grad():
            # cumulative_rewards = rewards
            # pred_s = next_states
            # tree_mask = dones.squeeze().bool()
            # q_means = []
            # q_vars = []

            # for hori in range(self.horizon):
            #     # As normal
            #     pred_a, _, _ = self.actor_net(pred_s)
            #     # Pred the future
            #     pred_s, pred_r, pred_done = self.env.tensor_query(pred_s, pred_a)
            #     pred_s = pred_s.to(self.device)
            #     pred_r = pred_r.to(self.device)
            #     pred_done = pred_done.bool().to(self.device)
            #     # Before adding pred to mask
            #     pred_r[tree_mask, :] = 0.0
            #     cumulative_rewards += pred_r * (self.gamma ** (hori + 1))
            #     # Kill the branch with the previous
            #     tree_mask = torch.logical_or(tree_mask, pred_done.squeeze())
            # q_target = cumulative_rewards

            # Expand the value estimation here to STEVE.
            # Maintain a list of Q values for each horizon with the same size!
            # Q = r + yr + y^2 * (q - pi)
            # Propagate uncertainty with sampling. [256, 17] -> [10, 256, 17] -> [100, 256, 17] -> [1000, 256, 17]
            # Q : [256, 1], [256, 1], [256, 1]

            # not_dones = (1 - dones).squeeze().bool()
            # pred_all_next_obs = next_states.unsqueeze(dim=0)
            # pred_all_next_rewards = torch.zeros(rewards.shape).unsqueeze(dim=0)
            #
            # q_means = []
            # q_vars = []
            #
            # for hori in range(self.horizon):
            #     horizon_rewards_list = []
            #     horizon_obs_list = []
            #     horizon_q_list = []
            #
            #     for stat in range(pred_all_next_obs.shape[0]):
            #         # Optimal sampling
            #         pred_action, pred_log_pi, _ = self.actor_net.sample(pred_all_next_obs[stat])
            #
            #         pred_q1, pred_q2 = self.target_critic_net(pred_all_next_obs[stat], pred_action)
            #         # V = Q - alpha * logi
            #         pred_v1 = pred_q1 - self._alpha * pred_log_pi
            #         pred_v2 = pred_q2 - self._alpha * pred_log_pi
            #
            #         # Predict a set of reward first
            #         _, pred_rewards = self.world_model.pred_rewards(observation=pred_all_next_obs[stat],
            #                                                         action=pred_action)
            #
            #         temp_disc_rewards = []
            #         # For each predict reward.
            #         for rwd in range(pred_rewards.shape[0]):
            #             disc_pred_reward = not_dones * (self.gamma ** (hori + 1)) * pred_rewards[rwd]
            #             if hori > 0:
            #                 # Horizon = 1, 2, 3, 4, 5
            #                 disc_sum_reward = pred_all_next_rewards[stat] + disc_pred_reward
            #             else:
            #                 disc_sum_reward = not_dones * disc_pred_reward
            #             temp_disc_rewards.append(disc_sum_reward)
            #             assert rewards.shape == not_dones.shape == disc_sum_reward.shape
            #             # Q = r + disc_rewards + pred_v
            #             pred_tq1 = rewards + disc_sum_reward + not_dones * (self.gamma ** (hori + 2)) * pred_v1
            #             pred_tq2 = rewards + disc_sum_reward + not_dones * (self.gamma ** (hori + 2)) * pred_v2
            #             horizon_q_list.append(pred_tq1)
            #             horizon_q_list.append(pred_tq2)
            #
            #         # Observation Level
            #         if hori < (self.horizon - 1):
            #             _, pred_obs, _, _ = self.world_model.pred_next_states(pred_all_next_obs[stat], pred_action)
            #
            #             horizon_obs_list.append(pred_obs)
            #             horizon_rewards_list.append(torch.stack(temp_disc_rewards))
            #
            #     # Horizon level.
            #     if hori < (self.horizon - 1):
            #         pred_all_next_obs = torch.vstack(horizon_obs_list)
            #         pred_all_next_rewards = torch.vstack(horizon_rewards_list)
            #
            #     # Statistics of target q
            #     h_0 = torch.stack(horizon_q_list)
            #     mean_0 = torch.mean(h_0, dim=0)
            #     q_means.append(mean_0)
            #     var_0 = torch.var(h_0, dim=0)
            #     var_0[torch.abs(var_0) < 0.001] = 0.001
            #     var_0 = 1.0 / var_0
            #     q_vars.append(var_0)
            # all_means = torch.stack(q_means)
            # all_vars = torch.stack(q_vars)
            # total_vars = torch.sum(all_vars, dim=0)
            # for n in range(self.horizon):
            #     all_vars[n] /= total_vars
            # q_target = torch.sum(all_vars * all_means, dim=0)

            next_actions, next_log_pi, _ = self.actor_net(next_states)

            target_q_one, target_q_two = self.target_critic_net(
                next_states, next_actions
            )
            target_q_values = (
                    torch.minimum(target_q_one, target_q_two) - self._alpha * next_log_pi
            )
            q_target = rewards + self.gamma * (1 - dones) * target_q_values

        q_values_one, q_values_two = self.critic_net(states, actions)
        critic_loss_one = ((q_values_one - q_target).pow(2)).mean()
        critic_loss_two = ((q_values_two - q_target).pow(2)).mean()
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

        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)

        # Brief Evaluate the world model and reward prediciton.
        next_s, _, _, _ = self.world_model.pred_next_states(states, actions)

        self.world_model.train_world(
            states=states,
            actions=actions,
            next_states=next_states,
        )
        self.world_model.train_reward(
            states=states,
            actions=actions,
            rewards=rewards,
        )


    def train_policy(self, memory: PrioritizedReplayBuffer, batch_size: int) -> None:
        self.learn_counter += 1

        # experiences = memory.sample_uniform(batch_size)
        # states, actions, rewards, next_states, dones, _ = experiences
        #
        # # Convert into tensor
        # states = torch.FloatTensor(np.asarray(states)).to(self.device)
        # actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        # rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device).unsqueeze(1)
        # next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        # dones = torch.LongTensor(np.asarray(dones)).to(self.device).unsqueeze(1)
        #
        # # Step 2 train as usual
        # self._train_policy(
        #     states=states,
        #     actions=actions,
        #     rewards=rewards,
        #     next_states=next_states,
        #     dones=dones,
        # )

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
