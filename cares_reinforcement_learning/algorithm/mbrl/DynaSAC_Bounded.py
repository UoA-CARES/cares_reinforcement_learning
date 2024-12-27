"""
Sutton, Richard S. "Dyna, an integrated architecture for learning, planning, and reacting."

Original Paper: https://dl.acm.org/doi/abs/10.1145/122344.122377

This code runs automatic entropy tuning
"""

import copy
import logging

import numpy as np
import torch
from torch import nn
from cares_reinforcement_learning.memory import MemoryBuffer

from cares_reinforcement_learning.networks.world_models.ensemble import (
    Ensemble_Dyna_Big,
)
import torch.nn.functional as F

class DynaSAC_Bounded:
    def __init__(
        self,
        actor_network: torch.nn.Module,
        critic_network: torch.nn.Module,
        world_network: Ensemble_Dyna_Big,
        gamma: float,
        tau: float,
        action_num: int,
        actor_lr: float,
        critic_lr: float,
        alpha_lr: float,
        num_samples: int,
        horizon: int,
        device: torch.device,
        train_reward: bool,
        train_both: bool,
        gripper:bool,
        threshold:float,
        exploration_sample:int
    ):
        logging.info("-----------------------------------------------")
        logging.info("----I am runing the DynaSAC_Bounded Agent! ----")
        logging.info("-----------------------------------------------")
        self.train_reward = train_reward
        self.train_both = train_both
        self.gripper = gripper
        self.exploration_sample = exploration_sample
        self.threshold = threshold
        self.set_stat = False
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
        self.log_alpha = torch.FloatTensor([np.log(1.0)]).to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -action_num
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # World model
        self.world_model = world_network

        self.k_l = nn.KLDivLoss(reduction='batchmean', log_target=True)

    @property
    def _alpha(self) -> float:
        return self.log_alpha.exp()

    def _jsd(self, p, q):
        p, q = p.view(-1, p.size(-1)).log_softmax(-1), q.view(-1, q.size(-1)).log_softmax(-1)
        m = (0.5 * (p + q))
        return 0.5 * (self.k_l(m, p) + self.k_l(m, q))

    def select_action_from_policy(
        self, state: np.ndarray, evaluation: bool = False, noise_scale: float = 0
    ) -> np.ndarray:
        # note that when evaluating this algorithm we need to select mu as
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if evaluation is False:
                if self.threshold == 0:
                    (action, _, _) = self.actor_net(state_tensor)
                else:
                    if self.set_stat:
                        multi_state_tensor = torch.repeat_interleave(state_tensor, self.exploration_sample, dim=0)
                        (multi_action, multi_log_pi, _) = self.actor_net(multi_state_tensor)
                        # Estimate uncertainty
                        # [6, 10, 17]
                        _, _, nstate_means, nstate_vars = self.world_model.pred_next_states(
                            observation=multi_state_tensor, actions=multi_action)
                        # [10, 17]
                        aleatoric = torch.mean(nstate_vars ** 2, dim=0) ** 0.5
                        epistemic = torch.var(nstate_means, dim=0) ** 0.5
                        aleatoric = torch.clamp(aleatoric, max=10e3)
                        epistemic = torch.clamp(epistemic, max=10e3)
                        total_unc = (aleatoric ** 2 + epistemic ** 2) ** 0.5
                        uncert = torch.mean(total_unc, dim=1)

                        multi_log_pi = multi_log_pi.squeeze()
                        policy_dist = F.softmax(multi_log_pi, dim=0)
                        world_dist = F.softmax(uncert, dim=0)
                        world_dist -= torch.min(world_dist)

                        final_dist = policy_dist + self.threshold * world_dist
                        #final_dist = F.softmax(final_dist, dim=0)
                        candi = torch.argmax(final_dist)
                        # new_dist = torch.distributions.Categorical(final_dist)
                        # candi = new_dist.sample([5]).squeeze()
                        # print(self._jsd(policy_dist, final_dist))

                        action = multi_action[candi]
                    else:
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
        if weights is None:
            weights = torch.ones(rewards.shape)
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
        self, memory: MemoryBuffer, batch_size: int
    ) -> None:

        experiences = memory.sample_uniform(batch_size)
        states, actions, rewards, next_states, _, _ = experiences

        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)

        self.world_model.train_world(
            states=states,
            actions=actions,
            next_states=next_states,
        )

        batch_size = len(states)
        # Reshape to batch_size x whatever
        if self.train_reward:
            rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device)
            rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
            if self.train_both:
                self.world_model.train_together(states, actions, rewards)
            else:
                self.world_model.train_reward(states, actions, next_states, rewards)

    def train_policy(self, memory: MemoryBuffer, batch_size: int) -> None:
        self.learn_counter += 1

        experiences = memory.sample_uniform(batch_size)
        states, actions, rewards, next_states, dones, _ = experiences

        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.LongTensor(np.asarray(dones)).to(self.device).unsqueeze(1)

        # Step 2 train as usual
        self._train_policy(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            weights=torch.ones(rewards.shape)
        )
        self._dyna_generate_and_train(next_states)

    def _dyna_generate_and_train(self, next_states: torch.Tensor) -> None:
        pred_states = []
        pred_actions = []
        pred_rs = []
        pred_n_states = []

        with torch.no_grad():
            pred_state = next_states
            for _ in range(self.horizon):
                pred_state = torch.repeat_interleave(pred_state, self.num_samples, dim=0)
                # This part is controversial. But random actions is empirically better.
                # rand_acts = np.random.uniform(-1, 1, (pred_state.shape[0], self.action_num))
                # pred_acts = torch.FloatTensor(rand_acts).to(self.device)
                pred_acts, _, _ = self.actor_net(pred_state)
                pred_next_state, _, _, _ = self.world_model.pred_next_states(
                    pred_state, pred_acts
                )

                if self.gripper:
                    pred_reward = self.reward_function(pred_state, pred_next_state)
                    pred_next_state[:, -2:] = pred_state[:, -2:]
                else:
                    pred_reward, _ = self.world_model.pred_rewards(observation=pred_state,
                                                                   action=pred_acts,
                                                                   next_observation=pred_next_state)

                pred_states.append(pred_state)
                pred_actions.append(pred_acts.detach())
                pred_rs.append(pred_reward.detach())
                pred_n_states.append(pred_next_state.detach())
                pred_state = pred_next_state.detach()
            pred_states = torch.vstack(pred_states)
            pred_actions = torch.vstack(pred_actions)
            pred_rs = torch.vstack(pred_rs)
            pred_n_states = torch.vstack(pred_n_states)
            # Pay attention to here! It is dones in the Cares RL Code!
            pred_dones = torch.FloatTensor(np.zeros(pred_rs.shape)).to(self.device)
            # states, actions, rewards, next_states, not_dones
        self._train_policy(
            pred_states, pred_actions, pred_rs, pred_n_states, pred_dones, torch.ones(pred_rs.shape)
        )

    def reward_function(self, curr_states, next_states):
        target_goal_tensor = curr_states[:, -2:]
        object_current = next_states[:, -4:-2]
        sq_diff = (target_goal_tensor - object_current) ** 2
        # [256, 1]
        goal_distance_after = torch.sqrt(torch.sum(sq_diff, dim=1)).unsqueeze(dim=1)
        pred_reward = -goal_distance_after + 70
        mask1 = goal_distance_after <= 10
        mask2 = goal_distance_after > 70
        pred_reward[mask1] = 800
        pred_reward[mask2] = 0
        return pred_reward

    def set_statistics(self, stats: dict) -> None:
        self.world_model.set_statistics(stats)
        self.set_stat = True

    def save_models(self, filename: str, filepath: str = "models") -> None:
        # if not os.path.exists(filepath):
        #     os.makedirs(filepath)
        # print(filepath)
        # logging.info(filepath)
        # torch.save(self.actor_net.state_dict(), f"{filepath}/{filename}_actor.pht")
        # torch.save(self.critic_net.state_dict(), f"{filepath}/{filename}_critic.pht")
        logging.info("models has been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        self.actor_net.load_state_dict(torch.load(f"{filepath}/{filename}_actor.pht"))
        self.critic_net.load_state_dict(torch.load(f"{filepath}/{filename}_critic.pht"))
        logging.info("models has been loaded...")
