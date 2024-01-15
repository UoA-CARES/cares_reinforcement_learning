"""
Original Paper: https://arxiv.org/abs/1812.05905
Code based on: https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py.

This code runs automatic entropy tuning
"""

import copy
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F


class SAC_MBRL:
    """
    Soft Actor Critic: Adding a entropy of policy term into the objective
    function for automatic exploration. This agent maximize the reward will
    maximizing the exploration.

    """

    def __init__(
            self,
            actor_network,
            critic_network,
            world_network,
            gamma,
            tau,
            action_num,
            actor_lr,
            critic_lr,
            device,
    ):
        self.use_bounded_active = False
        self.use_mve_steve = False
        self.use_mve_actor = False
        self.use_mve_critic = False
        self.use_dyna = False
        self.horizon = 5

        self.type = "policy"
        # this may be called policy_net in other implementations
        self.actor_net = actor_network.to(device)
        # this may be called soft_q_net in other implementations
        self.critic_net = critic_network.to(device)
        self.target_critic_net = copy.deepcopy(self.critic_net).to(device)
        self.gamma = gamma
        self.tau = tau
        self.learn_counter = 0
        self.policy_update_freq = 1
        self.device = device

        self.target_entropy = -action_num
        self.actor_net_optimiser = torch.optim.Adam(self.actor_net.parameters()
                                                    , lr=actor_lr)
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters()
            , lr=critic_lr)

        # Set to initial alpha to 1.0 according to other baselines.
        init_temperature = 1.0
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha])
        # World model
        self.world_model = world_network

    # pylint: disable-next=unused-argument to keep the same interface
    def select_action_from_policy(self, state, evaluation=False,
                                  noise_scale=0):
        # note that when evaluating this algorithm we need to select mu as action
        # so _, _, action = self.actor_net.sample(state_tensor)
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            state_tensor = state_tensor.unsqueeze(0).to(self.device)
            if evaluation is False:
                (
                    action,
                    _,
                    _,
                ) = self.actor_net.sample(state_tensor)
            else:
                (
                    _,
                    _,
                    action,
                ) = self.actor_net.sample(state_tensor)
            action = action.cpu().data.numpy().flatten()
        self.actor_net.train()
        return action

    @property
    def alpha(self):
        """
        A variatble decide to what extend entropy shoud be valued.
        """
        return self.log_alpha.exp()

    def train_policy(self, experiences):
        """
        Train the policy with Model-Based Value Expansion. A family of MBRL.

        """
        self.learn_counter += 1
        info = {}
        states, actions, rewards, next_states, dones = experiences
        batch_size = len(states)
        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(
            self.device)
        dones = torch.LongTensor(np.asarray(dones)).to(self.device)
        # Reshape to batch_size x whatever
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        dones = dones.unsqueeze(0).reshape(batch_size, 1)

        not_dones = 1 - dones
        assert len(states.shape) >= 2
        assert len(actions.shape) == 2
        assert len(rewards.shape) == 2 and rewards.shape[1] == 1
        assert len(next_states.shape) >= 2
        assert len(not_dones.shape) == 2 and not_dones.shape[1] == 1

        with torch.no_grad():
            #    Expand the critic with reward and next state prediction    #
            if self.use_mve_critic:
                pred_rewards = torch.zeros(rewards.shape)
                pred_next_obs = next_states
                for i in range(self.horizon):
                    _, pred_target_us, pred_log_pi, _ = self.actor_net.forward(
                        pred_next_obs)
                    pred_reward, _ = self.world_model.predict_rewards(
                        obs=pred_next_obs, actions=pred_target_us)
                    pred_rewards += (self.gamma ** (i + 1)) * pred_reward
                    pred_next_obs, _, _, _ = self.world_model.pred_next_states(
                        pred_next_obs, pred_target_us)
                target_q1, target_q2 = self.critic_net.forward(pred_next_obs,
                                                               pred_target_us)
                target_q = torch.min(target_q1,
                                     target_q2)-self.alpha.detach()*pred_log_pi
                q_target = rewards + not_dones * pred_rewards + not_dones * (
                        self.gamma ** (i + 2)) * target_q
            if self.use_mve_steve:
                # For next episodes used
                pred_all_next_obs = next_states.unsqueeze(dim=0)
                pred_nt_rds = rewards.unsqueeze(dim=0)
                means = []
                vars = []
                for hori in range(self.horizon):
                    pred_all_next_rewards_list = []
                    pred_all_next_next_obs = []
                    est_target_q = []
                    # For each state batch [256, 17], reward extend 5 times,
                    # next extend 5 time.
                    for stat in range(pred_all_next_obs.shape[0]):
                        _, pred_a, pred_log_pi, _ = self.actor_net(
                            pred_all_next_obs[stat])
                        pred_target_q1, pred_target_q2 = self.critic_net(
                            pred_all_next_obs[stat], pred_a)
                        pred_target_q = torch.min(
                            pred_target_q1,
                            pred_target_q2) - self.alpha.detach(
                        ) * pred_log_pi

                        _, pred_rewards = self.world_model.predict_rewards(
                            obs=pred_all_next_obs[stat], actions=pred_a)

                        temp_disc_rewards = []
                        for rwd in range(pred_rewards.shape[0]):
                            dis_pred_rd = (self.gamma ** (hori + 1)) * \
                                          pred_rewards[rwd]
                            if hori > 0:
                                pred_rd = pred_nt_rds[
                                              stat] + not_dones * dis_pred_rd
                            else:
                                pred_rd = not_dones * dis_pred_rd
                            temp_disc_rewards.append(pred_rd)
                            assert rewards.shape == not_dones.shape == pred_rd.shape == pred_target_q.shape
                            qs = not_dones * (self.gamma ** (
                                    hori + 2)) * pred_target_q
                            pred_q = rewards + pred_rd + qs
                            est_target_q.append(pred_q)

                    if hori < self.horizon - 1:
                        preds = self.world_model.pred_next_states(
                            pred_all_next_obs[stat], pred_a)
                        (_, pred_all_next_ob, _, _) = preds
                        temp_disc_rewards = torch.stack(temp_disc_rewards)
                        pred_all_next_rewards_list.append(temp_disc_rewards)
                        pred_all_next_next_obs.append(pred_all_next_ob)
                        # Predict the future.
                        pred_all_next_obs = torch.vstack(
                            pred_all_next_next_obs)
                        pred_nt_rds = torch.vstack(pred_all_next_rewards_list)

                    # Statistics of target q
                    h_0 = torch.stack(est_target_q)
                    mean_0 = torch.mean(h_0, dim=0)
                    means.append(mean_0)
                    var_0 = torch.var(h_0, dim=0)
                    var_0[torch.abs(var_0) < 0.001] = 0.001
                    var_0 = 1.0 / var_0
                    vars.append(var_0)
                all_means = torch.stack(means)
                all_vars = torch.stack(vars)
                total_vars = torch.sum(all_vars, dim=0)
                for n in range(self.horizon):
                    all_vars[n] /= total_vars
                q_target = torch.sum(all_vars * all_means, dim=0)

            if (not self.use_mve_steve) and (not self.use_mve_critic):
                next_actions, next_log_pi, _ = self.actor_net.sample(
                    next_states)
                target_q_one, target_q_two = self.target_critic_net(
                    next_states, next_actions
                )
                target_q_values = (
                        torch.minimum(target_q_one, target_q_two)
                        - self.alpha * next_log_pi
                )
                q_target = rewards + self.gamma * (1 - dones) * target_q_values

        q_values_one, q_values_two = self.critic_net(states, actions)

        critic_loss_one = F.mse_loss(q_values_one, q_target)
        critic_loss_two = F.mse_loss(q_values_two, q_target)
        critic_loss_total = critic_loss_one + critic_loss_two

        # Update the Critic
        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        ###     Updating the actor   ###

        if self.use_mve_actor:
            pred_xs = [states]
            u_s = []
            log_p_us = []
            first_log_p = 0
            for h in range(self.horizon):
                # Roll out
                _, sample_ut, log_p_ut, _ = self.actor_net.forward(states)
                if h == 0:
                    first_log_p = log_p_ut
                unnormalized_mean1, _, _, _ = self.world_model.pred_next_states(
                    states, sample_ut)
                # Add to list
                pred_xs.append(states)
                u_s.append(sample_ut)
                log_p_us.append(log_p_ut.squeeze())
                obs = unnormalized_mean1
            #    Last step    #
            _, sample_ut, log_p_ut, _ = self.actor_net.forward(states)
            u_s.append(sample_ut)
            log_p_us.append(log_p_ut.squeeze())
            #    Stacking all produced data    #
            pred_xs = torch.stack(pred_xs)
            u_s = torch.stack(u_s)
            log_p_us = torch.stack(log_p_us)
            #    Computing the loss of the Actor    #
            pred_v = 0
            for i in range(self.horizon):
                q_1, q_2 = self.critic_net.forward(pred_xs[i, :], u_s[i,
                                                                  :])
                # all_obs : [3, 64], us: [3, 64]
                # V = Q -alpha * log
                v_min = torch.min(q_1, q_2).reshape(
                    self.batch_size) - self.alpha.detach() * log_p_us[i, :]
                pred_v += v_min.sum()
            actor_loss = -1 * pred_v
        else:
            pi, log_pi, _ = self.actor_net.sample(states)
            qf1_pi, qf2_pi = self.critic_net(states, pi)
            min_qf_pi = torch.minimum(qf1_pi, qf2_pi)
            actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        # Update the Actor
        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        # update the temperature
        alpha_loss = -(self.log_alpha * (
                log_pi + self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        if self.learn_counter % self.policy_update_freq == 0:
            for target_param, param in zip(
                    self.target_critic_net.parameters(),
                    self.critic_net.parameters()
            ):
                target_param.data.copy_(
                    param.data * self.tau + target_param.data * (
                            1.0 - self.tau)
                )

        info["q_target"] = q_target
        info["q_values_one"] = q_values_one
        info["q_values_two"] = q_values_two
        info["q_values_min"] = torch.minimum(q_values_one, q_values_two)
        info["critic_loss_total"] = critic_loss_total
        info["critic_loss_one"] = critic_loss_one
        info["critic_loss_two"] = critic_loss_two
        info["actor_loss"] = actor_loss

        return info

    def dyna_generate_and_train(self, transitions, on_policy=False):
        states, actions, rewards, next_states, not_dones, _, _ = transitions
        pred_states = [states]
        pred_actions = [actions]
        pred_rewards = [rewards]
        pred_next_states = [next_states]
        pred_not_dones = [not_dones]
        pred_state = next_states
        for _ in range(self.horizon):
            ###    Rewards   ###
            pred_action = []
            for _ in range(states.shape[0]):
                if on_policy:
                    pred_act, _, _, _ = self.actor.forward(pred_state)
                else:
                    pred_act = self.env.action_space.sample()
                pred_action.append(pred_act)
            pred_action = torch.FloatTensor(np.array(pred_action)).to(device)
            ###    Predictions    ###
            pred_next_state, _, means, stds = self.world_model.predict_next_states(
                pred_state, pred_action)
            pred_reward, _ = self.world_model.predict_rewards(pred_state,
                                                              pred_action)
            ###    Append    ###
            pred_states.append(pred_state)
            pred_actions.append(pred_action)
            pred_rewards.append(pred_reward.detach())
            pred_next_states.append(pred_next_state.detach())
            ###    Move on to the next    ###
            # pred_state = pred_next_state.detach()
        pred_states = torch.vstack(pred_states)
        pred_actions = torch.vstack(pred_actions)
        pred_rewards = torch.vstack(pred_rewards)
        pred_next_states = torch.vstack(pred_next_states)
        pred_not_dones = (torch.FloatTensor(np.ones(pred_rewards.shape)).
                          to(device))

        pred_not_dones[:self.batch_size] = not_dones
        # states, actions, rewards, next_states, not_dones
        self.train_policy((pred_states, pred_actions, pred_rewards,
                           pred_next_states, pred_not_dones, None, None))

    def train_world_model(self, statistics, transitions):
        self.world_model.set_statistics(statistics)
        states, actions, rewards, next_states, _, next_actions, next_rewards = (
            transitions)
        # mask the nones and zeros out.
        ok_masks = []
        for i in range(len(states)):
            if torch.sum(next_actions[i]) == 0 or next_rewards[i] == np.inf:
                ok_masks.append(False)
            else:
                ok_masks.append(True)
        states = states[ok_masks]
        actions = actions[ok_masks]
        rewards = rewards[ok_masks]
        next_states = next_states[ok_masks]
        next_actions = next_actions[ok_masks]
        next_rewards = next_rewards[ok_masks]
        self.world_model.train_world(states, actions, rewards,
                                     next_states, next_actions, next_rewards)

    def save_models(self, filename, filepath="models"):
        path = f"{filepath}/models" if filepath != "models" else filepath
        dir_exists = os.path.exists(path)
        if not dir_exists:
            os.makedirs(path)
        torch.save(self.actor_net.state_dict(), f"{path}/{filename}_actor.pth")
        torch.save(self.critic_net.state_dict(),
                   f"{path}/{filename}_critic.pth")
        logging.info("models has been saved...")

    def load_models(self, filepath, filename):
        path = f"{filepath}/models" if filepath != "models" else filepath
        self.actor_net.load_state_dict(
            torch.load(f"{path}/{filename}_actor.pth"))
        self.critic_net.load_state_dict(
            torch.load(f"{path}/{filename}_critic.pth"))
        logging.info("models has been loaded...")
