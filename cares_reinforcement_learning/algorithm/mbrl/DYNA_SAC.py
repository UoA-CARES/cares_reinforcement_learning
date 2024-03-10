"""
Sutton, Richard S. "Dyna, an integrated architecture for learning, planning, and reacting."

This code runs automatic entropy tuning
"""

import copy
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F


class MBRL_DYNA_SAC:
    """
    Use the Soft Actor Critic as the Actor Critic framework.

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
        alpha_lr,
        num_samples,
        horizon,
        device,
    ):
        self.type = "mbrl"
        # Switches
        self.num_samples = num_samples
        self.horizon = horizon
        self.action_num = action_num
        # Other Variables
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.batch_size = None

        # this may be called policy_net in other implementations
        self.actor_net = actor_network.to(device)
        # this may be called soft_q_net in other implementations
        self.critic_net = critic_network.to(device)
        self.target_critic_net = copy.deepcopy(self.critic_net).to(device)

        # Set to initial alpha to 1.0 according to other baselines.
        self.log_alpha = torch.tensor(np.log(1.0)).to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -action_num

        # optimizer
        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=actor_lr
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=critic_lr
        )
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # World model
        self.world_model = world_network
        self.world_model.to(device)
        self.learn_counter = 0
        self.policy_update_freq = 1

    @property
    def alpha(self):
        """
        A variatble decide to what extend entropy shoud be valued.
        """
        return self.log_alpha.exp()

    # pylint: disable-next=unused-argument to keep the same interface
    def select_action_from_policy(self, state, evaluation=False, noise_scale=0):
        """
        Select a action for executing. It is the only channel that an agent
        will communicate the the actual environment.

        """
        # note that when evaluating this algorithm we need to select mu as
        # action so _, _, action = self.actor_net.sample(state_tensor)
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if evaluation is False:
                (action, _, _) = self.actor_net.sample(state_tensor)
            else:
                (_, _, action) = self.actor_net.sample(state_tensor)
            action = action.cpu().data.numpy().flatten()
        self.actor_net.train()
        return action

    def true_train_policy(self, states, actions, rewards, next_states, dones):
        """
        Train the policy with Model-Based Value Expansion. A family of MBRL.

        """
        info = {}
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor_net.sample(next_states)
            target_q_one, target_q_two = self.target_critic_net(
                next_states, next_actions
            )
            target_q_values = (
                torch.minimum(target_q_one, target_q_two) - self.alpha * next_log_pi
            )
            q_target = rewards + self.gamma * (1 - dones) * target_q_values
        q_target = q_target.detach()
        assert (len(q_target.shape) == 2) and (q_target.shape[1] == 1)

        q_values_one, q_values_two = self.critic_net(states, actions)
        critic_loss_one = F.mse_loss(q_values_one, q_target)
        critic_loss_two = F.mse_loss(q_values_two, q_target)
        critic_loss_total = critic_loss_one + critic_loss_two
        # Update the Critic
        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        ##################     Update the Actor Second     ####################
        pi, first_log_p, _ = self.actor_net.sample(states)
        qf1_pi, qf2_pi = self.critic_net(states, pi)
        min_qf_pi = torch.minimum(qf1_pi, qf2_pi)
        actor_loss = ((self.alpha * first_log_p) - min_qf_pi).mean()

        # Update the Actor
        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        # update the temperature
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

        info["q_target"] = q_target
        info["q_values_one"] = q_values_one
        info["q_values_two"] = q_values_two
        info["q_values_min"] = torch.minimum(q_values_one, q_values_two)
        info["critic_loss_total"] = critic_loss_total
        info["critic_loss_one"] = critic_loss_one
        info["critic_loss_two"] = critic_loss_two
        info["actor_loss"] = actor_loss
        return info

    def train_world_model(self, experiences):
        """
        Sample the buffer again for training the world model can reach higher rewards.

        :param experiences:
        """
        (
            states,
            actions,
            rewards,
            next_states,
            _,
            next_actions,
            next_rewards,
        ) = experiences
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        next_rewards = (
            torch.FloatTensor(np.asarray(next_rewards)).to(self.device).unsqueeze(1)
        )
        next_actions = torch.FloatTensor(np.asarray(next_actions)).to(self.device)
        assert len(states.shape) >= 2
        assert len(actions.shape) == 2
        assert len(rewards.shape) == 2 and rewards.shape[1] == 1
        assert len(next_rewards.shape) == 2 and next_rewards.shape[1] == 1
        assert len(next_states.shape) >= 2
        # # Step 1 train the world model.
        self.world_model.train_world(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            next_actions=next_actions,
            next_rewards=next_rewards,
        )

    def train_policy(self, experiences):
        """
        Interface to training loop.

        """
        self.learn_counter += 1
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = experiences
        self.batch_size = len(states)
        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.LongTensor(np.asarray(dones)).to(self.device).unsqueeze(1)
        assert len(states.shape) >= 2
        assert len(actions.shape) == 2
        assert len(rewards.shape) == 2 and rewards.shape[1] == 1
        assert len(next_states.shape) >= 2
        # Step 2 train as usual
        self.true_train_policy(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
        )
        # # # Step 3 Dyna add more data
        self.dyna_generate_and_train(next_states=next_states)

    def dyna_generate_and_train(self, next_states):
        """
        Only off-policy Dyna will work.
        :param next_states:
        """
        pred_states = []
        pred_actions = []
        pred_rs = []
        pred_n_states = []
        pred_state = next_states
        for _ in range(self.horizon):
            pred_state = torch.repeat_interleave(pred_state, self.num_samples, dim=0)
            # This part is controversial. But random actions is empirically better.
            rand_acts = np.random.uniform(-1, 1, (pred_state.shape[0], self.action_num))
            pred_acts = torch.FloatTensor(rand_acts).to(self.device)
            pred_next_state, _, _, _ = self.world_model.pred_next_states(
                pred_state, pred_acts
            )
            pred_reward, _ = self.world_model.pred_rewards(pred_state, pred_acts)
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
        self.true_train_policy(
            pred_states, pred_actions, pred_rs, pred_n_states, pred_dones
        )

    def save_models(self, filename, filepath="models"):
        """
        Save the intrim actor critics.
        """
        path = f"{filepath}/models" if filepath != "models" else filepath
        dir_exists = os.path.exists(path)
        if not dir_exists:
            os.makedirs(path)
        torch.save(self.actor_net.state_dict(), f"{path}/{filename}_actor.pth")
        torch.save(self.critic_net.state_dict(), f"{path}/{filename}_critic.pth")
        logging.info("models has been saved...")

    def load_models(self, filepath, filename):
        """
        Load trained networks
        """
        path = f"{filepath}/models" if filepath != "models" else filepath
        self.actor_net.load_state_dict(torch.load(f"{path}/{filename}_actor.pth"))
        self.critic_net.load_state_dict(torch.load(f"{path}/{filename}_critic.pth"))
        logging.info("models has been loaded...")
