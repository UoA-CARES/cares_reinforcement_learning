

import copy
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from cares_reinforcement_learning.memory import ManageBuffers



class EpisodicTD3:
    def __init__(
        self,
        actor_network: torch.nn.Module,
        critic_network: torch.nn.Module,
        gamma: float,
        tau: float,
        alpha: float,
        min_priority: float,
        prioritized_fraction: float,
        action_num: int,
        actor_lr: float,
        critic_lr: float,
        device: torch.device,
    ):
        self.type = "policy"
        self.device = device

        self.actor_net = actor_network.to(self.device)
        self.critic_net = critic_network.to(self.device)

        self.target_actor_net = copy.deepcopy(self.actor_net)
        self.target_critic_net = copy.deepcopy(self.critic_net)

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.noise_clip = 0.5
        self.policy_noise = 0.2

        self.min_priority = min_priority
        self.prioritized_fraction = prioritized_fraction

        self.learn_counter = 0
        self.policy_update_freq = 2

        self.action_num = action_num

        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=actor_lr
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=critic_lr
        )

    def select_action_from_policy(
        self, state: np.ndarray, evaluation: bool = False, noise_scale: float = 0.1
    ) -> np.ndarray:
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)
            action = self.actor_net(state_tensor)
            action = action.cpu().data.numpy().flatten()
            if not evaluation:
                # this is part the TD3 too, add noise to the action
                noise = np.random.normal(0, scale=noise_scale, size=self.action_num)
                action = action + noise
                action = np.clip(action, -1, 1)
        self.actor_net.train()
        return action

    def _update_target_network(self) -> None:
        # Update target network params
        for target_param, param in zip(
            self.target_critic_net.Q1.parameters(), self.critic_net.Q1.parameters()
        ):
            target_param.data.copy_(
                param.data * self.tau + target_param.data * (1.0 - self.tau)
            )

        for target_param, param in zip(
            self.target_critic_net.Q2.parameters(), self.critic_net.Q2.parameters()
        ):
            target_param.data.copy_(
                param.data * self.tau + target_param.data * (1.0 - self.tau)
            )

        for target_param, param in zip(
            self.target_actor_net.parameters(), self.actor_net.parameters()
        ):
            target_param.data.copy_(
                param.data * self.tau + target_param.data * (1.0 - self.tau)
            )

    def _train_actor(self, states: np.ndarray) -> None:
        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states)).to(self.device).squeeze(0)

        # Update Actor
        actor_q_value_one, actor_q_value_two = self.critic_net(states, self.actor_net(states))
        actor_q_values = torch.minimum(actor_q_value_one, actor_q_value_two)
        actor_loss = -actor_q_values.mean()

        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()
        
        # Update the target network
        self._update_target_network()
    

    def _train_critic(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        uniform_sampling: bool,
    ) -> np.ndarray:
        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states)).to(self.device).squeeze(0)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device).squeeze(0)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device).squeeze(0)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device).squeeze(0)
        dones = torch.LongTensor(np.asarray(dones)).to(self.device).squeeze(0)
        
        # Reshape to batch_size
        rewards = rewards.unsqueeze(0).reshape(len(rewards), 1)
        dones = dones.unsqueeze(0).reshape(len(dones), 1)


        with torch.no_grad():
            next_actions = self.target_actor_net(next_states)
            target_noise = self.policy_noise * torch.randn_like(next_actions)
            target_noise = torch.clamp(target_noise, -self.noise_clip, self.noise_clip)
            next_actions = next_actions + target_noise
            next_actions = torch.clamp(next_actions, min=-1, max=1)

            target_q_values_one, target_q_values_two = self.target_critic_net(
                next_states, next_actions
            )
            target_q_values = torch.minimum(target_q_values_one, target_q_values_two)

            q_target = rewards + self.gamma * (1 - dones) * target_q_values

        q_values_one, q_values_two = self.critic_net(states, actions)

        td_error_one = F.mse_loss(q_values_one, q_target)
        td_error_two = F.mse_loss(q_values_two, q_target)
        critic_loss_total = td_error_one + td_error_two

        # Update the Critic
        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()


    def train_policy(self, memory:ManageBuffers, batch_size: int) -> None:
        self.learn_counter += 1

        #uniform_batch_size = int(batch_size * (1 - self.prioritized_fraction))
        #priority_batch_size = int(batch_size * self.prioritized_fraction)
        uniform_batch_size = batch_size
        priority_batch_size = batch_size

        policy_update = self.learn_counter % self.policy_update_freq == 0

        ######################### UNIFORM SAMPLING #########################
        experiences = memory.short_term_memory.sample_random_episode(uniform_batch_size)
        states, actions, rewards, next_states, dones, episode_nums, episode_steps = experiences

        self._train_critic(
            states,
            actions,
            rewards,
            next_states,
            dones,
            uniform_sampling=True,
        )

        if policy_update:
            self._train_actor(states)

        ######################### Episodic SAMPLING #########################
        if (memory.long_term_memory.get_length() != 0):
            crucial_episodes_ids, crucial_episodes_rewards = memory.long_term_memory.sample_uniform(1)
           # print(f"crucial_episodes_ids:{crucial_episodes_ids}, crucial_episodes_rewards:{crucial_episodes_rewards}")
            for i in range(len(crucial_episodes_ids)):
                experiences = memory.episodic_memory.sample_episode(crucial_episodes_ids[i], crucial_episodes_rewards[i], priority_batch_size)
                states, actions, rewards, next_states, dones, episode_nums, episode_steps = experiences

                self._train_critic(
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones,
                    uniform_sampling=False,
                )

                if policy_update:
                    self._train_actor(states)
        
       
    def save_models(self, filename: str, filepath: str = "models") -> None:
        path = f"{filepath}/models" if filepath != "models" else filepath
        dir_exists = os.path.exists(path)

        if not dir_exists:
            os.makedirs(path)

        torch.save(self.actor_net.state_dict(), f"{path}/{filename}_actor.pht")
        torch.save(self.critic_net.state_dict(), f"{path}/{filename}_critic.pht")
        logging.info("models has been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        path = f"{filepath}/models" if filepath != "models" else filepath

        self.actor_net.load_state_dict(torch.load(f"{path}/{filename}_actor.pht"))
        self.critic_net.load_state_dict(torch.load(f"{path}/{filename}_critic.pht"))
        logging.info("models has been loaded...")
