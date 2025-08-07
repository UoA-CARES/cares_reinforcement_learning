import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import MultivariateNormal

from cares_reinforcement_learning.memory import MemoryFactory


class RePPO:
    
    def __init__(
        self,
        actor_network: torch.nn.Module,
        critic_network: torch.nn.Module,
        gamma: float,
        updates_per_iteration: int,
        eps_clip: float,
        action_num: int,
        actor_lr: float,
        critic_lr: float,
        device: torch.device,
    ):
        self.actor_net = actor_network.to(device)
        self.critic_net = critic_network.to(device)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.device = device
        self.actor_net_optimiser = torch.optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.critic_net_optimiser = torch.optim.Adam(self.critic_net.parameters(), lr=critic_lr)
        self.updates_per_iteration = updates_per_iteration
        self.action_num = action_num
        self.cov_var = torch.full(size=(self.action_num,), fill_value=0.5).to(
            self.device
        )
        self.cov_mat = torch.diag(self.cov_var)
        
    def _calculate_log_prob(self, state, action):
        # Handle state input
        if isinstance(state, tuple):  # e.g., (obs, info)
            state = state[0]
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if isinstance(state, torch.Tensor) and state.ndim == 1:
            state = state.unsqueeze(0)

        # Handle action input
        if isinstance(action, tuple):
            action = action[0]
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action).to(self.device)
        if isinstance(action, torch.Tensor) and action.ndim == 1:
            action = action.unsqueeze(0)

        # Forward through actor net
        # mean, std = self.actor_net(state)
        mean = self.actor_net(state)
        dist = MultivariateNormal(mean, self.cov_mat)

        log_prob = dist.log_prob(action)
        return log_prob
    
    
    # def _calculate_log_prob(
    #     self, state: torch.Tensor, action: torch.Tensor
    # ) -> torch.Tensor:
    #     self.actor_net.eval()
    #     with torch.no_grad():
    #         mean = self.actor_net(state)

    #         dist = MultivariateNormal(mean, self.cov_mat)
    #         log_prob = dist.log_prob(action)

    #     self.actor_net.train()
    #     return log_prob


    def select_action_from_policy(
        self, state: np.ndarray, evaluation: bool = False
    ) -> np.ndarray:
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)

            # mean,std = self.actor_net(state_tensor)
            mean = self.actor_net(state_tensor)
            dist = MultivariateNormal(mean, self.cov_mat)

            # Sample an action from the distribution and get its log prob
            sample = dist.sample()

            action = sample.cpu().data.numpy().flatten()

        self.actor_net.train()

        return action


    # def _evaluate_policy(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
   
    #     # Get the value from the critic
    #     values = self.critic(state).squeeze()

    #     # Get the action distribution from the actor
    #     mean, std = self.actor(state)
    #     dist = torch.distributions.Normal(mean, std)

    #     # Log probability of the taken action under the current policy
    #     log_probs = dist.log_prob(action).sum(axis=-1)

    #     return values, log_probs
    def _evaluate_policy(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.critic_net(state).squeeze()  # shape 5000
        # mean,std = self.actor_net(state)  # shape, 5000, 1
        mean = self.actor_net(state)  # shape, 5000, 1
        dist = MultivariateNormal(mean, self.cov_mat)
        log_prob = dist.log_prob(action)  # shape, 5000
        return v, log_prob

    def compute_advantages(
        self, rewards: torch.Tensor, dones: torch.Tensor,
        values: torch.Tensor, next_values: torch.Tensor
    ) -> torch.Tensor:
        returns = rewards + self.gamma * next_values * (1 - dones)
        advantages = returns - values
        return advantages
    
    def _calculate_rewards_to_go(
        self, batch_rewards: torch.Tensor, batch_dones: torch.Tensor
    ) -> torch.Tensor:
        rtgs: list[float] = []
        discounted_reward = 0
        for reward, done in zip(reversed(batch_rewards), reversed(batch_dones)):
            discounted_reward = reward + self.gamma * (1 - done) * discounted_reward
            rtgs.insert(0, discounted_reward)
        batch_rtgs = torch.tensor(rtgs, dtype=torch.float).to(self.device)  # shape 5000
        return batch_rtgs


    def train_policy(self, memory:  MemoryFactory, batch_size: int = 0) -> None:
        # Sample from memory
    
        experiences = memory.short_term_memory.flush()
        states, actions, rewards, next_states, dones, episode_nums, episode_steps = experiences

        # Convert to tensors and move to device
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.LongTensor(dones).to(self.device)
        #old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        log_probs_tensor = self._calculate_log_prob(states, actions)
        
        # compute reward to go:
        rtgs = self._calculate_rewards_to_go(rewards_tensor, dones_tensor)
        # rtgs = (rtgs - rtgs.mean()) / (rtgs.std() + 1e-7)

        # calculate advantages
        v, _ = self._evaluate_policy(states_tensor, actions_tensor)

        advantages = rtgs.detach() - v.detach()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        td_errors = torch.abs(advantages).data.cpu().numpy()

        for _ in range(self.updates_per_iteration):
            v, curr_log_probs = self._evaluate_policy(states_tensor, actions_tensor)

            # Calculate ratios
            ratios = torch.exp(curr_log_probs - log_probs_tensor.detach())

            # Finding Surrogate Loss
            surrogate_lose_one = ratios * advantages
            surrogate_lose_two = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # final loss of clipped objective PPO
            actor_loss = (-torch.minimum(surrogate_lose_one, surrogate_lose_two)).mean()
            critic_loss = F.mse_loss(v, rtgs)

            self.actor_net_optimiser.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_net_optimiser.step()

            self.critic_net_optimiser.zero_grad()
            critic_loss.backward()
            self.critic_net_optimiser.step()


        # # Compute values and advantages
        # with torch.no_grad():
        #     values = self.critic(states).squeeze()
        #     next_values = self.critic(next_states).squeeze()
        #     advantages = self.compute_advantages(rewards, dones, values, next_values)
        #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)  # normalize

        # # PPO update loop
        # for _ in range(self.updates_per_iteration):
        #     # Re-evaluate policy: get new log_probs and value estimates
        #     new_values, log_probs = self._evaluate_policy(states, actions)

        #     # Compute ratio (new / old policy)
        #     ratios = torch.exp(log_probs - old_log_probs.detach())

        #     # Surrogate loss
        #     surrogate1 = ratios * advantages
        #     surrogate2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

        #     # Actor loss
        #     actor_loss = -torch.min(surrogate1, surrogate2).mean()

        #     # Update actor
        #     self.actor_optimizer.zero_grad()
        #     actor_loss.backward()
        #     self.actor_optimizer.step()

        #     # Recompute target values for critic loss
        #     with torch.no_grad():
        #         next_values = self.critic(next_states).squeeze()
        #         target_values = rewards + self.gamma * next_values * (1 - dones)

        #     # Critic loss
        #     critic_loss = F.mse_loss(new_values, target_values)

        #     # Update critic
        #     self.critic_optimizer.zero_grad()
        #     critic_loss.backward()
        #     self.critic_optimizer.step()

    def save_models(self, filename: str, filepath: str = "models"):
        path = f"{filepath}/models" if filepath != "models" else filepath
        dir_exists = os.path.exists(path)

        if not dir_exists:
            os.makedirs(path)

        torch.save(self.actor_net.state_dict(), f"{path}/{filename}_actor.pht")
        torch.save(self.critic_net.state_dict(), f"{path}/{filename}_critic.pht")
        logging.info("models has been saved...")

    def load_models(self, filepath: str, filename: str):
        path = f"{filepath}/models" if filepath != "models" else filepath

        self.actor_net.load_state_dict(torch.load(f"{path}/{filename}_actor.pht"))
        self.critic_net.load_state_dict(torch.load(f"{path}/{filename}_critic.pht"))
        logging.info("models has been loaded...")
