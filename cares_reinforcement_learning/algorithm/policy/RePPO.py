import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from cares_reinforcement_learning.memory import MemoryFactory


# class PPO:
#     def __init__(
#         self,
#         actor_network: torch.nn.Module,
#         critic_network: torch.nn.Module,
#         gamma: float,
#         updates_per_iteration: int,
#         eps_clip: float,
#         action_num: int,
#         actor_lr: float,
#         critic_lr: float,
#         device: torch.device,
#     ):
#         self.type = "policy"
#         self.actor_net = actor_network.to(device)
#         self.critic_net = critic_network.to(device)

#         self.gamma = gamma
#         self.action_num = action_num
#         self.device = device

#         self.actor_net_optimiser = torch.optim.Adam(
#             self.actor_net.parameters(), lr=actor_lr
#         )
#         self.critic_net_optimiser = torch.optim.Adam(
#             self.critic_net.parameters(), lr=critic_lr
#         )

#         self.updates_per_iteration = updates_per_iteration
#         self.eps_clip = eps_clip
#         self.cov_var = torch.full(size=(action_num,), fill_value=0.5).to(self.device)
#         self.cov_mat = torch.diag(self.cov_var)

#     def select_action_from_policy(self, state: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
#         self.actor_net.eval()
#         with torch.no_grad():
#             state_tensor = torch.FloatTensor(state).to(self.device)
#             state_tensor = state_tensor.unsqueeze(0)

#             mean, std = self.actor_net(state_tensor)
#             probs = Normal(mean, std)
            
#             # Sample an action from the distribution and get its log prob
#             action = probs.sample()
#             log_prob = probs.log_prob(action).sum(dim=-1)  # sum over action dimensions

#             action = action.cpu().numpy().flatten()
#             log_prob = log_prob.cpu().numpy().flatten()

#         self.actor_net.train()
#         return action, log_prob

#     def _evaluate_policy(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         # Compute value (v) from the critic network (squeezed to remove extra dimensions)
#         v = self.critic_net(state).squeeze()  # shape should be [batch_size]

#         # Get mean and standard deviation from the actor network
#         mean, std = self.actor_net(state)  # shape: [batch_size, num_actions]

#         # Construct the Normal distribution using mean and std
#         dist = Normal(mean, std)

#         # Compute log probability of the actions
#         log_prob = dist.log_prob(action).sum(dim=-1)  # sum over action dimensions

#         return v, log_prob

#     def _calculate_rewards_to_go(self, batch_rewards: torch.FloatTensor, batch_dones: torch.FloatTensor) -> torch.Tensor:
#         rtgs = []
#         discounted_reward = 0
#         for reward, done in zip(reversed(batch_rewards), reversed(batch_dones)):
#             discounted_reward = reward + self.gamma * (1 - done) * discounted_reward
#             rtgs.insert(0, discounted_reward)
#         batch_rtgs = torch.tensor(rtgs, dtype=torch.float).to(self.device)  # shape 5000
#         return batch_rtgs

#     def train_policy(self, memory: PrioritizedReplayBuffer, batch_size: int = 0) -> None:
#         experiences = memory.flush()
#         states, actions, rewards, next_states, dones, log_probs = experiences

#         states = torch.FloatTensor(np.asarray(states)).to(self.device)
#         actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
#         rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device)
#         next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
#         dones = torch.LongTensor(np.asarray(dones)).to(self.device)
#         log_probs = torch.FloatTensor(np.asarray(log_probs)).to(self.device)

#         log_probs = log_probs.squeeze()

#         # compute reward to go:
#         rtgs = self._calculate_rewards_to_go(rewards, dones)

#         # calculate advantages
#         v, _ = self._evaluate_policy(states, actions)

#         advantages = rtgs.detach() - v.detach()

#         # Compute TD errors (optional, useful for priority computation if needed)
#         td_errors = torch.abs(advantages).cpu().numpy()

#         for _ in range(self.updates_per_iteration):
#             v, curr_log_probs = self._evaluate_policy(states, actions)

#             # Calculate ratios
#             ratios = torch.exp(curr_log_probs - log_probs.detach())

#             # Finding Surrogate Loss
#             surrogate_lose_one = ratios * advantages
#             surrogate_lose_two = (
#                 torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
#             )

#             # final loss of clipped objective PPO
#             actor_loss = (-torch.minimum(surrogate_lose_one, surrogate_lose_two)).mean()
#             critic_loss = F.mse_loss(v, rtgs)

#             self.actor_net_optimiser.zero_grad()
#             actor_loss.backward()  # retain_graph=True)
#             self.actor_net_optimiser.step()

#             self.critic_net_optimiser.zero_grad()
#             critic_loss.backward()
#             self.critic_net_optimiser.step()

#     def save_models(self, filename: str, filepath: str = "models"):
#         path = f"{filepath}/models" if filepath != "models" else filepath
#         dir_exists = os.path.exists(path)

#         if not dir_exists:
#             os.makedirs(path)

#         torch.save(self.actor_net.state_dict(), f"{path}/{filename}_actor.pht")
#         torch.save(self.critic_net.state_dict(), f"{path}/{filename}_critic.pht")
#         logging.info("models have been saved...")

#     def load_models(self, filepath: str, filename: str):
#         path = f"{filepath}/models" if filepath != "models" else filepath

#         self.actor_net.load_state_dict(torch.load(f"{path}/{filename}_actor.pht"))
#         self.critic_net.load_state_dict(torch.load(f"{path}/{filename}_critic.pht"))
#         logging.info("models have been loaded...")

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
        self.actor = actor_network.to(device)
        self.critic = critic_network.to(device)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.device = device
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.updates_per_iteration = updates_per_iteration
        self.action_num = action_num

    def select_action_from_policy(self, state: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Select an action from the current policy given a state.

        Args:
            state (np.ndarray): The input state as a NumPy array.

        Returns:
            action (torch.Tensor): The sampled action.
            log_prob (torch.Tensor): The log probability of the sampled action.
        """
        self.actor.eval()
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _= self.actor.sample(state_tensor)
            action = action.cpu().numpy().flatten()
        self.actor.train()
        return action
    
    
    def _calculate_log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        
        # Use torch.no_grad() to prevent gradient calculation during inference
        with torch.no_grad():
            # Forward pass through the actor network
            mean, std = self.actor(states)  # Assuming the actor outputs mean and std for continuous actions
            
            # Create the action distribution (assuming a normal distribution for continuous actions)
            dist = torch.distributions.Normal(mean, std)
            
            # Calculate the log probability for the actions
            log_probs = dist.log_prob(actions)  # Log probability for each action
            
            # If multi-dimensional actions, sum log probabilities across the dimensions
            log_probs = log_probs.sum(dim=-1)  # Sum across action dimensions if needed

        return log_probs



    def _evaluate_policy(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # with torch.no_grad():
            # Get the value from the critic
        values = self.critic(state).squeeze()

            # Get the action distribution from the actor
        mean, std = self.actor(state)
        dist = torch.distributions.Normal(mean, std)

        # Log probability of the taken action under the current policy
        log_probs = dist.log_prob(action).sum(axis=-1)

        return values, log_probs

    def compute_advantages(
        self, rewards: torch.Tensor, dones: torch.Tensor,
        values: torch.Tensor, next_values: torch.Tensor
    ) -> torch.Tensor:
        returns = rewards + self.gamma * next_values * (1 - dones)
        advantages = returns - values
        return advantages


    def train_policy(self, memory:MemoryFactory, batch_size: int = 0) -> None:
        # Sample from memory
        experiences = memory.short_term_memory.flush()
        states, actions, rewards, next_states, dones, episode_nums, episode_steps = experiences
        # Convert to tensors and move to device
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.LongTensor(dones).to(self.device)
        old_log_probs = self._calculate_log_prob(states, actions)

        # Compute values and advantages
        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
            advantages = self.compute_advantages(rewards, dones, values, next_values)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)  # normalize

        # PPO update loop
        for _ in range(self.updates_per_iteration):
            # Re-evaluate policy: get new log_probs and value estimates
            new_values, log_probs = self._evaluate_policy(states, actions)

            # Compute ratio (new / old policy)
            ratios = torch.exp(log_probs - old_log_probs.detach())

            # Surrogate loss
            surrogate1 = ratios * advantages
            surrogate2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Actor loss
            actor_loss = -torch.min(surrogate1, surrogate2).mean()

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Recompute target values for critic loss
            with torch.no_grad():
                next_values = self.critic(next_states).squeeze()
                target_values = rewards + self.gamma * next_values * (1 - dones)

            # Critic loss
            critic_loss = F.mse_loss(new_values, target_values)

            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

    def save_models(self, filename: str, filepath: str = "models") -> None:
        path = f"{filepath}/models" if filepath != "models" else filepath
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{path}/{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path}/{filename}_critic.pth")
        logging.info(f"Models saved at {path}/{filename}.pth")

    def load_models(self, filename: str, filepath: str = "models") -> None:
        path = f"{filepath}/models" if filepath != "models" else filepath
        self.actor.load_state_dict(torch.load(f"{path}/{filename}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{path}/{filename}_critic.pth"))
        logging.info(f"Models loaded from {path}/{filename}.pth")

