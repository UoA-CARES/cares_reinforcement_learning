import numpy as np
import torch
import random
from cares_reinforcement_learning.memory import PrioritizedReplayBuffer
#
# self-imitation learning
class SelfImitation:
    def __init__(self, actor=None, critic=None, optimizer_actor=None, optimizer_critic=None, gamma=None, clip=None,cuda=None):
        self.actor = actor  # The actor network
        self.critic = critic  # The critic network
        self.optimizer_actor = optimizer_actor  # The optimizer for the actor
        self.optimizer_critic = optimizer_critic  # The optimizer for the critic
        self.buffer = PrioritizedReplayBuffer() # The replay buffer (Prioritized Experience Replay)
        self.batch_size =  256 # batch_size  # Batch size
        self.gamma = gamma  # Discount factor
        self.delayed_policy_update = 2  # Number of steps before updating the policy
        self.learn_counter = 0
        self.mini_batch_size = 64  # Mini batch size for training
        self.device = torch.device
        self.sil_beta = 0.4  # Beta parameter for prioritized sampling
        self.clip = clip  # Clipping parameter for advantages
        self.total_steps = []
        self.total_rewards = []
       
       
    
    
    # add the batch information into it...
    def step(self, obs, action, reward, next_obs, done):
        episode_data = [obs, action, reward, next_obs, done]

        if done:
            self.update_buffer([episode_data])
                        
    
    def train(self):
        
        obs, actions, returns, next_obs, dones, idxes, weights = self.sample_batch(self.batch_size)
        mean_adv=None 
        num_valid_samples=None
        if obs is not None:
                # Convert into tensor
            # Define the device (e.g., 'cuda' for GPU or 'cpu' for CPU)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Convert NumPy arrays to PyTorch tensors on the specified device
            obs = torch.tensor(np.array(obs), dtype=torch.float32, device=device)
            actions = torch.tensor(actions, dtype=torch.float32, device=device)
            returns = torch.tensor(returns, dtype=torch.float32, device=device)
            next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32, device=device)
            dones = torch.tensor(dones, dtype=torch.long, device=device)
            weights = torch.tensor(weights, dtype=torch.float32, device=device)

             # Reshape if necessary, keeping dimensions consistent
            returns = returns.view(-1, 1)  # Ensure rewards are in shape [batch_size, 1]
            dones = dones.view(-1, 1)  # Ensure dones are in shape [batch_size, 1]
            weights = weights.view(-1, 1)  # Ensure weights are in shape [batch_size, 1]

            # Batch size from the states
            self.batch_size = len(obs)

            # --- Compute Q-values using Twin Critics ---
            q1, q2 = self.critic(obs, actions)
            value = torch.min(q1, q2)
            advantages = returns - value
            advantages = advantages.detach()

            # # --- Create masks for positive advantages ---
            # masks = (advantages > 0).float()
            # masks = masks.unsqueeze(-1)

            # num_valid_samples = masks.sum().item()
            # num_samples = max(num_valid_samples, self.mini_batch_size)  # always at least mini_batch_size = 64
            # --- Compute Masks for Positive Advantages ---
            masks = (advantages > 0).float()  # Positive advantages are masked as 1
            num_valid_samples = masks.sum().item()
            num_samples = max(num_valid_samples, self.mini_batch_size)  # At least mini_batch_size
                    # process the mask
            #masks = torch.tensor(masks, dtype=torch.float32)
            # Move masks to the correct device
            
            #If masks is a NumPy array or list
            masks = masks.clone().detach().to(dtype=torch.float32, device=device)

            # --- Clip advantages ---
            clipped_advantages = torch.clamp(advantages, 1e-5, self.clip)
            mean_adv = clipped_advantages.sum() / num_samples
            mean_adv = mean_adv.item()

            # --- Compute Action Loss (SIL for TD3) ---
            actor_actions = self.actor(obs)
            # print("obs shape:", obs.shape)
            if isinstance(actor_actions, tuple):
                 actor_actions = actor_actions[0]
                 
            # Fix shapes if necessary
            if weights.dim() == 1:
                weights = weights.view(-1, 1)
            if masks.dim() == 1:
                masks = masks.view(-1, 1)

            # --- Compute Value Loss ---
            delta = torch.clamp(value - returns, -self.clip, 0) * masks.squeeze(-1)
            delta = delta.detach()

            value_loss = (weights.squeeze(-1) * value.squeeze(-1) * delta).sum() / num_samples

            total_loss = value_loss  # Actor loss optimized separately
            # print(f"total_loss: {total_loss}")
            # print(f"advantages:{clipped_advantages}")

            # # --- Optimize Critic ---
            self.optimizer_critic.zero_grad()
            total_loss.backward(retain_graph=True)
            self.optimizer_critic.step()
             
            actor_loss = ((actor_actions - actions).pow(2) * weights * masks).mean()
            #actor_loss = -torch.min(q1, q2).mean()
            # print(f"actor_loss: {actor_loss}")

            # --- Optimize Actor (delayed updates) ---
            # #if self.learn_counter  % self.delayed_policy_update == 0:
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            # Update Priorities in Buffer
            self.buffer.update_priorities(idxes, clipped_advantages.cpu().numpy().flatten())
        return mean_adv, num_valid_samples
    
    # update buffer
    def update_buffer(self, trajectory):
        positive_reward = False
        for (ob, a, r, next_ob, d) in trajectory:
            if r > 0:
                positive_reward = True
                break
        if positive_reward:
            self.add_episode(trajectory)
            self.total_steps.append(len(trajectory))
            self.total_rewards.append(np.sum([x[2] for x in trajectory]))
            # while np.sum(self.total_steps) > self.args.capacity and len(self.total_steps) > 1:
            #     self.total_steps.pop(0)
            #     self.total_rewards.pop(0)

    def add_episode(self, trajectory):
        obs = []
        actions = []
        rewards = []
        next_obs = []
        dones = []
        next_obs = []
        for (ob, action, reward, next_ob, done) in trajectory:
            if ob is not None:
                obs.append(ob)
            else:
                obs.append(None)
            actions.append(action)
            rewards.append(np.sign(reward))
            dones.append(done)
            next_obs.append(next_ob)
        dones[len(dones) - 1] = True
        returns = self.discount_with_dones(rewards, dones, self.gamma)
        for (ob, action, R, next_ob, done) in list(zip(obs, actions, returns, next_obs, dones)):
            self.buffer.add(ob, action, R, next_ob, done)

    def fn_reward(self, reward):
        return np.sign(reward)

    def get_best_reward(self):
        if len(self.total_rewards) > 0:
            return np.max(self.total_rewards)
        return 0
    
    def num_episodes(self):
        return len(self.total_rewards)

    def num_steps(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        if len(self.buffer) > self.mini_batch_size:
            batch_size = min(batch_size, len(self.buffer))
            return self.buffer. sample_priority(batch_size)
        else:
            return None, None, None, None, None,None,None
   
        
    def discount_with_dones(self, rewards, dones, gamma):
        discounted = []
        r = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            r = reward + gamma * r * (1. - done)
            discounted.append(r)
        return discounted[::-1]
  