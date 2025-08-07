import numpy as np
import torch
import random
from cares_reinforcement_learning.memory import PrioritizedReplayBuffer
from torch.distributions import Categorical
#
# self-imitation learning
class SelfImitation:
    def __init__(self, actor,critic,  optimizer_actor, optimizer_critic, ac_ph, fn_reward=None, fn_obs=None,
                 n_env=1, batch_size=512, n_update=4,
                 clip=1, w_value=0.01, w_entropy=0.01,
                 max_steps=int(1e6), gamma=0.99,
                 max_nlogp=5, min_batch_size=64,
                 alpha=0.6, beta=0.1):

        self.fn_reward = fn_reward
        self.fn_obs = fn_obs
        self.model_qf1 = model_qf1
        self.model_qf2 = model_qf2
        self.model_pif = model_pif
        self.model_ob = model_ob

        self.beta = beta
        self.buffer = PrioritizedReplayBuffer(max_capacity=max_steps, beta=alpha)
        self.n_env = n_env
        self.batch_size = batch_size
        self.n_update = n_update
        self.clip = clip
        self.w_loss = 1.0
        self.w_value = w_value
        self.w_entropy = w_entropy
        self.max_steps = max_steps
        self.gamma = gamma
        self.max_nlogp = max_nlogp
        self.min_batch_size = min_batch_size

        self.train_count = 0
        self.update_count = 0
        self.total_steps = []
        self.total_rewards = []
        self.running_episodes = [[] for _ in range(n_env)]

        self.A = ac_ph
        self.build_loss_op()
        self.returns = torch.empty(0, dtype=torch.float32)
        self.weights = torch.empty(0, dtype=torch.float32)
        
    
    
        
    # add the batch information into it...
    def step(self, obs, actions, rewards, next_obs, dones):
        for n in range(self.args.num_processes):
            self.running_episodes[n].append([obs[n], actions[n], rewards[n], next_obs[n], dones[n]])
        # to see if can update the episode...
        for n, done in enumerate(dones):
            if done:
                self.update_buffer(self.running_episodes[n])
                self.running_episodes[n] = []
    
    def set_loss_weight(self, w):
        self.w_loss = w
                    
    def build_loss_op(self):
        if self.returns.numel() == 0 or self.weights.numel() == 0:
            raise ValueError("Loss inputs must be set using set_loss_inputs() before calling build_loss_op().")
        
        # Mask calculation
        mask_1 = (self.returns - self.model_qf1 > 0).float()
        mask_2 = (self.returns - self.model_qf2 > 0).float()
        
        num_valid_samples = (mask_1.sum() + mask_2.sum()) / 2
        num_samples = torch.maximum(num_valid_samples, torch.tensor(self.min_batch_size, dtype=torch.float32))
        
        # Advantage computation
        self.adv = 0.5 * (torch.clamp(self.returns - self.model_qf1, 0.0, self.clip).detach() +
                           torch.clamp(self.returns - self.model_qf2, 0.0, self.clip).detach())
        self.mean_adv = self.adv.sum() / num_samples
        
        # Value update
        v_target = self.returns
        v_estimate_1 = self.model_qf1.squeeze()
        v_estimate_2 = self.model_qf2.squeeze()
        
        delta_1 = torch.clamp(v_estimate_1 - v_target, -self.clip, 0) * mask_1
        delta_2 = torch.clamp(v_estimate_2 - v_target, -self.clip, 0) * mask_2
        
        self.vf_loss = (self.weights * v_estimate_1 * delta_1.detach()).sum() / num_samples
        self.vf_loss += (self.weights * v_estimate_2 * delta_2.detach()).sum() / num_samples
        
        self.loss = 0.5 * self.w_loss * self.vf_loss
        
        return self.loss
    
    def build_train_op(self, params, optim, max_grad_norm=0.5):
        optim.zero_grad()  # Clear previous gradients
        self.loss.backward()  # Backpropagate the loss
        
        if max_grad_norm is not None:
            # Clip gradients globally if necessary
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        
        optim.step()  # Update the parameters
    
    
    def _train(self):
        experiences = memory.sample_priority(self.batch_size)
        obs, actions, returns, next_states, dones, indices, weights = experiences
        if obs is None:
            return 0, 0, 0
        
        self.set_loss_inputs(returns.reshape(-1, 1), weights.reshape(-1, 1))
        loss = self.build_loss_op()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        adv = self.adv.detach()
        mean_adv = self.mean_adv.detach()
        num_valid_samples = (self.returns - self.model_qf1 > 0).float().sum() / 2 + \
                            (self.returns - self.model_qf2 > 0).float().sum() / 2
        
        self.buffer.update_priorities(idxes, adv)
        
        return loss.item(), mean_adv, num_valid_samples
            
    def train(self):
        if self.n_update == 0:
            return 0, 0, 0

        self.train_count += 1
        loss, adv, samples = 0, 0, 0
        if self.n_update < 1:
            update_ratio = int(1 / self.n_update + 1e-8)
            if self.train_count % update_ratio == 0:
                loss, adv, samples = self._train()
        else:  # n_update > 1
            for n in range(int(self.n_update)):
                loss, adv, samples = self._train()

        return loss, adv, samples
    
    
    # train the sil model...
    def train_sil_model(self):
        for n in range(self.args.n_update):
            experiences = memory.sample_priority(self.batch_size)
            obs, actions, returns, next_states, dones, indices, weights = experiences
            mean_adv, num_valid_samples = 0, 0
            if obs is not None:
                # need to get the masks
                # get basic information of network..
                obs = torch.tensor(obs, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(1)
                returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
                weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
                max_nlogp = torch.tensor(np.ones((len(idxes), 1)) * self.args.max_nlogp, dtype=torch.float32)
                if self.args.cuda:
                    obs = obs.cuda()
                    actions = actions.cuda()
                    returns = returns.cuda()
                    weights = weights.cuda()
                    dones = dones.cuda()
                    next_states = next_states.cuda()
                    max_nlogp = max_nlogp.cuda()
                # start to next...
                value, pi = self.network(obs)
                action_log_probs, dist_entropy =  self.evaluate_actions_sil(pi, actions)
                action_log_probs = -action_log_probs
                clipped_nlogp = torch.min(action_log_probs, max_nlogp)
                # process returns
                advantages = returns - value
                advantages = advantages.detach()
                masks = (advantages.cpu().numpy() > 0).astype(np.float32)
                # get the num of vaild samples
                num_valid_samples = np.sum(masks)
                num_samples = np.max([num_valid_samples, self.args.mini_batch_size])
                # process the mask
                masks = torch.tensor(masks, dtype=torch.float32)
                if self.args.cuda:
                    masks = masks.cuda()
                # clip the advantages...
                clipped_advantages = torch.clamp(advantages, 0, self.args.clip)
                mean_adv = torch.sum(clipped_advantages) / num_samples 
                mean_adv = mean_adv.item() 
                # start to get the action loss...
                action_loss = torch.sum(clipped_advantages * weights * clipped_nlogp) / num_samples
                entropy_reg = torch.sum(weights * dist_entropy * masks) / num_samples
                policy_loss = action_loss - entropy_reg * self.args.entropy_coef
                # start to process the value loss..
                # get the value loss
                delta = torch.clamp(value - returns, -self.args.clip, 0) * masks
                delta = delta.detach()
                value_loss = torch.sum(weights * value * delta) / num_samples
                total_loss = policy_loss + 0.5 * self.args.w_value * value_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                # update the priorities
                self.buffer.update_priorities(idxes, clipped_advantages.squeeze(1).cpu().numpy())
        return total_loss,mean_adv, num_valid_samples
    
    # update buffer
    def update_buffer(self, trajectory):
        positive_reward = False
        for (ob, a, r, nob, d) in trajectory:
            if r > 0:
                positive_reward = True
                break
        if positive_reward:
            self.add_episode(trajectory)
            self.total_steps.append(len(trajectory))
            self.total_rewards.append(np.sum([x[2] for x in trajectory]))
            while np.sum(self.total_steps) > self.args.capacity and len(self.total_steps) > 1:
                self.total_steps.pop(0)
                self.total_rewards.pop(0)

    def add_episode(self, trajectory):
        obs = []
        actions = []
        rewards = []
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
        returns = self.discount_with_dones(rewards, dones, self.args.gamma)
        for (ob, action, R, next_ob,done) in list(zip(obs, actions, returns, next_obs, dones)):
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
        if len(self.buffer) > 100:
            batch_size = min(batch_size, len(self.buffer))
            return self.buffer.sample(batch_size, beta=self.args.sil_beta)
        else:
            return None, None, None, None, None

    def discount_with_dones(self, rewards, dones, gamma):
        discounted = []
        r = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            r = reward + gamma * r * (1. - done)
            discounted.append(r)
        return discounted[::-1]
    
    def evaluate_actions_sil(self, pi, actions):
        cate_dist = Categorical(pi)
        return cate_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1), cate_dist.entropy().unsqueeze(-1)
    


class SILModule:
    def __init__(self, actor, critic, optimizer_actor, optimizer_critic, buffer, batch_size, gamma, clip, entropy_coef):
        self.actor = actor  # The actor network
        self.critic = critic  # The critic network
        self.optimizer_actor = optimizer_actor  # The optimizer for the actor
        self.optimizer_critic = optimizer_critic  # The optimizer for the critic
        self.buffer = buffer  # The replay buffer (Prioritized Experience Replay)
        self.batch_size = batch_size  # Batch size
        self.gamma = gamma  # Discount factor
        self.clip = clip  # Clipping parameter for advantages
        self.entropy_coef = entropy_coef  # Coefficient for entropy regularization

    def _train(self, lr):
        # Sample a batch of experiences from the replay buffer
        obs, actions, returns, weights, idxes = self.sample_batch(self.batch_size)
        
        if obs is None:
            return 0, 0, 0

        # Convert inputs to PyTorch tensors
        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(1)  # Add extra dimension for actions
        returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)  # Add extra dimension for returns
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)  # Add extra dimension for weights
        
        # Move to GPU if needed
        if torch.cuda.is_available():
            obs = obs.cuda()
            actions = actions.cuda()
            returns = returns.cuda()
            weights = weights.cuda()

        # Forward pass through the actor and critic networks
        pi = self.actor(obs)  # Get the action probabilities (or actions)
        value = self.critic(obs)  # Get the value estimate from the critic

        # Compute the action log probability and entropy
        action_log_probs, dist_entropy = self.evaluate_actions(pi, actions)
        
        # Clip the log probability
        clipped_nlogp = torch.min(action_log_probs, self.args.max_nlogp)

        # Compute advantages and apply clipping
        advantages = returns - value
        advantages = advantages.detach()  # Don't backprop through advantages
        
        # Mask valid samples where advantages are positive
        masks = (advantages.cpu().numpy() > 0).astype(np.float32)
        masks = torch.tensor(masks, dtype=torch.float32)
        if torch.cuda.is_available():
            masks = masks.cuda()

        # Clip advantages
        clipped_advantages = torch.clamp(advantages, 0, self.clip)
        mean_adv = clipped_advantages.mean().item()

        # Calculate action loss with weights
        action_loss = torch.sum(clipped_advantages * weights * clipped_nlogp) / advantages.size(0)
        
        # Calculate entropy regularization term
        entropy_reg = torch.sum(weights * dist_entropy * masks) / advantages.size(0)
        
        # Calculate total policy loss
        policy_loss = action_loss - self.entropy_coef * entropy_reg

        # Compute the value loss (Critic loss)
        delta = torch.clamp(value - returns, -self.clip, 0) * masks
        delta = delta.detach()
        value_loss = torch.sum(weights * value * delta) / advantages.size(0)

        # Total loss: policy loss + value loss
        total_loss = policy_loss + 0.5 * value_loss
        
        # Backpropagate and update the model (actor and critic)
        self.optimizer_actor.zero_grad()
        policy_loss.backward()
        self.optimizer_actor.step()

        self.optimizer_critic.zero_grad()
        value_loss.backward()
        self.optimizer_critic.step()

        # Update the priorities in the buffer
        self.buffer.update_priorities(idxes, clipped_advantages.squeeze(1).cpu().numpy())

        return total_loss.item(), mean_adv, advantages.size(0)

    def evaluate_actions(self, pi, actions):
        # Calculate the log probability of the actions
        cate_dist = torch.distributions.Categorical(logits=pi)
        action_log_probs = cate_dist.log_prob(actions.squeeze(-1))  # Squeeze to match the shape of actions
        entropy = cate_dist.entropy()  # Compute the entropy of the policy distribution
        return action_log_probs.unsqueeze(-1), entropy.unsqueeze(-1)

    def sample_batch(self, batch_size):
        # Sample a batch from the replay buffer
        if len(self.buffer) > 100:
            batch_size = min(batch_size, len(self.buffer))
            return self.buffer.sample(batch_size, beta=self.args.sil_beta)
        else:
            return None, None, None, None, None
