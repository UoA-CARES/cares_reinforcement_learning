"""
This is an example script that shows how one uses the cares reinforcement learning package.
To run this specific example, move the file so that it is at the same level as the package root

directory
    -- sac_test.py
    -- summer_reinforcement_learning/
"""

# from cares_reinforcement_learning.networks import sac

from cares_reinforcement_learning.util import MemoryBuffer
from Gaussian_actor import GaussianPolicy
from Critic import Critic

import gym
import torch

import copy

import torch
from torch.distributions.uniform import Uniform
import numpy as np


class SAC:

    def __init__(self,
                 actor_network,
                 critic_one,
                 critic_two,
                 max_actions,
                 min_actions,
                 gamma,
                 tau,
                 alpha,
                 device):


        self.actor_net = actor_network.to(device) ##### move the network to the appropriate device
        #self.target_actor_net = copy.deepcopy(actor_network).to(device) ##### create the target actor network

        self.critic_one_net = critic_one.to(device) ##### move the network to the appropriate device
        self.target_critic_one_net = copy.deepcopy(critic_one).to(device) ##### create the target critic network

        self.critic_two_net = critic_two.to(device) ##### move the network to the appropriate device
        self.target_critic_two_net = copy.deepcopy(critic_two).to(device) ##### create the target critic network

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.max_actions = torch.FloatTensor(max_actions).to(device) ##### move the network to the appropriate device
        self.min_actions = torch.FloatTensor(min_actions).to(device) ##### move the network to the appropriate device

        self.learn_counter = 0
        self.policy_update_freq = 2  # Hard coded

        self.device = device

    ##### Single forward pass
    def forward(self, observation):
        pi_action, logp_pi = self.actor_net.forward(observation)
        return pi_action, logp_pi

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences #####
        batch_size = len(states) #####

        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states)).to(self.device) #####
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device) #####
        rewards = torch.FloatTensor(rewards).to(self.device) #####
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device) #####
        dones = torch.LongTensor(dones).to(self.device) #####

        # Reshape to batch_size x whatever
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1) ##### Append a dimension
        dones = dones.unsqueeze(0).reshape(batch_size, 1) ##### Append a dimension

        with torch.no_grad():

            # Unlike in TD3, the next-state actions used in the target come from the current policy instead of a target policy.
            # next_actions = self.target_actor_net(next_states).to(self.device) #### retrieve ā~π(.|s') from actor network
            next_actions, logp_pi_batch = self.actor_net(next_states)  #### retrieve ā~π(.|s') from the CURRENT actor network

            next_actions = next_actions.to(self.device)
            logp_pi_batch = logp_pi_batch.to(self.device).unsqueeze(1)


            # noise = Uniform(-0.5, 0.5).sample(next_actions.size()).to(self.device) # retrieve noise

            # next_actions = torch.clip(next_actions + noise, self.min_actions,
            #                           self.max_actions) # add noise to action but keep within max and min bounds

            target_q_values_one = self.target_critic_one_net(next_states, next_actions) ##### retrieve Q(s', a') from critic one
            target_q_values_two = self.target_critic_two_net(next_states, next_actions) ##### retrieve Q(s', a') from critic two

            target_q_values = torch.min(target_q_values_one, target_q_values_two) ##### use the minimum of the two Q(s', a')

            q_target = rewards + self.gamma * (1 - dones) * (target_q_values - self.alpha * logp_pi_batch)  # calculate y(r,s',d)

        q_values_one = self.critic_one_net(states, actions) # retrieve Q(s, a) from critic one
        q_values_two = self.critic_two_net(states, actions) # retrieve Q(s, a) from critic two
        # TODO: check whether the loss is divided by the batch size
        # Update the Critic One
        critic_one_loss = self.critic_one_net.loss(q_values_one, q_target) # Calculate square loss, (Q(s, a) - y(r,s',d))^2
        critic_one_loss = critic_one_loss / batch_size;
        self.critic_one_net.optimiser.zero_grad() # clear gradients
        critic_one_loss.backward() # calculate gradients
        self.critic_one_net.optimiser.step() # optimise the model parameters by performing gradient descent

        # Update Critic Two
        critic_two_loss = self.critic_two_net.loss(q_values_two, q_target) # Calculate square loss, (Q(s, a) - y(r,s',d))^2
        critic_two_loss = critic_two_loss / batch_size;
        self.critic_two_net.optimiser.zero_grad() # clear gradients
        critic_two_loss.backward() # calculate gradients
        self.critic_two_net.optimiser.step() # optimise the model parameters by performing gradient descent

        # updating the target network and the actor network
        if self.learn_counter % self.policy_update_freq == 0:
            # TODO: check whether the loss is divided by the batch size
            # Update Actor

            pi_action, logp_pi_batch = self.actor_net(states)
            logp_pi_batch = logp_pi_batch.to(self.device).unsqueeze(1)

            actor_q_one = self.critic_one_net(states, pi_action)# retrieve Q(s, a) from critic one
            actor_q_two = self.critic_two_net(states, pi_action)# retrieve Q(s, a) from critic two

            actor_q_values = torch.min(actor_q_one, actor_q_two) # Take the minimum of the Q(s,a) from critics

            actor_loss = -(actor_q_values - self.alpha * logp_pi_batch).mean() # Since the experience is a batch, then the actor Q(s,a) is averaged to find the average Q of the actor
            actor_loss = actor_loss / batch_size;
            # Update Actor network parameters
            self.actor_net.optimiser.zero_grad()
            actor_loss.backward()
            self.actor_net.optimiser.step()

            # Update target network params with an exponential filter
            for target_param, param in zip(self.target_critic_one_net.parameters(), self.critic_one_net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.target_critic_two_net.parameters(), self.critic_two_net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            # for target_param, param in zip(self.target_actor_net.parameters(), self.actor_net.parameters()):
            #     target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))



if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("Working with GPU")
else:
    DEVICE = torch.device('cpu')
    print("Working with CPU")

BUFFER_CAPACITY = 10_000

GAMMA = 0.995
TAU = 0.005
ALPHA = 0.1

ACTOR_LR = 1e-4
CRITIC_LR = 1e-3

EPISODE_NUM = 100
BATCH_SIZE = 64

env = gym.make('Pendulum-v1', g=9.81)

def main():

    observation_space = env.observation_space##########
    action_space = env.action_space##########

    memory = MemoryBuffer(BUFFER_CAPACITY)######

    actor = GaussianPolicy(observation_space.shape[0], action_space.shape[0], ACTOR_LR, env.action_space.high) ##### initiate an actor
    critic_one = Critic(observation_space.shape[0], action_space.shape[0], CRITIC_LR) ##### initiate critic 1
    critic_two = Critic(observation_space.shape[0], action_space.shape[0], CRITIC_LR) ##### initiate critic 2

    max_actions = env.action_space.high ##### retrieve maximum possible action value
    min_actions = env.action_space.low ##### retrieve minimum possible action value

    sac = SAC(
        actor_network=actor,
        critic_one=critic_one,
        critic_two=critic_two,
        max_actions=max_actions,
        min_actions=min_actions,
        gamma=GAMMA,
        tau=TAU,
        alpha=ALPHA,
        device=DEVICE
    )

    print(f"Filling Buffer...")

    fill_buffer(memory)#######

    train(sac, memory)


def train(sac, memory: MemoryBuffer):
    historical_reward = []########

    for episode in range(0, EPISODE_NUM):

        state, _ = env.reset()######
        episode_reward = 0########

        while True:

            # Select an Action
            sac.actor_net.eval() ###### setting the network to evaluation mode
            with torch.no_grad():######
                state_tensor = torch.FloatTensor(state) ##### convert the current state to a tensor
                state_tensor = state_tensor.unsqueeze(0) ###### add a new dimension to the tensor
                state_tensor = state_tensor.to(DEVICE)  ######## move the tensor to the appropriate form
                action, _ = sac.forward(state_tensor) ###### retrieve an action using the actor network in sac
                action = action.cpu().data.numpy() ######3 converts the action tensor to a NumPy array and transfers it to CPU memory
            sac.actor_net.train(True) ###### setting the network to training mode

            action = action[0] ###### retrieve the action

            next_state, reward, terminated, truncated, _ = env.step(action) ####### step

            memory.add(state, action, reward, next_state, terminated) #######3 store memory

            experiences = memory.sample(BATCH_SIZE) ####### sample memory

            for _ in range(0, 10): # Why update so frequently, how frequent should I update?
                sac.learn(experiences)

            state = next_state ########
            episode_reward += reward #########

            if terminated or truncated:#######
                break

        historical_reward.append(episode_reward)##########
        print(f"Episode #{episode} Reward {episode_reward}")###########


def fill_buffer(memory):

    while len(memory.buffer) < memory.buffer.maxlen:

        state, _ = env.reset()

        while True:

            action = env.action_space.sample()

            next_state, reward, terminated, truncated, _ = env.step(action)

            memory.add(state, action, reward, next_state, terminated)

            state = next_state

            if terminated or truncated:
                break


if __name__ == '__main__':
    main()

