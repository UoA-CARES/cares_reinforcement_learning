"""
This is an example script that demonstrates how you go about using the package. If you've installed the module via.

    python setup.py install

then you should be able to run directly with

    python script.py

Else, move the script to the root of the package so that it is inline with the src (cares_reinforcement_learning
sub-folder) and run using the above

Notes:
    This script contains both the Actor and Critic Networks (Ideally, these should be in their own files), but to keep
    the script self-sustaining, we include them here for ease of use.
"""
from cares_reinforcement_learning.networks import TD3
from cares_reinforcement_learning.util import MemoryBuffer
from cares_reinforcement_learning.util.Plot import Plot

import gym
import torch
import torch.nn as nn
import torch.optim as optim

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("Working with GPU")
else:
    DEVICE = torch.device('cpu')
    print("Working with CPU")

BUFFER_CAPACITY = 10_000

GAMMA = 0.995
TAU = 0.005

ACTOR_LR = 1e-4
CRITIC_LR = 1e-3

EPISODE_NUM = 100
BATCH_SIZE = 64

env = gym.make('Pendulum-v1', g=9.81)


def main():
    observation_size = env.observation_space.shape[0]
    action_num = env.action_space.shape[0]

    max_actions = env.action_space.high
    min_actions = env.action_space.low

    memory = MemoryBuffer(BUFFER_CAPACITY)

    actor = Actor(observation_size, action_num, ACTOR_LR, max_actions)
    critic_one = Critic(observation_size, action_num, CRITIC_LR)
    critic_two = Critic(observation_size, action_num, CRITIC_LR)

    td3 = TD3(
        actor_network=actor,
        critic_one=critic_one,
        critic_two=critic_two,
        max_actions=max_actions,
        min_actions=min_actions,
        gamma=GAMMA,
        tau=TAU,
        device=DEVICE
    )

    print(f"Filling Buffer...")

    fill_buffer(memory)

    print(f"Buffer Filled!")

    print(f"Training Beginning")
    train(td3, memory)


def train(td3, memory: MemoryBuffer):
    plot = Plot(plot_freq=2)

    for episode in range(0, EPISODE_NUM):

        state, _ = env.reset()
        episode_reward = 0

        while True:

            # Select an Action
            action = td3.forward(state)

            next_state, reward, terminated, truncated, _ = env.step(action)

            memory.add(state, action, reward, next_state, terminated)

            experiences = memory.sample(BATCH_SIZE)

            for _ in range(0, 10):
                td3.learn(experiences)

            state = next_state
            episode_reward += reward

            if terminated or truncated:
                break

        plot.post(episode_reward)
        print(f"Episode #{episode} Reward {episode_reward}")

    plot.save_plot('New_Plot')
    plot.save_csv()
    plot.plot()


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


class Actor(nn.Module):
    def __init__(self, observation_size, num_actions, learning_rate, max_action):
        super(Actor, self).__init__()

        self.max_action = max_action

        self.hidden_size = [128, 64, 32]

        self.h_linear_1 = nn.Linear(in_features=observation_size, out_features=self.hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=self.hidden_size[0], out_features=self.hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=self.hidden_size[1], out_features=self.hidden_size[2])
        self.h_linear_4 = nn.Linear(in_features=self.hidden_size[2], out_features=num_actions)

        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = torch.relu(self.h_linear_1(state))
        x = torch.relu(self.h_linear_2(x))
        x = torch.relu(self.h_linear_3(x))
        x = torch.tanh(self.h_linear_4(x)) * self.max_action[0]
        return x


class Critic(nn.Module):
    def __init__(self, observation_size, num_actions, learning_rate):
        super(Critic, self).__init__()

        self.hidden_size = [128, 64, 32]

        self.Q1 = nn.Sequential(
            nn.Linear(observation_size + num_actions, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], self.hidden_size[2]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[2], 1)
        )

        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        q1 = self.Q1(x)
        return q1


if __name__ == '__main__':
    main()
