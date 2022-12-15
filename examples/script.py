"""
This is an example script that shows how one uses the cares reinforcement learning package.
To run this specific example, move the file so that it is at the same level as the package root

directory
    -- script.py
    -- summer_reinforcement_learning/
"""
from summer_reinforcement_learning.networks import TD3
from summer_reinforcement_learning.util import MemoryBuffer
from summer_reinforcement_learning.examples.Actor import Actor
from summer_reinforcement_learning.examples.Critic import Critic

import gym
import torch

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

    observation_space = env.observation_space
    action_space = env.action_space

    memory = MemoryBuffer(BUFFER_CAPACITY)

    actor = Actor(observation_space.shape[0], action_space.shape[0], ACTOR_LR, env.action_space.high)
    critic_one = Critic(observation_space.shape[0], action_space.shape[0], CRITIC_LR)
    critic_two = Critic(observation_space.shape[0], action_space.shape[0], CRITIC_LR)

    max_actions = env.action_space.high
    min_actions = env.action_space.low

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

    train(td3, memory)


def train(td3, memory: MemoryBuffer):
    historical_reward = []

    for episode in range(0, EPISODE_NUM):

        state, _ = env.reset()
        episode_reward = 0

        while True:

            # Select an Action
            td3.actor_net.eval()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                state_tensor = state_tensor.unsqueeze(0)
                state_tensor = state_tensor.to(DEVICE)
                action = td3.forward(state_tensor)
                action = action.cpu().data.numpy()
            td3.actor_net.train(True)

            action = action[0]

            next_state, reward, terminated, truncated, _ = env.step(action)

            memory.add(state, action, reward, next_state, terminated)

            experiences = memory.sample(BATCH_SIZE)

            for _ in range(0, 10):
                td3.learn(experiences)

            state = next_state
            episode_reward += reward

            if terminated or truncated:
                break

        historical_reward.append(episode_reward)
        print(f"Episode #{episode} Reward {episode_reward}")


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
