"""
This script demonstrates the basic usage for the cares reinforcement learning package.

This illustrates how minimal code is required to apply reinforcement learning via our package to a given environment.
"""

from cares_reinforcement_learning.algorithm.policy import TD3
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.networks.TD3 import Actor, Critic
from cares_reinforcement_learning.util import helpers as hlp, Record

import gymnasium as gym
import torch

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("Working with GPU")
else:
    DEVICE = torch.device('cpu')
    print("Working with CPU")

BUFFER_CAPACITY = 100_000

GAMMA = 0.995
TAU = 0.005

ACTOR_LR = 1e-4
CRITIC_LR = 1e-3

EPISODE_NUM = 10
BATCH_SIZE = 64

MAX_STEPS_TRAINING = 1_000_000
MAX_STEPS_EXPLORATION = 10_000
G = 10


env = gym.make('Pendulum-v1', g=9.81)


def main():

    record = Record()

    observation_size = env.observation_space.shape[0]
    action_num = env.action_space.shape[0]

    memory = MemoryBuffer(BUFFER_CAPACITY)

    actor_network = Actor(observation_size, action_num)
    critic_network = Critic(observation_size, action_num)

    td3 = TD3(
        actor_network=actor_network,
        critic_network=critic_network,
        gamma=GAMMA,
        tau=TAU,
        action_num=action_num,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        device=DEVICE
    )

    print(f"Training Beginning")
    train(td3, memory, record)


def train(td3: TD3, memory: MemoryBuffer, record: Record):

    episode_num = 1

    for total_step_counter in range(int(MAX_STEPS_TRAINING)):
        episode_timesteps += 1

        if total_step_counter < MAX_STEPS_EXPLORATION:
            print(
                f"Running Exploration Steps {total_step_counter + 1}/{MAX_STEPS_EXPLORATION}"
            )

            action_env = env.action_space.sample()
        else:
            action = td3.select_action_from_policy(state)

            action_env = hlp.denormalize(
                action, env.max_action_value, env.min_action_value
            )

        next_state, reward, done, truncated = env.step(action_env)

        memory.add(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward  # Note we only track the extrinsic reward for the episode for proper comparison

        if total_step_counter >= MAX_STEPS_EXPLORATION:
            for _ in range(G):
                experience = memory.sample(BATCH_SIZE)
                td3.train_policy(experience)

        if done or truncated:
            record.log_train(
                total_steps=total_step_counter + 1,
                episode=episode_num,
                episode_steps=episode_timesteps,
                episode_reward=episode_reward,
                display=True,
            )

            print(f'Episode: {episode_num} | Reward: {episode_reward} | Steps: {episode_timesteps}')

            # Reset environment
            state = env.reset()
            episode_timesteps = 0
            episode_reward = 0
            episode_num += 1

if __name__ == '__main__':
    main()