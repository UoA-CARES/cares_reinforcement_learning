"""
This script demonstrates the basic usage for the cares reinforcement learning package.

This illustrates how minimal code is required to apply reinforcement learning via our package to a given environment.
"""

from cares_reinforcement_learning.algorithm.policy import TD3
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.networks.TD3 import Actor, Critic
from cares_reinforcement_learning.util import helpers as hlp, Record
from cares_reinforcement_learning.util.configurations import TD3Config

import gymnasium as gym

env = gym.make("Pendulum-v1", g=9.81)


def main():

    config = TD3Config()

    observation_size = env.observation_space.shape[0]
    action_num = env.action_space.shape[0]

    memory = MemoryBuffer(config.buffer_size)

    actor_network = Actor(observation_size, action_num)
    critic_network = Critic(observation_size, action_num)

    td3 = TD3(
        actor_network=actor_network,
        critic_network=critic_network,
        config=config,
    )

    record = Record(
        base_directory="logs",
        algorithm="TD3",
        task="Pendulum",
        agent=td3,
    )

    print(f"Training Beginning")
    train(td3, memory, record)


def train(td3: TD3, memory: MemoryBuffer, record: Record, config: TD3Config):

    episode_num = 1
    episode_timesteps = 0
    episode_reward = 0

    state, _ = env.reset()

    for total_step_counter in range(int(config.max_steps_training)):
        episode_timesteps += 1

        if total_step_counter < config.max_steps_exploration:
            print(
                f"Running Exploration Steps {total_step_counter + 1}/{config.max_steps_exploration}"
            )

            action_env = env.action_space.sample()
            action = hlp.normalize(
                action_env, env.action_space.high[0], env.action_space.low[0]
            )
        else:
            action = td3.select_action_from_policy(state)

            action_env = hlp.denormalize(
                action, env.action_space.high[0], env.action_space.low[0]
            )

        next_state, reward, done, truncated, _ = env.step(action_env)

        memory.add(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if total_step_counter >= config.max_steps_exploration:
            for _ in range(config.G):
                experience = memory.sample(config.batch_size)
                td3.train_policy(experience)

        if done or truncated:
            record.log_train(
                total_steps=total_step_counter + 1,
                episode=episode_num + 1,
                episode_steps=episode_timesteps,
                episode_reward=episode_reward,
                display=True,
            )

            print(
                f"Episode: {episode_num} | Reward: {episode_reward} | Steps: {episode_timesteps}"
            )

            # Reset environment
            state, _ = env.reset()
            episode_timesteps = 0
            episode_reward = 0
            episode_num += 1


if __name__ == "__main__":
    main()
