from cares_reinforcement_learning.memory import *
from cares_reinforcement_learning.util import helpers as hlp
from cares_reinforcement_learning.util import Record, Plot as plt

import time
import gym
import logging
import random


def evaluate_ppo_network(env, agent, args):
    evaluation_seed = args["evaluation_seed"]
    max_steps_evaluation = args["max_steps_evaluation"]
    if max_steps_evaluation == 0:
        return

    min_action_value = env.action_space.low[0]
    max_action_value = env.action_space.high[0]

    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0

    env = gym.make(env.spec.id, render_mode="human")
    state, _ = env.reset(seed=evaluation_seed)

    for total_step_counter in range(int(max_steps_evaluation)):
        episode_timesteps += 1
        action, log_prob = agent.select_action_from_policy(state)
        action_env = hlp.denormalize(action, max_action_value, min_action_value)

        state, reward, done, truncated, _ = env.step(action_env)
        episode_reward += reward

        if done or truncated:
            logging.info(
                f" Evaluation Episode {episode_num + 1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}")
            # Reset environment
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

def ppo_train(env, agent, args):
    start_time = time.time()

    seed = args["seed"]
    max_steps_training = args["max_steps_training"]
    max_steps_per_batch = args["max_steps_per_batch"]

    min_action_value = env.action_space.low[0]
    max_action_value = env.action_space.high[0]

    episode_timesteps = 0
    episode_num = 0
    episode_reward = 0
    time_step = 1

    memory = MemoryBuffer()

    state, _ = env.reset(seed=seed)
    historical_reward = {"step": [], "episode_reward": []}

    record = Record(networks={'agent':agent}, config={})
    plot = plt.Plot()

    for total_step_counter in range(int(max_steps_training)):
        episode_timesteps += 1

        action, log_prob = agent.select_action_from_policy(state)
        action_env = hlp.denormalize(action, max_action_value, min_action_value)

        next_state, reward, done, truncated, _ = env.step(action_env)
        memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done, log_prob=log_prob)

        state = next_state
        episode_reward += reward

        if time_step % max_steps_per_batch == 0:
            experience = memory.flush()
            agent.train_policy((
                experience['state'],
                experience['action'],
                experience['reward'],
                experience['next_state'],
                experience['done'],
                experience['log_prob']
            ))

        time_step += 1

        if done or truncated:
            record.log(
                steps = total_step_counter + 1,
                episode= episode_num + 1, 
                epi_timesteps=episode_timesteps,
                reward= episode_reward
            )

            plot.post(episode_reward)

            # Reset environment
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

    record.save()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Triaining time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    plot.plot()
    plot.save_plot()