from cares_reinforcement_learning.memory import *
from cares_reinforcement_learning.util import helpers as hlp, Record, Plot as plt

import time
import gym
import logging
import random


def evaluate_value_network(env, agent, record, args):
    evaluation_seed = args["evaluation_seed"]
    max_steps_evaluation = args["max_steps_evaluation"]
    if max_steps_evaluation == 0:
        return

    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0

    env = gym.make(env.spec.id, render_mode="human")
    state, _ = env.reset(seed=evaluation_seed)
    exploration_rate = args["exploration_min"]

    plot = plt.Plot()

    for total_step_counter in range(int(max_steps_evaluation)):
        episode_timesteps += 1

        if random.random() < exploration_rate:
            action = env.action_space.sample()
        else:
            action = agent.select_action_from_policy(state)

        state, reward, done, truncated, _ = env.step(action)
        episode_reward += reward

        if done or truncated:
            record.log(
                Eval_steps = total_step_counter + 1,
                Eval_episode= episode_num + 1, 
                Eval_timesteps=episode_timesteps,
                Eval_reward= episode_reward
            )

            plot.post(episode_reward)

            # Reset environment
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1


def value_based_train(env, agent, memory, record, args):
    start_time = time.time()

    max_steps_training = args["max_steps_training"]
    exploration_min = args["exploration_min"]
    exploration_decay = args["exploration_decay"]

    batch_size = args["batch_size"]
    seed = args["seed"]
    G = args["G"]

    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0

    state, _ = env.reset(seed=seed)
    exploration_rate = 1

    plot = plt.Plot()

    for total_step_counter in range(int(max_steps_training)):
        episode_timesteps += 1

        exploration_rate *= exploration_decay
        exploration_rate = max(exploration_min, exploration_rate)

        if random.random() < exploration_rate:
            action = env.action_space.sample()
        else:
            action = agent.select_action_from_policy(state)

        next_state, reward, done, truncated, _ = env.step(action)
        memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done)
        state = next_state
        episode_reward += reward

        if len(memory) > batch_size:
            for _ in range(G):
                experience = memory.sample(batch_size)
                info = agent.train_policy((
                    experience['state'],
                    experience['action'],
                    experience['reward'],
                    experience['next_state'],
                    experience['done']
                ))
                memory.update_priorities(experience['indices'], info)

        if done or truncated:
            record.log(
                Train_steps = total_step_counter + 1,
                Train_exploration_rate = exploration_rate,
                Train_episode= episode_num + 1, 
                Train_timesteps=episode_timesteps,
                Train_reward= episode_reward
            )

            plot.post(episode_reward)

            # Reset environment
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Training time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    record.save()