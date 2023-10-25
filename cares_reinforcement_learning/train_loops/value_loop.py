from cares_reinforcement_learning.util.configurations import TrainingConfig, AlgorithmConfig
from cares_reinforcement_learning.util import helpers as hlp

import numpy as np
import time
import gym
import logging
import random

from random import randrange

from timeit import default_timer as timer

def evaluate_value_network(env, agent, train_config: TrainingConfig, alg_config: AlgorithmConfig, record=None, total_steps=0):

    if record is not None:
        frame = env.grab_frame()
        record.start_video(total_steps+1, frame)

    number_eval_episodes = int(train_config.number_eval_episodes)
    
    state = env.reset()
    
    exploration_rate = alg_config.exploration_min

    for eval_episode_counter in range(number_eval_episodes):
        episode_timesteps = 0
        episode_reward = 0
        episode_num = 0
        done = False
        truncated = False
        
        while not done and not truncated:
            episode_timesteps += 1

            if random.random() < exploration_rate:
                action = randrange(env.action_num)
            else:
                action = agent.select_action_from_policy(state)

            state, reward, done, truncated = env.step(action)
            episode_reward += reward

            if eval_episode_counter == 0 and record is not None:
                frame = env.grab_frame()
                record.log_video(frame)

            if done or truncated:
                if record is not None:
                    record.log_eval(
                        total_steps=total_steps+1,
                        episode=eval_episode_counter+1, 
                        episode_reward=episode_reward,
                        display=True
                    )

                # Reset environment
                state = env.reset()
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

    record.stop_video()

def value_based_train(env, agent, memory, record, train_config: TrainingConfig, alg_config: AlgorithmConfig):
    start_time = time.time()

    exploration_min = alg_config.exploration_min
    exploration_decay = alg_config.exploration_decay

    max_steps_training = train_config.max_steps_training
    number_steps_per_evaluation = train_config.number_steps_per_evaluation

    batch_size = train_config.batch_size
    G = train_config.G

    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0
    
    evaluate = False

    state = env.reset()

    exploration_rate = 1

    episode_start = time.time()
    for total_step_counter in range(int(max_steps_training)):
        episode_timesteps += 1

        exploration_rate *= exploration_decay
        exploration_rate = max(exploration_min, exploration_rate)

        if random.random() < exploration_rate:
            action = randrange(env.action_num)
        else:
            action = agent.select_action_from_policy(state)

        next_state, reward, done, truncated = env.step(action)
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
                # record.log_info(info, display=False)
            
        if (total_step_counter+1) % number_steps_per_evaluation == 0:
            evaluate = True

        if done or truncated:
            episode_time = time.time() - episode_start
            record.log_train(
                total_steps = total_step_counter + 1,
                episode = episode_num + 1,
                episode_steps=episode_timesteps,
                episode_reward = episode_reward,
                episode_time = episode_time,
                display = True
            )

            if evaluate:
                logging.info("*************--Evaluation Loop--*************")
                evaluate_value_network(env, agent, train_config, alg_config, record=record, total_steps=total_step_counter)
                logging.info("--------------------------------------------")
                evaluate = False

            # Reset environment
            state = env.reset()
            episode_timesteps = 0
            episode_reward = 0
            episode_num += 1
            episode_start = time.time()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Training time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))