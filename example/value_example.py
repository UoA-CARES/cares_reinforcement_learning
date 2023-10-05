from cares_reinforcement_learning.memory import *
from cares_reinforcement_learning.util import helpers as hlp, Record

import time
import gym
import logging
import random

from timeit import default_timer as timer

def evaluate_value_network(env, agent, args, record=None, total_steps=0):

    number_eval_episodes = int(args["number_eval_episodes"])
    
    state, _ = env.reset()
    
    exploration_rate = args["exploration_min"]

    for eval_episode_counter in range(number_eval_episodes):
        episode_timesteps = 0
        episode_reward = 0
        episode_num = 0
        done = False
        truncated = False
        
        while not done and not truncated:
            episode_timesteps += 1

            if random.random() < exploration_rate:
                action = env.action_space.sample()
            else:
                action = agent.select_action_from_policy(state)

            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward

            if done or truncated:
                if record is not None:
                    record.log_eval(
                        total_steps=total_steps+1,
                        episode=eval_episode_counter+1, 
                        episode_reward=episode_reward,
                        display=True
                    )

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
    number_steps_per_evaluation = args["number_steps_per_evaluation"]

    batch_size = args["batch_size"]
    seed = args["seed"]
    G = args["G"]

    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0
    
    evaluate = False

    state, _ = env.reset(seed=seed)

    exploration_rate = 1

    episode_start = time.time()
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
                # TODO add saving info information from train_policy as seperate recording
            
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
                args["evaluation_seed"] = seed
                evaluate_value_network(env, agent, args, record=record, total_steps=total_step_counter)
                logging.info("--------------------------------------------")
                evaluate = False

            # Reset environment
            state, _ = env.reset()
            episode_timesteps = 0
            episode_reward = 0
            episode_num += 1
            episode_start = time.time()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Training time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))