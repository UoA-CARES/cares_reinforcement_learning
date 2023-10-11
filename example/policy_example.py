from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.memory.augments import *
from cares_reinforcement_learning.util import helpers as hlp, Record

import cv2
import time
import gym
import logging
import numpy as np

def evaluate_policy_network(env, agent, args, record=None, total_steps=0):

    if record is not None:
        frame = env.grab_frame()
        record.start_video(total_steps+1, frame)

    number_eval_episodes = int(args["number_eval_episodes"])
    
    state = env.reset()

    for eval_episode_counter in range(number_eval_episodes):
        episode_timesteps = 0
        episode_reward = 0
        episode_num = 0
        done = False
        truncated = False

        while not done and not truncated:
            episode_timesteps += 1
            action = agent.select_action_from_policy(state, evaluation=True)
            action_env = hlp.denormalize(action, env.max_action_value, env.min_action_value)

            state, reward, done, truncated = env.step(action_env)
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

def policy_based_train(env, agent, memory, record, args):
    start_time = time.time()

    max_steps_training = args["max_steps_training"]
    max_steps_exploration = args["max_steps_exploration"]
    number_steps_per_evaluation = args["number_steps_per_evaluation"]

    logging.info(f"Training {max_steps_training} Exploration {max_steps_exploration} Evaluation {number_steps_per_evaluation}")

    batch_size = args["batch_size"]
    seed = args["seed"]
    G = args["G"]

    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0

    evaluate = False

    state = env.reset()

    episode_start = time.time()
    for total_step_counter in range(int(max_steps_training)):
        episode_timesteps += 1

        if total_step_counter < max_steps_exploration:
            logging.info(f"Running Exploration Steps {total_step_counter+1}/{max_steps_exploration}")
            # action range the env uses [e.g. -2 , 2 for pendulum]
            action_env = np.random.uniform(env.min_action_value, env.max_action_value, size=env.action_num)
            # algorithm range [-1, 1] - note for DMCS this is redudenant but required for openai
            action = hlp.normalize(action_env, env.max_action_value, env.min_action_value)  
        else:
            # algorithm range [-1, 1]
            action = agent.select_action_from_policy(state)
            # mapping to env range [e.g. -2 , 2 for pendulum] - note for DMCS this is redudenant but required for openai
            action_env = hlp.denormalize(action, env.max_action_value, env.min_action_value)  

        next_state, reward, done, truncated = env.step(action_env)
        memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

        state = next_state
        episode_reward += reward

        if total_step_counter >= max_steps_exploration:
            for i in range(G):
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
                args["evaluation_seed"] = seed
                evaluate_policy_network(env, agent, args, record=record, total_steps=total_step_counter)
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