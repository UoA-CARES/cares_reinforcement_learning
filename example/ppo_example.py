from cares_reinforcement_learning.memory import *
from cares_reinforcement_learning.util import helpers as hlp
from cares_reinforcement_learning.util import Record

import time
import gym
import logging

def evaluate_ppo_network(env, agent, record, args, training_step=0):
    
    number_eval_episodes = int(args["number_eval_episodes"])

    min_action_value = env.action_space.low[0]
    max_action_value = env.action_space.high[0]

    state, _ = env.reset()

    for eval_episode_counter in range(number_eval_episodes):
        episode_timesteps = 0
        episode_reward = 0
        episode_num = 0
        done = False
        truncated = False
        
        while not done and not truncated:
            episode_timesteps += 1
            action, log_prob = agent.select_action_from_policy(state)
            action_env = hlp.denormalize(action, max_action_value, min_action_value)

            state, reward, done, truncated, _ = env.step(action_env)
            episode_reward += reward

            if done or truncated:
                record.log_eval(
                        train_step=training_step+1,
                        episode=eval_episode_counter+1, 
                        episode_reward=episode_reward,
                        display=True
                    )
                
                # Reset environment
                state, _ = env.reset()
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

def ppo_train(env, agent, record, args):
    start_time = time.time()

    seed = args["seed"]
    max_steps_training = args["max_steps_training"]
    max_steps_per_batch = args["max_steps_per_batch"]
    number_steps_per_evaluation = args["number_steps_per_evaluation"]

    min_action_value = env.action_space.low[0]
    max_action_value = env.action_space.high[0]

    episode_num = 1
    episode_reward = 0

    memory = MemoryBuffer()

    evaluate = False

    state, _ = env.reset(seed=seed)

    for total_step_counter in range(int(max_steps_training)):
        action, log_prob = agent.select_action_from_policy(state)
        action_env = hlp.denormalize(action, max_action_value, min_action_value)

        next_state, reward, done, truncated, _ = env.step(action_env)
        memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done, log_prob=log_prob)

        state = next_state
        episode_reward += reward

        if (total_step_counter+1) % max_steps_per_batch == 0:
            experience = memory.flush()
            info = agent.train_policy((
                experience['state'],
                experience['action'],
                experience['reward'],
                experience['next_state'],
                experience['done'],
                experience['log_prob']
            ))
            # TODO add saving info information from train_policy as seperate recording

        if (total_step_counter+1) % number_steps_per_evaluation == 0:
            evaluate = True

        if done or truncated:
            record.log_train(
                total_steps = total_step_counter + 1,
                episode = episode_num,
                reward = episode_reward,
                display = True
            )

            if evaluate:
                logging.info("*************--Evaluation Loop--*************")
                args["evaluation_seed"] = seed
                evaluate_ppo_network(env, agent, record, args, training_step=total_step_counter)
                logging.info("--------------------------------------------")
                evaluate = False

            # Reset environment
            state, _ = env.reset()
            episode_reward = 0
            episode_num += 1

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Training time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))