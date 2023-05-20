from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.util import helpers as hlp

import gym
import logging
import random

def evaluate_value_network(env, agent, args):
    evaluation_seed = args["evaluation_seed"]
    max_steps_evaluation = args["max_steps_evaluation"]

    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0

    env = gym.make(env.spec.id, render_mode="human")
    state, _ = env.reset(seed=evaluation_seed)

    historical_reward = {"step": [], "episode_reward": []}
    exploration_rate  = args["exploration_min"]

    for total_step_counter in range(int(max_steps_evaluation)):
        episode_timesteps += 1

        if random.random() < exploration_rate:
            action = env.action_space.sample()
        else:
            action = agent.select_action_from_policy(state)

        state, reward, done, truncated, _ = env.step(action)
        episode_reward += reward

        if done or truncated:
            logging.info(f"Total T:{total_step_counter+1} Episode {episode_num+1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}.")

            historical_reward["step"].append(total_step_counter)
            historical_reward["episode_reward"].append(episode_reward)

            # Reset environment
            state, _ = env.reset()
            episode_reward    = 0
            episode_timesteps = 0
            episode_num += 1

def value_based_train(env, agent, args):
    max_steps_training = args["max_steps_training"]
    exploration_min    = args["exploration_min"]
    exploration_decay  = args["exploration_decay"]
    
    batch_size = args["batch_size"]
    seed = args["seed"]
    G = args["G"]

    memory = MemoryBuffer()

    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0

    state, _ = env.reset(seed=seed)

    historical_reward = {"step": [], "episode_reward": []}
    exploration_rate  = 1

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
                agent.train_policy((
                    experience['state'],
                    experience['action'],
                    experience['reward'],
                    experience['next_state'],
                    experience['done']
                ))

        if done or truncated:
            logging.info(f"Total T:{total_step_counter+1} Episode {episode_num+1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}. Exploration Rate: {exploration_rate}")

            historical_reward["step"].append(total_step_counter)
            historical_reward["episode_reward"].append(episode_reward)

            # Reset environment
            state, _ = env.reset()
            episode_reward    = 0
            episode_timesteps = 0
            episode_num += 1

    hlp.plot_reward_curve(historical_reward)

    evaluate_value_network(env, agent, args)