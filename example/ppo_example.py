from cares_reinforcement_learning.util import RolloutBuffer
from cares_reinforcement_learning.util import helpers as hlp

import gym
import logging
import random

def evaluate_ppo_network(env, agent, args):
    evaluation_seed = args["evaluation_seed"]
    max_steps_evaluation = args["max_steps_evaluation"]

    min_action_value = env.action_space.low[0]
    max_action_value = env.action_space.high[0]

    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0

    env = gym.make(env.spec.id, render_mode="human")
    state, _ = env.reset(seed=evaluation_seed)

    for total_step_counter in range(int(max_steps_evaluation)):
        episode_timesteps += 1
        action = agent.select_action_from_policy(state, evaluation=True)
        action_env = hlp.denormalize(action, max_action_value, min_action_value)

        state, reward, done, truncated, _ = env.step(action_env)
        episode_reward += reward

        if done or truncated:
            logging.info(f" Evaluation Episode {episode_num+1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}")
            # Reset environment
            state, _ = env.reset()
            episode_reward    = 0
            episode_timesteps = 0
            episode_num += 1

def ppo_train(env, agent, args):
    seed = args["seed"]
    max_steps_training = args["max_steps_training"]
    max_steps_per_batch = args["max_steps_per_batch"]

    min_action_value = env.action_space.low[0]
    max_action_value = env.action_space.high[0]

    episode_timesteps = 0
    episode_num       = 0
    episode_reward    = 0
    time_step         = 0

    memory = RolloutBuffer()# TODO genericify the memory buffer

    state, _ = env.reset(seed=seed)
    historical_reward = {"step": [], "episode_reward": []}

    for total_step_counter in range(int(max_steps_training)):
        episode_timesteps += 1

        action, log_prob = agent.select_action_from_policy(state)
        action_env = hlp.normalize(action, max_action_value, min_action_value)  # mapping the env range [e.g. -2 , 2 for pendulum]

        next_state, reward, done, truncated, _ = env.step(action_env)

        # ------------------------------------------------------------------------------------------------
        # save rollouts in memory, TODO generic rollout buffer to match a general buffer
        memory.states.append(state)
        memory.next_states.append(next_state)
        memory.actions.append(action)
        memory.log_probs.append(log_prob)
        memory.rewards.append(reward)
        memory.dones.append(done)
        # ------------------------------------------------------------------------------------------------

        state = next_state
        episode_reward += reward

        time_step += 1  # I need this otherwise the next if is true at the first interaction
        if time_step % max_steps_per_batch == 0:
            agent.train_policy(memory)

        if done or truncated:
            logging.info(f"Total T:{total_step_counter + 1} Episode {episode_num + 1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}")

            historical_reward["step"].append(total_step_counter+1)
            historical_reward["episode_reward"].append(episode_reward)

            # Reset environment
            state, _ = env.reset()
            episode_reward    = 0
            episode_timesteps = 0
            episode_num       += 1

    hlp.plot_reward_curve(historical_reward)

    evaluate_ppo_network(env, agent, args)