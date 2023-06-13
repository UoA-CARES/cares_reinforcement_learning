from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.memory.augments import *
from cares_reinforcement_learning.util import helpers as hlp, Record, Plot as plt

import time
import gym
import logging

def evaluate_policy_network(env, agent, record, args):
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
        action = agent.select_action_from_policy(state, evaluation=True)
        action_env = hlp.denormalize(action, max_action_value, min_action_value)

        state, reward, done, truncated, _ = env.step(action_env)
        episode_reward += reward

        if done or truncated:
            record.log(
                Eval_episode= episode_num + 1, 
                Eval_timesteps=episode_timesteps,
                Eval_reward= episode_reward
            )
            # Reset environment
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1


def policy_based_train(env, agent, memory, record, args):
    start_time = time.time()

    max_steps_training = args["max_steps_training"]
    max_steps_exploration = args["max_steps_exploration"]
    batch_size = args["batch_size"]
    seed = args["seed"]
    G = args["G"]

    min_action_value = env.action_space.low[0]
    max_action_value = env.action_space.high[0]

    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0

    state, _ = env.reset(seed=seed)
    env.render()

    plot = plt.Plot()

    for total_step_counter in range(int(max_steps_training)):
        episode_timesteps += 1

        if total_step_counter < max_steps_exploration:
            logging.info(f"Running Exploration Steps {total_step_counter}/{max_steps_exploration}")
            action_env = env.action_space.sample()  # action range the env uses [e.g. -2 , 2 for pendulum]
            action = hlp.normalize(action_env, max_action_value, min_action_value)  # algorithm range [-1, 1]
        else:
            action = agent.select_action_from_policy(state)  # algorithm range [-1, 1]
            action_env = hlp.denormalize(action, max_action_value,
                                         min_action_value)  # mapping to env range [e.g. -2 , 2 for pendulum]

        next_state, reward, done, truncated, info = env.step(action_env)
        memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

        state = next_state
        episode_reward += reward

        if total_step_counter >= max_steps_exploration:
            actor_loss = 0
            critic_loss = 0
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
                critic_loss += info['critic_loss_total'].item()
                if (i+1) % agent.policy_update_freq == 0:
                    actor_loss += info['actor_loss'].item()
                
            # record average losses
            record.log(
                Actor_loss = actor_loss/(G/agent.policy_update_freq),
                Critic_loss = critic_loss/G
            )

        if done or truncated:
            record.log(
                Train_steps = total_step_counter + 1,
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