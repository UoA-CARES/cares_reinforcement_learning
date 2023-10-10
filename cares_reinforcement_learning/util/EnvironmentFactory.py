import logging

import gym
from gym import spaces

from dm_control import suite

import numpy as np
from collections import deque

# from typing import override
from functools import cached_property

class EnvironmentFactory:
    def __init__(self) -> None:
        pass

    def create_environment(self, gym_environment, args):
        logging.info(f"Training Environment: {gym_environment}")
        if gym_environment == 'dmcs':
            env = DMCSImageEnvironment(args=args) if args['image_observation'] else DMCS(args=args)
        elif gym_environment == "gym":
            env = OpenAIGym(args=args)
        else:
            raise ValueError(f"Unkown environment: {gym_environment}")
        return env
        
class OpenAIGym:
    def __init__(self, args) -> None:
        logging.info(f"Training task {args['task']}")
        self.env = gym.make(args["task"], render_mode=(None if args['render'] == "None" else args['render']))
        self.env.action_space.seed(args['seed'])
        
    @cached_property
    def max_action_value(self):
        return self.env.action_space.high[0]

    @cached_property
    def min_action_value(self):
        return self.env.action_space.low[0]

    @cached_property
    def observation_space(self):
        return self.env.observation_space.shape[0]
    
    @cached_property
    def action_num(self):
        if type(self.env.action_space) == spaces.Box:
            action_num = self.env.action_space.shape[0]
        elif type(self.env.action_space) == spaces.Discrete:
            action_num= self.env.action_space.n
        else:
            raise ValueError(f"Unhandled action space type: {type(self.env.action_space)}")
        return action_num

    def reset(self):
        state, _ = self.env.reset()
        return state

    def step(self, action):
        state, reward, done, truncated, _ = self.env.step(action)
        return state, reward, done, truncated
        
class DMCS:
    def __init__(self, args) -> None:
        logging.info(f"Training on Domain {args['domain']}")
        logging.info(f"Training with Task {args['task']}")
      
        self.env = suite.load(args['domain'], args['task'], task_kwargs={'random': args['seed']})
    
    @cached_property
    def min_action_value(self):
        return self.env.action_spec().minimum[0]
    
    @cached_property
    def max_action_value(self):
        return self.env.action_spec().maximum[0]

    @cached_property
    def observation_space(self):
        time_step = self.env.reset()
        observation = np.hstack(list(time_step.observation.values())) # # e.g. position, orientation, joint_angles
        return len(observation)
    
    @cached_property
    def action_num(self):
        return self.env.action_spec().shape[0]

    def reset(self):
        time_step = self.env.reset()
        observation = np.hstack(list(time_step.observation.values())) # # e.g. position, orientation, joint_angles
        return observation

    def step(self, action):
        time_step = self.env.step(action)
        state, reward, done = np.hstack(list(time_step.observation.values())), time_step.reward, time_step.last()
        return state, reward, done, False # for consistency with open ai gym just add false for truncated

# TODO paramatise the observation size 3x84x84
class DMCSImageEnvironment(DMCS):
    def __init__(self, args, k=3):
        self.k    = k  # number of frames to be stacked
        self.frames_stacked = deque([], maxlen=k)

        super().__init__(args=args)

    # @override
    @property
    def observation_space(self):
        return self.env.observation_space.shape[0]

    # @override
    def reset(self):
        _ = self.env.reset()
        frame = self.env.physics.render(84, 84, camera_id=0) # --> shape= (84, 84, 3)
        frame = np.moveaxis(frame, -1, 0)                    # --> shape= (3, 84, 84)
        for _ in range(self.k):
            self.frames_stacked.append(frame)
        stacked_frames = np.concatenate(list(self.frames_stacked), axis=0) # --> shape = (9, 84, 84)
        return stacked_frames

    # @override
    def step(self, action):
        time_step    = self.env.step(action)
        reward, done = time_step.reward, time_step.last()
        frame = self.env.physics.render(84, 84, camera_id=0)
        frame = np.moveaxis(frame, -1, 0)
        self.frames_stacked.append(frame)
        stacked_frames = np.concatenate(list(self.frames_stacked), axis=0)
        return stacked_frames, reward, done, False # for consistency with open ai gym just add false for truncated