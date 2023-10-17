import logging

import cv2

import gym
from gym import spaces

from dm_control import suite

import numpy as np
from collections import deque

# from typing import override
from functools import cached_property

from cares_reinforcement_learning.util.configurations import EnvironmentConfig

class EnvironmentFactory:
    def __init__(self) -> None:
        pass

    def create_environment(self, config: EnvironmentConfig):
        logging.info(f"Training Environment: {config.gym_environment}")
        if config.gym_environment == 'dmcs':
            env = DMCSImage(config) if config.image_observation else DMCS(config)
        elif config.gym_environment == "openai":
            env = OpenAIGymImage(config) if config.image_observation else OpenAIGym(config)
        else:
            raise ValueError(f"Unkown environment: {config.gym_environment}")
        return env
        
class OpenAIGym:
    def __init__(self, config: EnvironmentConfig) -> None:
        logging.info(f"Training task {config.task}")
        self.env = gym.make(config.task, render_mode="rgb_array")
    
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

    def set_seed(self, seed):
        self.env.action_space.seed(seed)

    def reset(self):
        state, _ = self.env.reset()
        return state

    def step(self, action):
        state, reward, done, truncated, _ = self.env.step(action)
        return state, reward, done, truncated
    
    def grab_frame(self, height=240, width=300):
        frame = self.env.render()
        frame = cv2.resize(frame, (width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to BGR for use with OpenCV
        return frame
    
class OpenAIGymImage(OpenAIGym):
    def __init__(self, config: EnvironmentConfig, k=3):
        self.k    = k  # number of frames to be stacked
        self.frames_stacked = deque([], maxlen=k)

        self.frame_width = 84
        self.frame_height = 84

        super().__init__(config)

    # @override
    @property
    def observation_space(self):
        return (9, self.frame_width, self.frame_height)

    # @override
    def reset(self):
        _ = self.env.reset()
        frame = self.grab_frame(height=self.frame_height, width=self.frame_width)
        frame = np.moveaxis(frame, -1, 0)                    
        for _ in range(self.k):
            self.frames_stacked.append(frame)
        stacked_frames = np.concatenate(list(self.frames_stacked), axis=0)
        return stacked_frames

    # @override
    def step(self, action):
        time_step    = self.env.step(action)
        reward, done = time_step.reward, time_step.last()
        frame = self.grab_frame(height=self.frame_height, width=self.frame_width)
        frame = np.moveaxis(frame, -1, 0)
        self.frames_stacked.append(frame)
        stacked_frames = np.concatenate(list(self.frames_stacked), axis=0)
        return stacked_frames, reward, done, False # for consistency with open ai gym just add false for truncated
        
class DMCS:
    def __init__(self, config: EnvironmentConfig) -> None:
        logging.info(f"Training on Domain {config.domain}")
        logging.info(f"Training with Task {config.task}")
      
        self.domain = config.domain
        self.task = config.task
        self.env = suite.load(self.domain, self.task)
        
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

    def set_seed(self, seed):
        self.env = suite.load(self.domain, self.task, task_kwargs={'random': seed})

    def reset(self):
        time_step = self.env.reset()
        observation = np.hstack(list(time_step.observation.values())) # # e.g. position, orientation, joint_angles
        return observation

    def step(self, action):
        time_step = self.env.step(action)
        state, reward, done = np.hstack(list(time_step.observation.values())), time_step.reward, time_step.last()
        return state, reward, done, False # for consistency with open ai gym just add false for truncated
    
    def grab_frame(self, camera_id=0, height=240, width=300):
        frame = self.env.physics.render(camera_id=camera_id, height=height, width=width)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to BGR for use with OpenCV
        return frame

# TODO paramatise the observation size 3x84x84
class DMCSImage(DMCS):
    def __init__(self, config: EnvironmentConfig, k=3):
        self.k    = k  # number of frames to be stacked
        self.frames_stacked = deque([], maxlen=k)

        self.frame_width = 84
        self.frame_height = 84

        super().__init__(config)

    # @override
    @property
    def observation_space(self):
        return (9, self.frame_width, self.frame_height)

    # @override
    def reset(self):
        _ = self.env.reset()
        frame = self.grab_frame(height=self.frame_height, width=self.frame_width, camera_id=0)
        frame = np.moveaxis(frame, -1, 0)                    
        for _ in range(self.k):
            self.frames_stacked.append(frame)
        stacked_frames = np.concatenate(list(self.frames_stacked), axis=0)
        return stacked_frames

    # @override
    def step(self, action):
        time_step    = self.env.step(action)
        reward, done = time_step.reward, time_step.last()
        frame = self.grab_frame(height=self.frame_height, width=self.frame_width, camera_id=0)
        frame = np.moveaxis(frame, -1, 0)
        self.frames_stacked.append(frame)
        stacked_frames = np.concatenate(list(self.frames_stacked), axis=0)
        return stacked_frames, reward, done, False # for consistency with open ai gym just add false for truncated