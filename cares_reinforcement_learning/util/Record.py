import os
import logging
import cv2
import json

import pandas as pd

import yaml

from pathlib import Path
from datetime import datetime

import cares_reinforcement_learning.util.plotter as plt

class Record:
    
    def __init__(self, glob_log_dir, log_dir, algorithm, task, plot_frequency=10, checkpoint_frequency=1000, network=None) -> None:
        self.glob_log_dir = glob_log_dir
        self.log_dir = log_dir
        self.directory = f'{self.glob_log_dir}/{self.log_dir}'

        self.algorithm = algorithm
        self.task = task

        self.plot_frequency = plot_frequency
        self.checkpoint_frequency = checkpoint_frequency
        
        self.train_data = pd.DataFrame()
        self.eval_data = pd.DataFrame()
        self.info_data = pd.DataFrame()
        
        self.network = network
        
        self.log_count = 0

        self.__initialise_directories()

    def save_config(self, configuration, file_name):
        with open(f'{self.directory}/{file_name}.json', 'w') as outfile:
            json.dump(configuration.dict(exclude_none=True), outfile)

    def start_video(self, file_name, frame):
        fps        = 30
        video_name = f"{self.directory}/videos/{file_name}.mp4"
        height, width, channels = frame.shape
        self.video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    def stop_video(self):
        self.video.release()

    def log_video(self, frame):
        self.video.write(frame)
    
    def log_info(self, info, display=False):
        self.info_data = pd.concat([self.info_data, pd.DataFrame([info])], ignore_index=True)
        self.save_data(self.info_data, "info", info, display=display)

    def log_train(self, display=False, **logs):
        self.log_count += 1

        self.train_data = pd.concat([self.train_data, pd.DataFrame([logs])], ignore_index=True)
        self.save_data(self.train_data, "train", logs, display=display)

        if self.log_count % self.plot_frequency == 0:
            plt.plot_train(self.train_data, f"Training-{self.algorithm}-{self.task}", f"{self.algorithm}", self.directory, "train", 20)

        if self.network is not None and self.log_count % self.checkpoint_frequency == 0:
            self.network.save_models(f"{self.algorithm}-checkpoint-{self.log_count}", self.directory)

    def log_eval(self, display=False, **logs):
        self.eval_data = pd.concat([self.eval_data, pd.DataFrame([logs])], ignore_index=True)
        self.save_data(self.eval_data, "eval", logs, display=display)

        plt.plot_eval(self.eval_data, f"Evaluation-{self.algorithm}-{self.task}", f"{self.algorithm}", self.directory, "eval")
         
    def save_data(self, data_frame, filename, logs, display=True):
        if data_frame.empty:
            logging.warning('Trying to save an Empty Dataframe')
            
        path = f'{self.directory}/data/{filename}.csv'
        data_frame.to_csv(path, index=False)

        string = [f'{key}: {str(val)[0:10]:6s}' for key, val in logs.items()]
        string = ' | '.join(string)
        string = '| ' + string + ' |'

        if display:
            logging.info(string)

    def save(self):
        logging.info(f"Saving final outputs")
        self.save_data(self.train_data, "train", {}, display=False)
        self.save_data(self.eval_data, "eval", {}, display=False)

        plt.plot_eval(self.eval_data, f"Evaluation-{self.algorithm}-{self.task}", f"{self.algorithm}", self.directory, "eval")
        plt.plot_train(self.train_data, f"Training-{self.algorithm}-{self.task}", f"{self.algorithm}", self.directory, "train", 20)

        if self.network is not None:
            self.network.save_models(self.algorithm, self.directory)

    def __initialise_directories(self):
        if not os.path.exists(self.glob_log_dir):
            os.makedirs(self.glob_log_dir)
            
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        
        if not os.path.exists(f'{self.directory}/data'):
            os.makedirs(f'{self.directory}/data')
            
        if not os.path.exists(f'{self.directory}/models'):
            os.makedirs(f'{self.directory}/models')
            
        if not os.path.exists(f'{self.directory}/figures'):
            os.makedirs(f'{self.directory}/figures')

        if not os.path.exists(f'{self.directory}/videos'):
            os.makedirs(f'{self.directory}/videos')
