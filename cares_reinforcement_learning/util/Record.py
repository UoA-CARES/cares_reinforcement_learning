import os
import logging

import pandas as pd

import yaml

from pathlib import Path
from datetime import datetime

import cares_reinforcement_learning.util.plotter as plt

class Record:
    
    def __init__(self, glob_log_dir=None, log_dir=None, network=None, config=None) -> None:
        self.task = config["args"]["task"]
        self.algoritm = config["args"]["algorithm"]
        self.plot_frequency = config["args"]["plot_frequency"]
        self.checkpoint_frequency = config["args"]["checkpoint_frequency"]
        
        self.glob_log_dir = glob_log_dir or f'{Path.home()}/cares_rl_logs'
        self.log_dir = log_dir or f"{self.algoritm}-{self.task}-{datetime.now().strftime('%y_%m_%d_%H:%M:%S')}"
        self.directory = f'{self.glob_log_dir}/{self.log_dir}'
        
        self.train_data = pd.DataFrame()
        self.eval_data = pd.DataFrame()
        self.info_data = pd.DataFrame()
        
        self.network = network
        
        self.log_count = 0

        self.__initialise_directories()

        if config:
            with open(f'{self.directory}/config.yml', 'w') as outfile:
                yaml.dump(config, outfile, default_flow_style=False)
    
    def log_info(self, info, display=False):
        self.info_data = pd.concat([self.info_data, pd.DataFrame([info])], ignore_index=True)
        self.save_data(self.info_data, "info", info, display=display)

    def log_train(self, display=False, **logs):
        self.log_count += 1

        self.train_data = pd.concat([self.train_data, pd.DataFrame([logs])], ignore_index=True)
        self.save_data(self.train_data, "train", logs, display=display)

        if self.log_count % self.plot_frequency == 0:
            plt.plot_train(self.train_data, f"Training-{self.algoritm}-{self.task}", f"{self.algoritm}", self.directory, "train", 20)

        if self.network is not None and self.log_count % self.checkpoint_frequency == 0:
            self.network.save_models(f"{self.algoritm}-checkpoint-{self.log_count}", self.directory)

    def log_eval(self, display=False, **logs):
        self.eval_data = pd.concat([self.eval_data, pd.DataFrame([logs])], ignore_index=True)
        self.save_data(self.eval_data, "eval", logs, display=display)

        plt.plot_eval(self.eval_data, f"Evaluation-{self.algoritm}-{self.task}", f"{self.algoritm}", self.directory, "eval")
         
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

        plt.plot_eval(self.eval_data, f"Evaluation-{self.algoritm}-{self.task}", f"{self.algoritm}", self.directory, "eval")
        plt.plot_train(self.train_data, f"Training-{self.algoritm}-{self.task}", f"{self.algoritm}", self.directory, "train", 20)

        if self.network is not None:
            self.network.save_models(self.algoritm, self.directory)

    def __initialise_directories(self):
        if not os.path.exists(self.glob_log_dir):
            os.mkdir(self.glob_log_dir)
            
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)
        
        if not os.path.exists(f'{self.directory}/data'):
            os.mkdir(f'{self.directory}/data')
            
        if not os.path.exists(f'{self.directory}/models'):
            os.mkdir(f'{self.directory}/models')
            
        if not os.path.exists(f'{self.directory}/figures'):
            os.mkdir(f'{self.directory}/figures') 
