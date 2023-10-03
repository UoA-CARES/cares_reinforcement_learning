import pandas as pd
import logging

import os
import yaml
from pathlib import Path
from datetime import datetime

from cares_reinforcement_learning.util.Plot import plot_average

class Record:
    
    def __init__(self, glob_log_dir=None, log_dir=None, network=None, config=None) -> None:
        self.glob_log_dir = glob_log_dir or f'{Path.home()}/cares_rl_logs'
        self.log_dir = log_dir or datetime.now().strftime("%y_%m_%d_%H:%M:%S")
        self.directory = f'{self.glob_log_dir}/{self.log_dir}'
        
        self.train_data = pd.DataFrame()
        self.eval_data = pd.DataFrame()
        
        self.network = network
        
        self.log_count = 0
        
        self.__initialise_directories()
        
        if config:
            with open(f'{self.directory}/config.yml', 'w') as outfile:
                yaml.dump(config, outfile, default_flow_style=False)
    
    def log_train(self, display=False, **logs):        
        self.train_data = pd.concat([self.train_data, pd.DataFrame([logs])], ignore_index=True)
        self.save_data(self.train_data, "train", logs, display=display)

    def log_eval(self, display=False, **logs):        
        self.eval_data = pd.concat([self.eval_data, pd.DataFrame([logs])], ignore_index=True)
        self.save_data(self.eval_data, "eval", logs, display=display)
         
    def save_data(self, data_frame, filename, logs, display=True):
        if data_frame.empty:
            logging.warning('Trying to save an Empty Dataframe')
            
        path = f'{self.directory}/data/{filename}.csv'
        data_frame.to_csv(path, mode='a', header=not os.path.exists(path), index=False)
        data_frame.drop(data_frame.index, inplace=True)

        string = [f'{key}: {str(val)[0:10]:6s}' for key, val in logs.items()]
        string = ' | '.join(string)
        string = '| ' + string + ' |'

        if display:
            print(string)
                
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
