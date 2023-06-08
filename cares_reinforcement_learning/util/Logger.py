import pandas as pd
from datetime import datetime
import os
import logging
import torch

# Python has no max int
MAX_INT = 9999999

class Logger:
    
    def __init__(self, glob_log_dir=None, log_dir=None, networks={}, checkpoint_freq=None) -> None:
        
        self.glob_log_dir = glob_log_dir or 'rl_logs'
        self.log_dir = log_dir or datetime.now().strftime("%y_%m_%d_%H:%M:%S")
        self.dir = f'{self.glob_log_dir}/{self.log_dir}'
        
        self.data = pd.DataFrame() 
        self.checkpoint_freq = checkpoint_freq
        
        self.networks = networks    
        
        self.log_count = 0
        
        self.initial_log_keys = set()
        self.__initialise_directories()
    
    def log(self, **logs):
        self.log_count += 1
        
        if not self.initial_log_keys:
            logging.info('Setting Log Values')
            self.initial_log_keys.union(logs.keys())
        
        if self.initial_log_keys != logs.keys():
            logging.warning('Introducing new columns')
            self.initial_log_keys = self.initial_log_keys.union(logs.keys())
        
        if self.checkpoint_freq and self.log_count % self.checkpoint_freq == 0:
            self.save(f'_{len(self.data)}')
    
        self.data = pd.concat([self.data, pd.DataFrame([logs])])
        
        str = [f'{key}: {val}' for key, val in logs.items()]
        str = ' | '.join(str)
        str = '| ' + str + ' |'

        print(str)
        
    def save(self, sfx='_final'):
        if self.data.empty:
            logging.warning('Trying to save an Empty Dataframe')
        
        self.data.to_csv(f'{self.dir}/data/data{sfx}.csv')
        
        if self.networks:
            for name, network in self.networks.items():
                torch.save(network.state_dict(), f'{self.dir}/models/{name}{sfx}.pht')
        
    def __initialise_directories(self):
        if not os.path.exists(self.glob_log_dir):
            os.mkdir(self.glob_log_dir)
            
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)
        
        if not os.path.exists(f'{self.dir}/data'):
            os.mkdir(f'{self.dir}/data')
            
        if not os.path.exists(f'{self.dir}/models'):
            os.mkdir(f'{self.dir}/models') 
