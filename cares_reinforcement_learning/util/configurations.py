import logging
import json 

from pathlib import Path
file_path = Path(__file__).parent.resolve()

from pydantic import BaseModel, Field
from typing import List, Optional

def create_environment_config_from_file(file_path):
    with open(file_path) as f:
        args = json.load(f)
        env_config = EnvironmentConfig.model_validate(args)
        print(f"Environment Configuration:\n{env_config}")
        return env_config

def create_training_config_from_file(file_path):
    with open(file_path) as f:
        args = json.load(f)
        training_config = TrainingConfig.model_validate(args)
        print(f"Training Configuration:\n{training_config}")
        return training_config

def create_algorithm_config_from_file(file_path):
    with open(file_path) as f:
        args = json.load(f)
        return create_algorithm_config(args)

def create_algorithm_config(args):
    algorithm = args['algorithm']
    if algorithm == "DQN":
        alg_config = DQNConfig.model_validate(args)
    elif algorithm == "DDQN":
        alg_config = DoubleDQNConfig.model_validate(args)
    elif algorithm == "DuelingDQN":
        alg_config = DuelingDQNConfig.model_validate(args)
    elif algorithm == "PPO":
        alg_config = PPOConfig.model_validate(args)
    elif algorithm == "DDPG":
        alg_config = DDPGConfig.model_validate(args)
    elif algorithm == "SAC":
        alg_config = SACConfig.model_validate(args)
    elif algorithm == "TD3":
        alg_config = TD3Config.model_validate(args)
    else:
        logging.warn(f"Unkown algorithm: {alg_config} for config {file_path}")
        return None
    print(f"Algorithm Configuration:\n{alg_config}")
    return alg_config

class EnvironmentConfig(BaseModel):
    gym_environment: str
    task: str
    domain: Optional[str] = ""
    image_observation: Optional[bool] = False

class TrainingConfig(BaseModel):
    seed: int = Field(description='Random seed to set for the environment')
    number_training_iterations: Optional[int] = 1

    G: Optional[int] = 1
    batch_size: Optional[int] = 10
    
    max_steps_exploration: Optional[int] = 1000
    max_steps_training: Optional[int] = 1000000

    number_steps_per_evaluation: Optional[int] = 10000
    number_eval_episodes: Optional[int] = 10

    plot_frequency: Optional[int] = 100
    checkpoint_frequency: Optional[int] = 100
    
class AlgorithmConfig(BaseModel):
    algorithm: str = Field(description='Name of the algorithm to be used')

class DQNConfig(AlgorithmConfig):
    lr: Optional[float] = 1e-3
    gamma: Optional[float] = 0.99
    memory: Optional[str] = "MemoryBuffer"

    exploration_min: Optional[float] = 1e-3
    exploration_decay: Optional[float] = 0.95

class DuelingDQNConfig(AlgorithmConfig):
    lr: Optional[float] = 1e-3
    gamma: Optional[float] = 0.99
    memory: Optional[str] = "MemoryBuffer"

    exploration_min: Optional[float] = 1e-3
    exploration_decay: Optional[float] = 0.95

class DoubleDQNConfig(AlgorithmConfig):
    lr: Optional[float] = 1e-3
    gamma: Optional[float] = 0.99
    memory: Optional[str] = "MemoryBuffer"

    exploration_min: Optional[float] = 1e-3
    exploration_decay: Optional[float] = 0.95

class DDPGConfig(AlgorithmConfig):
    actor_lr: Optional[float] = 1e-4
    critic_lr: Optional[float] = 1e-3
    
    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005
    
    memory: Optional[str] = "MemoryBuffer"

class PPOConfig(AlgorithmConfig):
    actor_lr: Optional[float] = 1e-4
    critic_lr: Optional[float] = 1e-3
    
    gamma: Optional[float] = 0.99
    
    memory: str = Field("MemoryBuffer", Literal=True)

class TD3Config(AlgorithmConfig):
    actor_lr: Optional[float] = 1e-4
    critic_lr: Optional[float] = 1e-3
    
    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005
    
    memory: Optional[str] = "MemoryBuffer"

class SACConfig(AlgorithmConfig):
    actor_lr: Optional[float] = 1e-4
    critic_lr: Optional[float] = 1e-3
    
    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005
    
    memory: Optional[str] = "MemoryBuffer"