import logging
import json 

from pathlib import Path
file_path = Path(__file__).parent.resolve()

from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# NOTE: If a parameter is a list then don't wrap with Optional leave as implicit optional - List[type] = default

class SubscriptableClass(BaseModel):
    def __getitem__(self, item):
        return getattr(self, item)

class EnvironmentConfig(SubscriptableClass):
    task: str

class TrainingConfig(SubscriptableClass):
    seeds: List[int] = [10]

    G: Optional[int] = 1
    buffer_size: Optional[int] = 1000000
    batch_size: Optional[int] = 10
    
    max_steps_exploration: Optional[int] = 1000
    max_steps_training: Optional[int] = 1000000

    number_steps_per_evaluation: Optional[int] = 10000
    number_eval_episodes: Optional[int] = 10

    plot_frequency: Optional[int] = 100
    checkpoint_frequency: Optional[int] = 100

class AlgorithmConfig(SubscriptableClass):
    algorithm: str = Field(description='Name of the algorithm to be used')

class DQNConfig(AlgorithmConfig):
    algorithm: str = Field("DQN", Literal=True)
    lr: Optional[float] = 1e-3
    gamma: Optional[float] = 0.99
    memory: Optional[str] = "MemoryBuffer"

    exploration_min: Optional[float] = 1e-3
    exploration_decay: Optional[float] = 0.95

class DuelingDQNConfig(AlgorithmConfig):
    algorithm: str = Field("DuelingDQN", Literal=True)
    lr: Optional[float] = 1e-3
    gamma: Optional[float] = 0.99
    memory: Optional[str] = "MemoryBuffer"

    exploration_min: Optional[float] = 1e-3
    exploration_decay: Optional[float] = 0.95

class DoubleDQNConfig(AlgorithmConfig):
    algorithm: str = Field("DoubleDQN", Literal=True)
    lr: Optional[float] = 1e-3
    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005
    memory: Optional[str] = "MemoryBuffer"

    exploration_min: Optional[float] = 1e-3
    exploration_decay: Optional[float] = 0.95

class PPOConfig(AlgorithmConfig):
    algorithm: str = Field("PPO", Literal=True)
    actor_lr: Optional[float] = 1e-4
    critic_lr: Optional[float] = 1e-3
    
    gamma: Optional[float] = 0.99
    max_steps_per_batch: Optional[int] = 5000
    
    memory: str = Field("MemoryBuffer", Literal=True)

class DDPGConfig(AlgorithmConfig):
    algorithm: str = Field("DDPG", Literal=True)
    actor_lr: Optional[float] = 1e-4
    critic_lr: Optional[float] = 1e-3
    
    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005
    
    memory: Optional[str] = "MemoryBuffer"

class TD3Config(AlgorithmConfig):
    algorithm: str = Field("TD3", Literal=True)
    actor_lr: Optional[float] = 3e-4
    critic_lr: Optional[float] = 3e-4
    
    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005
    
    memory: Optional[str] = "MemoryBuffer"

class SACConfig(AlgorithmConfig):
    algorithm: str = Field("SAC", Literal=True)
    actor_lr: Optional[float] = 1e-4
    critic_lr: Optional[float] = 1e-3
    
    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005
    
    memory: Optional[str] = "MemoryBuffer"

class NaSATD3Config(AlgorithmConfig):
    algorithm: str = Field("NaSATD3", Literal=True)
    # actor_lr: Optional[float] = 1e-4
    # critic_lr: Optional[float] = 1e-3
    
    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005
    
    memory: Optional[str] = "MemoryBuffer"

    latent_size: Optional[int] = 200
    intrinsic_on: Optional[bool] = True

                # lr_actor   = 1e-4
                # lr_critic  = 1e-3

                # lr_encoder = 1e-3
                # lr_decoder = 1e-3

                # lr_epm      = 1e-4
                # w_decay_epm = 1e-3