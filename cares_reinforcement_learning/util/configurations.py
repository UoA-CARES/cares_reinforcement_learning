from typing import List, Optional
from pydantic import BaseModel, Field
from torch import nn


# pylint disbale-next=unused-import

# NOTE: If a parameter is a list then don't wrap with Optional leave as implicit optional - list[type] = default


class SubscriptableClass(BaseModel):
    def __getitem__(self, item):
        return getattr(self, item)


class EnvironmentConfig(SubscriptableClass):
    task: str


class TrainingConfig(SubscriptableClass):
    """
    Configuration class for training.

    Attributes:
        seeds (list[int]): list of random seeds for reproducibility. Default is [10].
        number_steps_per_evaluation (int]): Number of steps per evaluation. Default is 10000.
        number_eval_episodes (int]): Number of episodes to evaluate during training. Default is 10.
    """

    seeds: list[int] = [10]
    number_steps_per_evaluation: int = 10000
    number_eval_episodes: int = 10


class MLPConfig(SubscriptableClass):
    hidden_sizes: list[int]

    input_layer: str = ""
    input_layer_args: dict[str, Any] = Field(default_factory=dict)

    linear_layer_args: dict[str, Any] = Field(default_factory=dict)

    batch_layer: str = ""
    batch_layer_args: dict[str, Any] = Field(default_factory=dict)

    dropout_layer: str = ""
    dropout_layer_args: dict[str, Any] = Field(default_factory=dict)

    norm_layer: str = ""
    norm_layer_args: dict[str, Any] = Field(default_factory=dict)

    hidden_activation_function: str = nn.ReLU.__name__
    hidden_activation_function_args: dict[str, Any] = Field(default_factory=dict)

    output_activation_function: str = ""
    output_activation_function_args: dict[str, Any] = Field(default_factory=dict)

    layer_order: list[str] = ["batch", "activation", "layernorm", "dropout"]

    @pydantic.root_validator(pre=True)
    # pylint: disable-next=no-self-argument
    def convert_none_to_dict(cls, values):
        if values.get("norm_layer_args") is None:
            values["norm_layer_args"] = {}
        if values.get("activation_function_args") is None:
            values["activation_function_args"] = {}
        if values.get("final_activation_args") is None:
            values["final_activation_args"] = {}
        if values.get("batch_layer_args") is None:
            values["batch_layer_args"] = {}
        if values.get("dropout_layer_args") is None:
            values["dropout_layer_args"] = {}
        return values


class AlgorithmConfig(SubscriptableClass):
    """
    Configuration class for the algorithm.

    These attributes are common to all algorithms. They can be overridden by the specific algorithm configuration.

    Attributes:
        algorithm (str): Name of the algorithm to be used.
        G (int]): Updates per step UTD-raio, for the actor and critic.
        G_model (int]): Updates per step UTD-ratio for MBRL.
        buffer_size (int]): Size of the memory buffer.
        batch_size (int]): Size of the training batch.
        max_steps_exploration (int]): Maximum number of steps for exploration.
        max_steps_training (int]): Maximum number of steps for training.
        number_steps_per_train_policy (int]): Number of steps per updating the training policy.

        min_noise (float]): Minimum noise value.
        noise_scale (float]): Noise scale.
        noise_decay (float]): Noise decay.

        image_observation (int]): Whether the observation is an image.
    """

    algorithm: str = Field(description="Name of the algorithm to be used")
    G: int = 1
    G_model: float = 1
    buffer_size: int = 1000000
    batch_size: int = 256
    max_steps_exploration: int = 1000
    max_steps_training: int = 1000000
    number_steps_per_train_policy: int = 1

    min_noise: float = 0.0
    noise_scale: float = 0.1
    noise_decay: float = 1.0

    image_observation: int = 0


###################################
#         DQN Algorithms          #
###################################


class DQNConfig(AlgorithmConfig):
    algorithm: str = Field("DQN", Literal=True)
    lr: float = 1e-3
    gamma: float = 0.99

    exploration_min: float = 1e-3
    exploration_decay: float = 0.95

    network_config: MLPConfig = MLPConfig(hidden_sizes=[512, 512])


class DoubleDQNConfig(DQNConfig):
    algorithm: str = Field("DoubleDQN", Literal=True)
    lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005

    exploration_min: float = 1e-3
    exploration_decay: float = 0.95

    network_config: MLPConfig = MLPConfig(hidden_sizes=[512, 512])


class DuelingDQNConfig(AlgorithmConfig):
    algorithm: str = Field("DuelingDQN", Literal=True)
    lr: float = 1e-3
    gamma: float = 0.99

    exploration_min: float = 1e-3
    exploration_decay: float = 0.95

    feature_layer_config: MLPConfig = MLPConfig(hidden_sizes=[512, 512])
    value_stream_config: MLPConfig = MLPConfig(hidden_sizes=[512])
    advantage_stream_config: MLPConfig = MLPConfig(hidden_sizes=[512])


###################################
#         PPO Algorithms          #
###################################


class PPOConfig(AlgorithmConfig):
    algorithm: str = Field("PPO", Literal=True)
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3

    gamma: float = 0.99
    eps_clip: float = 0.2
    updates_per_iteration: int = 10

    max_steps_per_batch: int = 5000

    actor_config: MLPConfig = MLPConfig(
        hidden_sizes=[1024, 1024], output_activation_function=nn.Tanh.__name__
    )
    critic_config: MLPConfig = MLPConfig(hidden_sizes=[1024, 1024])


###################################
#         SAC Algorithms          #
###################################
class SACConfig(AlgorithmConfig):
    algorithm: str = Field("SAC", Literal=True)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    reward_scale: float = 1.0
    log_std_bounds: list[float] = [-20, 2]
    policy_update_freq: int = 1
    target_update_freq: int = 1
    actor_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])
    critic_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])


class DynaSAC_SASConfig(AlgorithmConfig):
    algorithm: str = Field("DynaSAC_SAS", Literal=True)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    reward_scale: float = 1.0
    log_std_bounds: list[float] = [-20, 2]
    policy_update_freq: int = 1
    target_update_freq: int = 1
    actor_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])
    critic_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])

    num_models: int = 5
    world_model_lr: float = 0.001
    horizon: int = 3
    num_samples: int = 10


class DynaSAC_NSConfig(AlgorithmConfig):
    algorithm: str = Field("DynaSAC_NS", Literal=True)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    reward_scale: float = 1.0
    log_std_bounds: list[float] = [-20, 2]
    policy_update_freq: int = 1
    target_update_freq: int = 1
    actor_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])
    critic_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])

    num_models: int = 5
    world_model_lr: float = 0.001
    horizon: int = 3
    num_samples: int = 10


class DynaSAC_BoundedNSConfig(AlgorithmConfig):
    algorithm: str = Field("DynaSAC_BoundedNS", Literal=True)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    reward_scale: float = 1.0
    log_std_bounds: list[float] = [-20, 2]
    policy_update_freq: int = 1
    target_update_freq: int = 1
    actor_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])
    critic_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])

    num_models: int = 5
    world_model_lr: float = 0.001
    horizon: int = 3
    num_samples: int = 10

    threshold: float = 0.1


class STEVE_MEANConfig(AlgorithmConfig):
    algorithm: str = Field("DynaSAC_NS", Literal=True)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    reward_scale: float = 1.0
    log_std_bounds: list[float] = [-20, 2]
    policy_update_freq: int = 1
    target_update_freq: int = 1
    actor_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])
    critic_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])

    num_models: int = 5
    world_model_lr: float = 0.001
    horizon: int = 3
    num_samples: int = 10


class DynaSAC_SAS_Immersive_WeightConfig(AlgorithmConfig):
    algorithm: str = Field("DynaSAC_IWNS", Literal=True)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    reward_scale: float = 1.0
    log_std_bounds: list[float] = [-20, 2]
    policy_update_freq: int = 1
    target_update_freq: int = 1
    actor_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])
    critic_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])

    num_models: int = 5
    world_model_lr: float = 0.001
    horizon: int = 3
    num_samples: int = 10

    threshold: float = 0.1
    reweight_actor: bool = False


class DynaSAC_BIVReweightConfig(AlgorithmConfig):
    algorithm: str = Field("DynaSAC_BIVNS", Literal=True)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    reward_scale: float = 1.0
    log_std_bounds: list[float] = [-20, 2]
    policy_update_freq: int = 1
    target_update_freq: int = 1
    actor_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])
    critic_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])

    num_models: int = 5
    world_model_lr: float = 0.001
    horizon: int = 3
    num_samples: int = 10

    threshold: float = 0.1
    reweight_actor: bool = False


class DynaSAC_SUNRISEReweightConfig(AlgorithmConfig):
    algorithm: str = Field("DynaSAC_SUNRISENS", Literal=True)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    reward_scale: float = 1.0
    log_std_bounds: list[float] = [-20, 2]
    policy_update_freq: int = 1
    target_update_freq: int = 1
    actor_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])
    critic_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])

    num_models: int = 5
    world_model_lr: float = 0.001
    horizon: int = 3
    num_samples: int = 10

    threshold: float = 0.1
    reweight_actor: bool = False


class DynaSAC_UWACReweightConfig(AlgorithmConfig):
    algorithm: str = Field("DynaSAC_UWACNS", Literal=True)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    reward_scale: float = 1.0
    log_std_bounds: list[float] = [-20, 2]
    policy_update_freq: int = 1
    target_update_freq: int = 1
    actor_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])
    critic_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])

    num_models: int = 5
    world_model_lr: float = 0.001
    horizon: int = 3
    num_samples: int = 10

    threshold: float = 0.1
    reweight_actor: bool = False
