from typing import Any

import pydantic
from pydantic import BaseModel, Field
from torch import nn

from cares_reinforcement_learning.encoders.configurations import (
    BurgessConfig,
    VanillaAEConfig,
)

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
        record_eval_video (int]): Whether to record a video of the evaluation. Default is 1.
    """

    seeds: list[int] = [10]
    number_steps_per_evaluation: int = 10000
    number_eval_episodes: int = 10
    record_eval_video: int = 1


class MLPConfig(SubscriptableClass):
    hidden_sizes: list[int]

    input_layer: str = ""
    input_layer_args: dict[str, Any] = Field(default_factory=dict)

    linear_layer_type: str = "linear"
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
    G_model: int = 1
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


class NoisyNetConfig(AlgorithmConfig):
    algorithm: str = Field("NoisyNet", Literal=True)
    lr: float = 1e-3
    gamma: float = 0.99

    exploration_min: float = 1e-3
    exploration_decay: float = 0.95

    network_config: MLPConfig = MLPConfig(
        hidden_sizes=[512, 512],
        linear_layer_type="noisy",
        linear_layer_args={"std_init": 0.1},
    )


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


class SACDConfig(AlgorithmConfig):
    algorithm: str = Field("SACD", Literal=True)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4

    batch_size: int = 64

    target_entropy_multiplier: float = 0.98

    max_steps_exploration: int = 20000
    number_steps_per_train_policy: int = 4

    gamma: float = 0.99
    tau: float = 0.005
    reward_scale: float = 1.0

    policy_update_freq: int = 1
    target_update_freq: int = 1

    actor_config: MLPConfig = MLPConfig(hidden_sizes=[512, 512])
    critic_config: MLPConfig = MLPConfig(hidden_sizes=[512, 512])


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


class SACAEConfig(SACConfig):
    algorithm: str = Field("SACAE", Literal=True)

    image_observation: int = 1
    batch_size: int = 128

    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    alpha_lr: float = 1e-4

    gamma: float = 0.99
    tau: float = 0.005
    reward_scale: float = 1.0

    log_std_bounds: list[float] = [-20, 2]

    policy_update_freq: int = 2
    target_update_freq: int = 2

    actor_config: MLPConfig = MLPConfig(hidden_sizes=[1024, 1024])
    critic_config: MLPConfig = MLPConfig(hidden_sizes=[1024, 1024])

    encoder_tau: float = 0.05
    decoder_update_freq: int = 1

    vector_observation: int = 0

    autoencoder_config: VanillaAEConfig = VanillaAEConfig(
        latent_dim=50,
        num_layers=4,
        num_filters=32,
        kernel_size=3,
        latent_lambda=1e-6,
        encoder_optim_kwargs={"lr": 1e-3},
        decoder_optim_kwargs={"lr": 1e-3, "weight_decay": 1e-7},
    )


class PERSACConfig(SACConfig):
    algorithm: str = Field("PERSAC", Literal=True)

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005

    beta: float = 0.4
    per_alpha: float = 0.6
    min_priority: float = 1e-6

    log_std_bounds: list[float] = [-20, 2]

    policy_update_freq: int = 1
    target_update_freq: int = 1

    actor_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])
    critic_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])


class REDQConfig(SACConfig):
    algorithm: str = Field("REDQ", Literal=True)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    gamma: float = 0.99
    tau: float = 0.005
    ensemble_size: int = 10
    num_sample_critics: int = 2

    G: int = 20

    policy_update_freq: int = 1
    target_update_freq: int = 1

    actor_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])
    critic_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])


class TQCConfig(SACConfig):
    algorithm: str = Field("TQC", Literal=True)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4

    gamma: float = 0.99
    tau: float = 0.005
    top_quantiles_to_drop: int = 2
    num_quantiles: int = 25
    num_critics: int = 5

    log_std_bounds: list[float] = [-20, 2]

    policy_update_freq: int = 1
    target_update_freq: int = 1

    actor_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])
    critic_config: MLPConfig = MLPConfig(hidden_sizes=[512, 512, 512])


class LAPSACConfig(SACConfig):
    algorithm: str = Field("LAPSAC", Literal=True)

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4

    gamma: float = 0.99
    tau: float = 0.005
    per_alpha: float = 0.6
    reward_scale: float = 1.0
    min_priority: float = 1.0

    log_std_bounds: list[float] = [-20, 2]

    policy_update_freq: int = 1
    target_update_freq: int = 1

    actor_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])
    critic_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])


class LA3PSACConfig(SACConfig):
    algorithm: str = Field("LA3PSAC", Literal=True)

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    reward_scale: float = 5.0

    beta: float = 0.4
    per_alpha: float = 0.4
    min_priority: float = 1.0
    prioritized_fraction: float = 0.5

    log_std_bounds: list[float] = [-20, 2]

    target_update_freq: int = 1

    actor_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])
    critic_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])


class MAPERSACConfig(SACConfig):
    algorithm: str = Field("MAPERSAC", Literal=True)

    max_steps_exploration: int = 10000

    actor_lr: float = 7.3e-4
    critic_lr: float = 7.3e-4
    alpha_lr: float = 7.3e-4
    gamma: float = 0.98
    tau: float = 0.02

    beta: float = 0.4
    per_alpha: float = 0.7
    min_priority: float = 1e-6

    G: int = 64
    number_steps_per_train_policy: int = 64

    log_std_bounds: list[float] = [-20, 2]

    policy_update_freq: int = 1
    target_update_freq: int = 1

    actor_config: MLPConfig = MLPConfig(hidden_sizes=[400, 300])
    critic_config: MLPConfig = MLPConfig(hidden_sizes=[400, 300])


class RDSACConfig(SACConfig):
    algorithm: str = Field("RDSAC", Literal=True)

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005

    beta: float = 0.4
    per_alpha: float = 0.7
    min_priority: float = 1.0

    log_std_bounds: list[float] = [-20, 2]

    policy_update_freq: int = 1
    target_update_freq: int = 1

    actor_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])
    critic_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])


class CrossQConfig(AlgorithmConfig):
    algorithm: str = Field("CrossQ", Literal=True)
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    alpha_lr: float = 1e-3

    gamma: float = 0.99
    reward_scale: float = 1.0

    log_std_bounds: list[float] = [-20, 2]

    policy_update_freq: int = 3

    actor_config: MLPConfig = MLPConfig(
        input_layer="BatchRenorm1d",
        linear_layer_args={"bias": False},
        hidden_sizes=[256, 256],
        batch_layer="BatchRenorm1d",
        batch_layer_args={"momentum": 0.01},
        layer_order=["activation", "batch"],
    )
    critic_config: MLPConfig = MLPConfig(
        input_layer="BatchRenorm1d",
        linear_layer_args={"bias": False},
        hidden_sizes=[2048, 2048],
        batch_layer="BatchRenorm1d",
        batch_layer_args={"momentum": 0.01},
        layer_order=["activation", "batch"],
    )


class DroQConfig(SACConfig):
    algorithm: str = Field("DroQ", Literal=True)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4

    gamma: float = 0.99
    tau: float = 0.005
    reward_scale: float = 1.0

    G: int = 20

    log_std_bounds: list[float] = [-20, 2]

    policy_update_freq: int = 1
    target_update_freq: int = 1

    hidden_size_actor: list[int] = [256, 256]
    hidden_size_critic: list[int] = [256, 256]

    actor_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])
    critic_config: MLPConfig = MLPConfig(
        hidden_sizes=[256, 256],
        dropout_layer="Dropout",
        dropout_layer_args={"p": 0.005},
        norm_layer="LayerNorm",
        layer_order=["dropout", "layernorm", "activation"],
    )


class DynaSACConfig(SACConfig):
    algorithm: str = Field("DynaSAC", Literal=True)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    alpha_lr: float = 3e-4

    # TODO this bool doesn't work as expected - needs to be int 1/0
    use_bounded_active: bool = False
    num_models: int = 5

    gamma: float = 0.99
    tau: float = 0.005

    log_std_bounds: list[float] = [-20, 2]

    policy_update_freq: int = 1
    target_update_freq: int = 1

    actor_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])
    critic_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])

    horizon: int = 3
    num_samples: int = 10
    world_model_lr: float = 0.001


###################################
#         TD3 Algorithms          #
###################################


class DDPGConfig(AlgorithmConfig):
    algorithm: str = Field("DDPG", Literal=True)
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3

    gamma: float = 0.99
    tau: float = 0.005

    actor_config: MLPConfig = MLPConfig(
        hidden_sizes=[1024, 1024], output_activation_function=nn.Tanh.__name__
    )
    critic_config: MLPConfig = MLPConfig(hidden_sizes=[1024, 1024])


class TD3Config(AlgorithmConfig):
    algorithm: str = Field("TD3", Literal=True)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    gamma: float = 0.99
    tau: float = 0.005

    policy_update_freq: int = 2

    actor_config: MLPConfig = MLPConfig(
        hidden_sizes=[256, 256], output_activation_function=nn.Tanh.__name__
    )
    critic_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])


class TD3AEConfig(TD3Config):
    algorithm: str = Field("TD3AE", Literal=True)

    image_observation: int = 1
    batch_size: int = 128

    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    alpha_lr: float = 1e-4

    gamma: float = 0.99
    tau: float = 0.005

    policy_update_freq: int = 2

    actor_config: MLPConfig = MLPConfig(
        hidden_sizes=[1024, 1024], output_activation_function=nn.Tanh.__name__
    )
    critic_config: MLPConfig = MLPConfig(hidden_sizes=[1024, 1024])

    encoder_tau: float = 0.05
    decoder_update_freq: int = 1

    vector_observation: int = 0

    autoencoder_config: VanillaAEConfig = VanillaAEConfig(
        latent_dim=50,
        num_layers=4,
        num_filters=32,
        kernel_size=3,
        latent_lambda=1e-6,
        encoder_optim_kwargs={"lr": 1e-3},
        decoder_optim_kwargs={"lr": 1e-3, "weight_decay": 1e-7},
    )


class NaSATD3Config(TD3Config):
    algorithm: str = Field("NaSATD3", Literal=True)

    image_observation: int = 1

    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    epm_lr: float = 1e-4

    gamma: float = 0.99
    tau: float = 0.005
    ensemble_size: int = 3

    policy_update_freq: int = 2

    actor_config: MLPConfig = MLPConfig(
        hidden_sizes=[1024, 1024], output_activation_function=nn.Tanh.__name__
    )
    critic_config: MLPConfig = MLPConfig(hidden_sizes=[1024, 1024])
    epm_config: MLPConfig = MLPConfig(hidden_sizes=[512, 512])

    intrinsic_on: int = 1

    vector_observation: int = 0

    autoencoder_config: VanillaAEConfig = VanillaAEConfig(
        latent_dim=200,
        num_layers=4,
        num_filters=32,
        kernel_size=3,
        latent_lambda=1e-6,
        encoder_optim_kwargs={"lr": 1e-3},
        decoder_optim_kwargs={"lr": 1e-3, "weight_decay": 1e-7},
    )

    # autoencoder_config: AEConfig] = VAEConfig(
    #     latent_dim=200,
    #     num_layers=4,
    #     num_filters=32,
    #     kernel_size=3,
    #     latent_lambda=1e-6,
    #     encoder_optim_kwargs={"lr": 1e-3},
    #     decoder_optim_kwargs={"lr": 1e-3, "weight_decay": 1e-7},
    # )


class PERTD3Config(TD3Config):
    algorithm: str = Field("PERTD3", Literal=True)

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005

    beta: float = 0.4
    per_alpha: float = 0.6
    min_priority: float = 1e-6

    policy_update_freq: int = 2

    actor_config: MLPConfig = MLPConfig(
        hidden_sizes=[256, 256], output_activation_function=nn.Tanh.__name__
    )
    critic_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])


class LAPTD3Config(TD3Config):
    algorithm: str = Field("LAPTD3", Literal=True)

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005

    beta: float = 0.4
    per_alpha: float = 0.4
    min_priority: float = 1.0

    policy_update_freq: int = 2

    actor_config: MLPConfig = MLPConfig(
        hidden_sizes=[256, 256], output_activation_function=nn.Tanh.__name__
    )
    critic_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])


class PALTD3Config(TD3Config):
    algorithm: str = Field("PALTD3", Literal=True)

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005

    beta: float = 0.4
    per_alpha: float = 0.4
    min_priority: float = 1.0

    policy_update_freq: int = 2

    actor_config: MLPConfig = MLPConfig(
        hidden_sizes=[256, 256], output_activation_function=nn.Tanh.__name__
    )
    critic_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])


class LA3PTD3Config(TD3Config):
    algorithm: str = Field("LA3PTD3", Literal=True)

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005

    beta: float = 0.4
    per_alpha: float = 0.4
    min_priority: float = 1.0
    prioritized_fraction: float = 0.5

    policy_update_freq: int = 2

    actor_config: MLPConfig = MLPConfig(
        hidden_sizes=[256, 256], output_activation_function=nn.Tanh.__name__
    )
    critic_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])


class MAPERTD3Config(TD3Config):
    algorithm: str = Field("MAPERTD3", Literal=True)

    max_steps_exploration: int = 10000

    batch_size: int = 100

    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    gamma: float = 0.98
    tau: float = 0.005

    beta: float = 1.0
    per_alpha: float = 0.7
    min_priority: float = 1e-6

    G: int = 64
    number_steps_per_train_policy: int = 64

    policy_update_freq: int = 2

    actor_config: MLPConfig = MLPConfig(
        hidden_sizes=[256, 256], output_activation_function=nn.Tanh.__name__
    )
    critic_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])


class RDTD3Config(TD3Config):
    algorithm: str = Field("RDTD3", Literal=True)

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005

    beta: float = 0.4
    per_alpha: float = 0.7
    min_priority: float = 1.0

    policy_update_freq: int = 2

    actor_config: MLPConfig = MLPConfig(
        hidden_sizes=[256, 256], output_activation_function=nn.Tanh.__name__
    )
    critic_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])


class CTD4Config(TD3Config):
    algorithm: str = Field("CTD4", Literal=True)

    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    ensemble_size: int = 3

    min_noise: float = 0.0
    noise_decay: float = 0.999999
    noise_scale: float = 0.1

    policy_update_freq: int = 2

    actor_config: MLPConfig = MLPConfig(
        hidden_sizes=[256, 256], output_activation_function=nn.Tanh.__name__
    )
    critic_config: MLPConfig = MLPConfig(hidden_sizes=[256, 256])

    fusion_method: str = "kalman"  # kalman, minimum, average
