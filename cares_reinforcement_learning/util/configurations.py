from typing import Any, Literal

from pydantic import BaseModel, Field

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


class TrainableLayer(BaseModel):
    layer_category: Literal["trainable"] = "trainable"  # Discriminator field
    layer_type: str
    in_features: int | None = None
    out_features: int | None = None
    params: dict[str, Any] = {}


class NormLayer(BaseModel):
    layer_category: Literal["norm"] = "norm"  # Discriminator field
    layer_type: str
    in_features: int | None = None
    params: dict[str, Any] = {}


class FunctionLayer(BaseModel):
    layer_category: Literal["function"] = "function"  # Discriminator field
    layer_type: str
    params: dict[str, Any] = {}


class MLPConfig(BaseModel):
    layers: list[TrainableLayer | NormLayer | FunctionLayer]


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


# def __init__(self, start_epsilon: float, end_epsilon: float, decay_steps: int):
class DQNConfig(AlgorithmConfig):
    algorithm: str = Field("DQN", Literal=True)
    lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 1.0
    target_update_freq: int = 1000

    # Double DQN
    use_double_dqn: int = 0

    # PER
    use_per_buffer: int = 0
    min_priority: float = 1e-6
    per_alpha: float = 0.6

    # n-step
    n_step: int = 1

    max_grad_norm: float | None = None

    start_epsilon: float = 1.0
    end_epsilon: float = 1e-3
    decay_steps: int = 100000

    batch_size: int = 32

    network_config: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=64),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=64, out_features=64),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=64),
        ]
    )


class DoubleDQNConfig(DQNConfig):
    algorithm: str = Field("DoubleDQN", Literal=True)

    use_double_dqn: Literal[1] = Field(default=1, frozen=True)


class PERDQNConfig(DQNConfig):
    algorithm: str = Field("PERDQN", Literal=True)

    use_double_dqn: int = 1
    use_per_buffer: Literal[1] = Field(default=1, frozen=True)


class DuelingDQNConfig(DQNConfig):
    algorithm: str = Field("DuelingDQN", Literal=True)
    lr: float = 5e-4
    gamma: float = 0.99
    tau: float = 0.005
    target_update_freq: int = 1

    max_grad_norm: float | None = 10.0

    use_double_dqn: int = 1

    feature_layer_config: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=128),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=128, out_features=128),
            FunctionLayer(layer_type="ReLU"),
        ]
    )

    value_stream_config: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", in_features=128, out_features=128),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=128, out_features=1),
        ]
    )

    advantage_stream_config: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", in_features=128, out_features=128),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=128),
        ]
    )


class NoisyNetConfig(DQNConfig):
    algorithm: str = Field("NoisyNet", Literal=True)

    max_grad_norm: float | None = 10.0

    start_epsilon: float = 0.0
    end_epsilon: float = 0.0
    decay_steps: int = 0

    use_double_dqn: int = 1

    network_config: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=64),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(
                layer_type="NoisyLinear",
                in_features=64,
                out_features=64,
                params={"sigma_init": 1.0},
            ),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(
                layer_type="NoisyLinear", in_features=64, params={"sigma_init": 0.5}
            ),
        ]
    )


class C51Config(DQNConfig):
    algorithm: str = Field("C51", Literal=True)

    num_atoms: int = 51
    v_min: float = 0.0
    v_max: float = 200.0


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
        layers=[
            TrainableLayer(layer_type="Linear", out_features=1024),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=1024, out_features=1024),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=1024),
            FunctionLayer(layer_type="Tanh"),
        ]
    )

    critic_config: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=1024),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=1024, out_features=1024),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=1024, out_features=1),
        ]
    )


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

    actor_config: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=256),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
            FunctionLayer(layer_type="ReLU"),
        ]
    )

    critic_config: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=256),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=256, out_features=1),
        ]
    )


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

    actor_config: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=1024),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=1024, out_features=1024),
            FunctionLayer(layer_type="ReLU"),
        ]
    )

    critic_config: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=1024),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=1024, out_features=1024),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=1024, out_features=1),
        ]
    )

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

    actor_config: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=256),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
            FunctionLayer(layer_type="ReLU"),
        ]
    )

    critic_config: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=512),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=512, out_features=512),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=512, out_features=512),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=512),
        ]
    )


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

    actor_config: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=400),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=400, out_features=300),
            FunctionLayer(layer_type="ReLU"),
        ]
    )

    critic_config: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=400),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=400, out_features=300),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=300),
        ]
    )


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

    critic_config: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=256),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=256),
        ]
    )


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
        layers=[
            NormLayer(layer_type="BatchRenorm1d", params={"momentum": 0.01}),
            TrainableLayer(layer_type="Linear", out_features=256),
            FunctionLayer(layer_type="ReLU"),
            NormLayer(layer_type="BatchRenorm1d", params={"momentum": 0.01}),
            TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
            FunctionLayer(layer_type="ReLU"),
            NormLayer(layer_type="BatchRenorm1d", params={"momentum": 0.01}),
        ]
    )

    critic_config: MLPConfig = MLPConfig(
        layers=[
            NormLayer(layer_type="BatchRenorm1d", params={"momentum": 0.01}),
            TrainableLayer(layer_type="Linear", out_features=2048),
            FunctionLayer(layer_type="ReLU"),
            NormLayer(layer_type="BatchRenorm1d", params={"momentum": 0.01}),
            TrainableLayer(layer_type="Linear", in_features=2048, out_features=2048),
            FunctionLayer(layer_type="ReLU"),
            NormLayer(layer_type="BatchRenorm1d", params={"momentum": 0.01}),
            TrainableLayer(layer_type="Linear", in_features=2048, out_features=1),
        ]
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

    critic_config: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=256),
            FunctionLayer(layer_type="Dropout", params={"p": 0.005}),
            NormLayer(layer_type="LayerNorm"),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
            FunctionLayer(layer_type="Dropout", params={"p": 0.005}),
            NormLayer(layer_type="LayerNorm"),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=256, out_features=1),
        ]
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

    horizon: int = 3
    num_samples: int = 10
    world_model_lr: float = 0.001


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

    actor_config: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=512),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=512, out_features=512),
            FunctionLayer(layer_type="ReLU"),
        ]
    )

    critic_config: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=512),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=512, out_features=512),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=512),
        ]
    )


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
        layers=[
            TrainableLayer(layer_type="Linear", out_features=1024),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=1024, out_features=1024),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=1024),
            FunctionLayer(layer_type="Tanh"),
        ]
    )

    critic_config: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=1024),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=1024, out_features=1024),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=1024, out_features=1),
        ]
    )


class TD3Config(AlgorithmConfig):
    algorithm: str = Field("TD3", Literal=True)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    gamma: float = 0.99
    tau: float = 0.005

    policy_update_freq: int = 2

    actor_config: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=256),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=256),
            FunctionLayer(layer_type="Tanh"),
        ]
    )

    critic_config: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=256),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=256, out_features=1),
        ]
    )


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
        layers=[
            TrainableLayer(layer_type="Linear", out_features=1024),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=1024, out_features=1024),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=1024),
            FunctionLayer(layer_type="Tanh"),
        ]
    )

    critic_config: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=1024),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=1024, out_features=1024),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=1024, out_features=1),
        ]
    )

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
        layers=[
            TrainableLayer(layer_type="Linear", out_features=1024),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=1024, out_features=1024),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=1024),
            FunctionLayer(layer_type="Tanh"),
        ]
    )

    critic_config: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=1024),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=1024, out_features=1024),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=1024, out_features=1),
        ]
    )

    epm_config: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=512),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=512, out_features=512),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=512),
        ]
    )

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

    critic_config: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=256),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=256),
        ]
    )


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

    critic_config: MLPConfig = MLPConfig(
        layers=[
            TrainableLayer(layer_type="Linear", out_features=256),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
            FunctionLayer(layer_type="ReLU"),
            TrainableLayer(layer_type="Linear", in_features=256),
        ]
    )


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

    fusion_method: str = "kalman"  # kalman, minimum, average
