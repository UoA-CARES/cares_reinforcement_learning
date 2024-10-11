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
        plot_frequency (int]): Frequency at which to plot training progress. Default is 100.
        checkpoint_frequency (int]): Frequency at which to save model checkpoints. Default is 100.
        number_steps_per_evaluation (int]): Number of steps per evaluation. Default is 10000.
        number_eval_episodes (int]): Number of episodes to evaluate during training. Default is 10.
    """

    seeds: list[int] = [10]
    plot_frequency: int = 100
    checkpoint_frequency: int = 100
    number_steps_per_evaluation: int = 10000
    number_eval_episodes: int = 10


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

        hidden_size (list[int]): list of hidden layer sizes - e.g. [256, 256].
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

    hidden_size: list[int] | None = None


class DQNConfig(AlgorithmConfig):
    algorithm: str = Field("DQN", Literal=True)
    lr: float = 1e-3
    gamma: float = 0.99

    exploration_min: float = 1e-3
    exploration_decay: float = 0.95


class DuelingDQNConfig(AlgorithmConfig):
    algorithm: str = Field("DuelingDQN", Literal=True)
    lr: float = 1e-3
    gamma: float = 0.99

    exploration_min: float = 1e-3
    exploration_decay: float = 0.95


class DoubleDQNConfig(AlgorithmConfig):
    algorithm: str = Field("DoubleDQN", Literal=True)
    lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005

    exploration_min: float = 1e-3
    exploration_decay: float = 0.95


class PPOConfig(AlgorithmConfig):
    algorithm: str = Field("PPO", Literal=True)
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3

    gamma: float = 0.99
    eps_clip: float = 0.2
    updates_per_iteration: int = 10

    max_steps_per_batch: int = 5000


class DDPGConfig(AlgorithmConfig):
    algorithm: str = Field("DDPG", Literal=True)
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3

    gamma: float = 0.99
    tau: float = 0.005


class TD3Config(AlgorithmConfig):
    algorithm: str = Field("TD3", Literal=True)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    gamma: float = 0.99
    tau: float = 0.005


class TD3AEConfig(AlgorithmConfig):
    algorithm: str = Field("TD3AE", Literal=True)

    image_observation: int = 1
    batch_size: int = 128

    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    alpha_lr: float = 1e-4

    gamma: float = 0.99
    tau: float = 0.005

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


class SACConfig(AlgorithmConfig):
    algorithm: str = Field("SAC", Literal=True)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4

    gamma: float = 0.99
    tau: float = 0.005
    reward_scale: float = 1.0

    log_std_bounds: list[float] = [-20, 2]


class SACAEConfig(AlgorithmConfig):
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


class SACDConfig(AlgorithmConfig):
    algorithm: str = Field("SACD", Literal=True)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4

    batch_size = 64

    target_entropy_multiplier = 0.98

    max_steps_exploration = 20000
    number_steps_per_train_policy = 4

    gamma: float = 0.99
    tau: float = 0.005
    reward_scale: float = 1.0


class DynaSACConfig(AlgorithmConfig):
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

    horizon: int = 3
    num_samples: int = 10
    world_model_lr: float = 0.001


class NaSATD3Config(AlgorithmConfig):
    algorithm: str = Field("NaSATD3", Literal=True)

    image_observation: int = 1

    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    epm_lr: float = 1e-4

    gamma: float = 0.99
    tau: float = 0.005
    ensemble_size: int = 3

    intrinsic_on: int = 1

    vector_observation: int = 0

    autoencoder_config: VanillaAEConfig | BurgessConfig = VanillaAEConfig(
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


class REDQConfig(AlgorithmConfig):
    algorithm: str = Field("REDQ", Literal=True)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    gamma: float = 0.99
    tau: float = 0.005
    ensemble_size: int = 10
    num_sample_critics: int = 2

    G: int = 20


class TQCConfig(AlgorithmConfig):
    algorithm: str = Field("TQC", Literal=True)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4

    gamma: float = 0.99
    tau: float = 0.005
    top_quantiles_to_drop: int = 2
    num_quantiles: int = 25
    num_nets: int = 5

    log_std_bounds: list[float] = [-20, 2]


class CTD4Config(AlgorithmConfig):
    algorithm: str = Field("CTD4", Literal=True)

    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    ensemble_size: int = 3

    min_noise: float = 0.0
    noise_decay: float = 0.999999
    noise_scale: float = 0.1

    fusion_method: str = "kalman"  # kalman, minimum, average


class PERTD3Config(AlgorithmConfig):
    algorithm: str = Field("PERTD3", Literal=True)

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005

    beta: float = 0.4
    per_alpha: float = 0.6
    min_priority: float = 1e-6


class PERSACConfig(AlgorithmConfig):
    algorithm: str = Field("PERSAC", Literal=True)

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005

    beta: float = 0.4
    per_alpha: float = 0.6
    min_priority: float = 1e-6

    log_std_bounds: list[float] = [-20, 2]


class LAPTD3Config(AlgorithmConfig):
    algorithm: str = Field("LAPTD3", Literal=True)

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005

    beta: float = 0.4
    per_alpha: float = 0.4
    min_priority: float = 1.0


class LAPSACConfig(AlgorithmConfig):
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


class PALTD3Config(AlgorithmConfig):
    algorithm: str = Field("PALTD3", Literal=True)

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005

    beta: float = 0.4
    per_alpha: float = 0.4
    min_priority: float = 1.0


class LA3PTD3Config(AlgorithmConfig):
    algorithm: str = Field("LA3PTD3", Literal=True)

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005

    beta: float = 0.4
    per_alpha: float = 0.4
    min_priority: float = 1.0
    prioritized_fraction: float = 0.5


class LA3PSACConfig(AlgorithmConfig):
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


class MAPERTD3Config(AlgorithmConfig):
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


class MAPERSACConfig(AlgorithmConfig):
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

    hidden_size: list[int] = [400, 300]
    log_std_bounds: list[float] = [-20, 2]


class RDTD3Config(AlgorithmConfig):
    algorithm: str = Field("RDTD3", Literal=True)

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005

    beta: float = 0.4
    per_alpha: float = 0.7
    min_priority: float = 1.0


class RDSACConfig(AlgorithmConfig):
    algorithm: str = Field("RDSAC", Literal=True)

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005

    beta: float = 0.4
    per_alpha: float = 0.7
    min_priority: float = 1.0

    log_std_bounds: list[float] = [-20, 2]
