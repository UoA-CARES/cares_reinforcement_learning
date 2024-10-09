from typing import List, Optional

from pydantic import BaseModel, Field

from cares_reinforcement_learning.encoders.configurations import (
    AEConfig,
    VAEConfig,
    VanillaAEConfig,
)

# pylint disbale-next=unused-import

# NOTE: If a parameter is a list then don't wrap with Optional leave as implicit optional - List[type] = default


class SubscriptableClass(BaseModel):
    def __getitem__(self, item):
        return getattr(self, item)


class EnvironmentConfig(SubscriptableClass):
    task: str


class TrainingConfig(SubscriptableClass):
    """
    Configuration class for training.

    Attributes:
        seeds (List[int]): List of random seeds for reproducibility. Default is [10].
        plot_frequency (Optional[int]): Frequency at which to plot training progress. Default is 100.
        checkpoint_frequency (Optional[int]): Frequency at which to save model checkpoints. Default is 100.
        number_steps_per_evaluation (Optional[int]): Number of steps per evaluation. Default is 10000.
        number_eval_episodes (Optional[int]): Number of episodes to evaluate during training. Default is 10.
    """

    seeds: List[int] = [10]
    plot_frequency: Optional[int] = 100
    checkpoint_frequency: Optional[int] = 100
    number_steps_per_evaluation: Optional[int] = 10000
    number_eval_episodes: Optional[int] = 10


class AlgorithmConfig(SubscriptableClass):
    """
    Configuration class for the algorithm.

    These attributes are common to all algorithms. They can be overridden by the specific algorithm configuration.

    Attributes:
        algorithm (str): Name of the algorithm to be used.
        G (Optional[int]): Updates per step UTD-raio, for the actor and critic.
        G_model (Optional[int]): Updates per step UTD-ratio for MBRL.
        buffer_size (Optional[int]): Size of the memory buffer.
        batch_size (Optional[int]): Size of the training batch.
        max_steps_exploration (Optional[int]): Maximum number of steps for exploration.
        max_steps_training (Optional[int]): Maximum number of steps for training.
        number_steps_per_train_policy (Optional[int]): Number of steps per updating the training policy.

        min_noise (Optional[float]): Minimum noise value.
        noise_scale (Optional[float]): Noise scale.
        noise_decay (Optional[float]): Noise decay.

        image_observation (Optional[int]): Whether the observation is an image.

        hidden_size (List[int]): List of hidden layer sizes - e.g. [256, 256].
    """

    algorithm: str = Field(description="Name of the algorithm to be used")
    G: Optional[int] = 1
    G_model: Optional[int] = 1
    buffer_size: Optional[int] = 1000000
    batch_size: Optional[int] = 256
    max_steps_exploration: Optional[int] = 1000
    max_steps_training: Optional[int] = 1000000
    number_steps_per_train_policy: Optional[int] = 1

    min_noise: Optional[float] = 0.0
    noise_scale: Optional[float] = 0.1
    noise_decay: Optional[float] = 1.0

    image_observation: Optional[int] = 0

    hidden_size: List[int] = None


class DQNConfig(AlgorithmConfig):
    algorithm: str = Field("DQN", Literal=True)
    lr: Optional[float] = 1e-3
    gamma: Optional[float] = 0.99

    exploration_min: Optional[float] = 1e-3
    exploration_decay: Optional[float] = 0.95


class DuelingDQNConfig(AlgorithmConfig):
    algorithm: str = Field("DuelingDQN", Literal=True)
    lr: Optional[float] = 1e-3
    gamma: Optional[float] = 0.99

    exploration_min: Optional[float] = 1e-3
    exploration_decay: Optional[float] = 0.95


class DoubleDQNConfig(AlgorithmConfig):
    algorithm: str = Field("DoubleDQN", Literal=True)
    lr: Optional[float] = 1e-3
    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005

    exploration_min: Optional[float] = 1e-3
    exploration_decay: Optional[float] = 0.95


class PPOConfig(AlgorithmConfig):
    algorithm: str = Field("PPO", Literal=True)
    actor_lr: Optional[float] = 1e-4
    critic_lr: Optional[float] = 1e-3

    gamma: Optional[float] = 0.99
    eps_clip: Optional[float] = 0.2
    updates_per_iteration: Optional[int] = 10

    max_steps_per_batch: Optional[int] = 5000


class DDPGConfig(AlgorithmConfig):
    algorithm: str = Field("DDPG", Literal=True)
    actor_lr: Optional[float] = 1e-4
    critic_lr: Optional[float] = 1e-3

    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005


class TD3Config(AlgorithmConfig):
    algorithm: str = Field("TD3", Literal=True)
    actor_lr: Optional[float] = 3e-4
    critic_lr: Optional[float] = 3e-4

    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005


class TD3AEConfig(AlgorithmConfig):
    algorithm: str = Field("TD3AE", Literal=True)

    image_observation: Optional[int] = 1
    batch_size: Optional[int] = 128

    actor_lr: Optional[float] = 1e-3
    critic_lr: Optional[float] = 1e-3
    alpha_lr: Optional[float] = 1e-4

    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005

    encoder_tau: Optional[float] = 0.05
    decoder_update_freq: Optional[int] = 1

    vector_observation: Optional[int] = 0

    autoencoder_config: Optional[VanillaAEConfig] = VanillaAEConfig(
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
    actor_lr: Optional[float] = 3e-4
    critic_lr: Optional[float] = 3e-4
    alpha_lr: Optional[float] = 3e-4

    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005
    reward_scale: Optional[float] = 1.0

    log_std_bounds: List[float] = [-20, 2]


class SACAEConfig(AlgorithmConfig):
    algorithm: str = Field("SACAE", Literal=True)

    image_observation: Optional[int] = 1
    batch_size: Optional[int] = 128

    actor_lr: Optional[float] = 1e-3
    critic_lr: Optional[float] = 1e-3
    alpha_lr: Optional[float] = 1e-4

    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005
    reward_scale: Optional[float] = 1.0

    log_std_bounds: List[float] = [-20, 2]

    encoder_tau: Optional[float] = 0.05
    decoder_update_freq: Optional[int] = 1

    vector_observation: Optional[int] = 0

    autoencoder_config: Optional[VanillaAEConfig] = VanillaAEConfig(
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
    actor_lr: Optional[float] = 3e-4
    critic_lr: Optional[float] = 3e-4
    alpha_lr: Optional[float] = 3e-4

    batch_size = 64

    target_entropy_multiplier = 0.98

    max_steps_exploration = 20000
    number_steps_per_train_policy = 4

    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005
    reward_scale: Optional[float] = 1.0


class DynaSACConfig(AlgorithmConfig):
    algorithm: str = Field("DynaSAC", Literal=True)
    actor_lr: Optional[float] = 3e-4
    critic_lr: Optional[float] = 3e-4

    alpha_lr: Optional[float] = 3e-4

    # TODO this bool doesn't work as expected - needs to be int 1/0
    use_bounded_active: Optional[bool] = False
    num_models: Optional[int] = 5

    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005

    log_std_bounds: List[float] = [-20, 2]

    horizon: Optional[int] = 3
    num_samples: Optional[int] = 10
    world_model_lr: Optional[float] = 0.001


class NaSATD3Config(AlgorithmConfig):
    algorithm: str = Field("NaSATD3", Literal=True)

    image_observation: Optional[int] = 1

    actor_lr: Optional[float] = 1e-4
    critic_lr: Optional[float] = 1e-3
    epm_lr: Optional[float] = 1e-4

    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005
    ensemble_size: Optional[int] = 3

    intrinsic_on: Optional[int] = 1

    vector_observation: Optional[int] = 0

    autoencoder_config: Optional[AEConfig] = VanillaAEConfig(
        latent_dim=200,
        num_layers=4,
        num_filters=32,
        kernel_size=3,
        latent_lambda=1e-6,
        encoder_optim_kwargs={"lr": 1e-3},
        decoder_optim_kwargs={"lr": 1e-3, "weight_decay": 1e-7},
    )

    # autoencoder_config: Optional[AEConfig] = VAEConfig(
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
    actor_lr: Optional[float] = 3e-4
    critic_lr: Optional[float] = 3e-4

    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005
    ensemble_size: Optional[int] = 10
    num_sample_critics: Optional[int] = 2

    G: Optional[int] = 20


class TQCConfig(AlgorithmConfig):
    algorithm: str = Field("TQC", Literal=True)
    actor_lr: Optional[float] = 3e-4
    critic_lr: Optional[float] = 3e-4
    alpha_lr: Optional[float] = 3e-4

    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005
    top_quantiles_to_drop: Optional[int] = 2
    num_quantiles: Optional[int] = 25
    num_nets: Optional[int] = 5

    log_std_bounds: List[float] = [-20, 2]


class CTD4Config(AlgorithmConfig):
    algorithm: str = Field("CTD4", Literal=True)

    actor_lr: Optional[float] = 1e-4
    critic_lr: Optional[float] = 1e-3
    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005
    ensemble_size: Optional[int] = 3

    min_noise: Optional[float] = 0.0
    noise_decay: Optional[float] = 0.999999
    noise_scale: Optional[float] = 0.1

    fusion_method: Optional[str] = "kalman"  # kalman, minimum, average


class PERTD3Config(AlgorithmConfig):
    algorithm: str = Field("PERTD3", Literal=True)

    actor_lr: Optional[float] = 3e-4
    critic_lr: Optional[float] = 3e-4
    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005

    beta: Optional[float] = 0.4
    per_alpha: Optional[float] = 0.6
    min_priority: Optional[float] = 1e-6


class PERSACConfig(AlgorithmConfig):
    algorithm: str = Field("PERSAC", Literal=True)

    actor_lr: Optional[float] = 3e-4
    critic_lr: Optional[float] = 3e-4
    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005

    beta: Optional[float] = 0.4
    per_alpha: Optional[float] = 0.6
    min_priority: Optional[float] = 1e-6

    log_std_bounds: List[float] = [-20, 2]


class LAPTD3Config(AlgorithmConfig):
    algorithm: str = Field("LAPTD3", Literal=True)

    actor_lr: Optional[float] = 3e-4
    critic_lr: Optional[float] = 3e-4
    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005

    beta: Optional[float] = 0.4
    per_alpha: Optional[float] = 0.4
    min_priority: Optional[float] = 1.0


class LAPSACConfig(AlgorithmConfig):
    algorithm: str = Field("LAPSAC", Literal=True)

    actor_lr: Optional[float] = 3e-4
    critic_lr: Optional[float] = 3e-4
    alpha_lr: Optional[float] = 3e-4

    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005
    per_alpha: Optional[float] = 0.6
    reward_scale: Optional[float] = 1.0
    min_priority: Optional[float] = 1.0

    log_std_bounds: List[float] = [-20, 2]


class PALTD3Config(AlgorithmConfig):
    algorithm: str = Field("PALTD3", Literal=True)

    actor_lr: Optional[float] = 3e-4
    critic_lr: Optional[float] = 3e-4
    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005

    beta: Optional[float] = 0.4
    per_alpha: Optional[float] = 0.4
    min_priority: Optional[float] = 1.0


class LA3PTD3Config(AlgorithmConfig):
    algorithm: str = Field("LA3PTD3", Literal=True)

    actor_lr: Optional[float] = 3e-4
    critic_lr: Optional[float] = 3e-4
    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005

    beta: Optional[float] = 0.4
    per_alpha: Optional[float] = 0.4
    min_priority: Optional[float] = 1.0
    prioritized_fraction: Optional[float] = 0.5


class LA3PSACConfig(AlgorithmConfig):
    algorithm: str = Field("LA3PSAC", Literal=True)

    actor_lr: Optional[float] = 3e-4
    critic_lr: Optional[float] = 3e-4
    alpha_lr: Optional[float] = 3e-4
    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005
    reward_scale: Optional[float] = 5.0

    beta: Optional[float] = 0.4
    per_alpha: Optional[float] = 0.4
    min_priority: Optional[float] = 1.0
    prioritized_fraction: Optional[float] = 0.5

    log_std_bounds: List[float] = [-20, 2]


class MAPERTD3Config(AlgorithmConfig):
    algorithm: str = Field("MAPERTD3", Literal=True)

    max_steps_exploration: Optional[int] = 10000

    batch_size: Optional[int] = 100

    actor_lr: Optional[float] = 1e-3
    critic_lr: Optional[float] = 1e-3
    gamma: Optional[float] = 0.98
    tau: Optional[float] = 0.005

    beta: Optional[float] = 1.0
    per_alpha: Optional[float] = 0.7
    min_priority: Optional[float] = 1e-6

    G: Optional[int] = 64
    number_steps_per_train_policy: Optional[int] = 64


class MAPERSACConfig(AlgorithmConfig):
    algorithm: str = Field("MAPERSAC", Literal=True)

    max_steps_exploration: Optional[int] = 10000

    actor_lr: Optional[float] = 7.3e-4
    critic_lr: Optional[float] = 7.3e-4
    alpha_lr: Optional[float] = 7.3e-4
    gamma: Optional[float] = 0.98
    tau: Optional[float] = 0.02

    beta: Optional[float] = 0.4
    per_alpha: Optional[float] = 0.7
    min_priority: Optional[float] = 1e-6

    G: Optional[int] = 64
    number_steps_per_train_policy: Optional[int] = 64

    hidden_size: List[int] = [400, 300]
    log_std_bounds: List[float] = [-20, 2]


class RDTD3Config(AlgorithmConfig):
    algorithm: str = Field("RDTD3", Literal=True)

    actor_lr: Optional[float] = 3e-4
    critic_lr: Optional[float] = 3e-4
    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005

    beta: Optional[float] = 0.4
    per_alpha: Optional[float] = 0.7
    min_priority: Optional[float] = 1.0


class RDSACConfig(AlgorithmConfig):
    algorithm: str = Field("RDSAC", Literal=True)

    actor_lr: Optional[float] = 3e-4
    critic_lr: Optional[float] = 3e-4
    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005

    beta: Optional[float] = 0.4
    per_alpha: Optional[float] = 0.7
    min_priority: Optional[float] = 1.0

    log_std_bounds: List[float] = [-20, 2]
