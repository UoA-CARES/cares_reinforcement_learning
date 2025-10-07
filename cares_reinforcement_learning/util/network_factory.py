"""
This module provides functions to create different types of reinforcement learning agents
with their corresponding network architectures.
"""

import inspect
import logging
import sys

import cares_reinforcement_learning.util.configurations as acf
import cares_reinforcement_learning.util.helpers as hlp

# Disable these as this is a deliberate use of dynamic imports
# pylint: disable=import-outside-toplevel
# pylint: disable=invalid-name

###################################
#         DQN Algorithms          #
###################################


def create_DQN(observation_size, action_num, config: acf.DQNConfig):
    from cares_reinforcement_learning.algorithm.value import DQN
    from cares_reinforcement_learning.networks.DQN import Network

    network = Network(observation_size, action_num, config=config)

    device = hlp.get_device()
    agent = DQN(network=network, config=config, device=device)
    return agent


def create_PERDQN(observation_size, action_num, config: acf.PERDQNConfig):
    from cares_reinforcement_learning.algorithm.value import PERDQN
    from cares_reinforcement_learning.networks.PERDQN import Network

    network = Network(observation_size, action_num, config=config)

    device = hlp.get_device()
    agent = PERDQN(network=network, config=config, device=device)
    return agent


def create_DuelingDQN(observation_size, action_num, config: acf.DuelingDQNConfig):
    from cares_reinforcement_learning.algorithm.value import DuelingDQN
    from cares_reinforcement_learning.networks.DuelingDQN import Network

    network = Network(observation_size, action_num, config=config)

    device = hlp.get_device()
    agent = DuelingDQN(network=network, config=config, device=device)
    return agent


def create_DoubleDQN(observation_size, action_num, config: acf.DoubleDQNConfig):
    from cares_reinforcement_learning.algorithm.value import DoubleDQN
    from cares_reinforcement_learning.networks.DoubleDQN import Network

    network = Network(observation_size, action_num, config=config)

    device = hlp.get_device()
    agent = DoubleDQN(
        network=network,
        config=config,
        device=device,
    )
    return agent


def create_NoisyNet(observation_size, action_num, config: acf.NoisyNetConfig):
    from cares_reinforcement_learning.algorithm.value import NoisyNet
    from cares_reinforcement_learning.networks.NoisyNet import Network

    network = Network(observation_size, action_num, config)

    device = hlp.get_device()
    agent = NoisyNet(network=network, config=config, device=device)
    return agent


def create_C51(observation_size, action_num, config: acf.C51Config):
    from cares_reinforcement_learning.algorithm.value import C51
    from cares_reinforcement_learning.networks.C51 import Network

    network = Network(observation_size, action_num, config=config)

    device = hlp.get_device()
    agent = C51(network=network, config=config, device=device)
    return agent


def create_QRDQN(observation_size, action_num, config: acf.QRDQNConfig):
    from cares_reinforcement_learning.algorithm.value import QRDQN
    from cares_reinforcement_learning.networks.QRDQN import Network

    network = Network(observation_size, action_num, config=config)

    device = hlp.get_device()
    agent = QRDQN(network=network, config=config, device=device)
    return agent


def create_Rainbow(observation_size, action_num, config: acf.RainbowConfig):
    from cares_reinforcement_learning.algorithm.value import Rainbow
    from cares_reinforcement_learning.networks.Rainbow import Network

    network = Network(observation_size, action_num, config=config)

    device = hlp.get_device()
    agent = Rainbow(network=network, config=config, device=device)
    return agent


###################################
#         PPO Algorithms          #
###################################


def create_PPO(observation_size, action_num, config: acf.PPOConfig):
    from cares_reinforcement_learning.algorithm.policy import PPO
    from cares_reinforcement_learning.networks.PPO import Actor, Critic

    actor = Actor(observation_size, action_num, config=config)
    critic = Critic(observation_size, config=config)

    device = hlp.get_device()
    agent = PPO(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


###################################
#         SAC Algorithms          #
###################################


def create_SAC(observation_size, action_num, config: acf.SACConfig):
    from cares_reinforcement_learning.algorithm.policy import SAC
    from cares_reinforcement_learning.networks.SAC import Actor, Critic

    actor = Actor(observation_size, action_num, config=config)
    critic = Critic(observation_size, action_num, config=config)

    device = hlp.get_device()
    agent = SAC(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_SACAE(observation_size, action_num, config: acf.SACAEConfig):
    from cares_reinforcement_learning.algorithm.policy import SACAE
    from cares_reinforcement_learning.encoders.vanilla_autoencoder import Decoder
    from cares_reinforcement_learning.networks.SACAE import Actor, Critic

    actor = Actor(observation_size, action_num, config=config)
    critic = Critic(observation_size, action_num, config=config)

    ae_config = config.autoencoder_config
    decoder = Decoder(
        observation_size["image"],
        out_dim=actor.encoder.out_dim,
        latent_dim=ae_config.latent_dim,
        num_layers=ae_config.num_layers,
        num_filters=ae_config.num_filters,
        kernel_size=ae_config.kernel_size,
    )

    device = hlp.get_device()
    agent = SACAE(
        actor_network=actor,
        critic_network=critic,
        decoder_network=decoder,
        config=config,
        device=device,
    )
    return agent


def create_PERSAC(observation_size, action_num, config: acf.PERSACConfig):
    from cares_reinforcement_learning.algorithm.policy import PERSAC
    from cares_reinforcement_learning.networks.PERSAC import Actor, Critic

    actor = Actor(observation_size, action_num, config=config)
    critic = Critic(observation_size, action_num, config=config)

    device = hlp.get_device()
    agent = PERSAC(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_REDQ(observation_size, action_num, config: acf.REDQConfig):
    from cares_reinforcement_learning.algorithm.policy import REDQ
    from cares_reinforcement_learning.networks.REDQ import Actor, Critic

    actor = Actor(observation_size, action_num, config=config)
    ensemble_critic = Critic(observation_size, action_num, config=config)

    device = hlp.get_device()
    agent = REDQ(
        actor_network=actor,
        ensemble_critic=ensemble_critic,
        config=config,
        device=device,
    )
    return agent


def create_TQC(observation_size, action_num, config: acf.TQCConfig):
    from cares_reinforcement_learning.algorithm.policy import TQC
    from cares_reinforcement_learning.networks.TQC import Actor, Critic

    actor = Actor(observation_size, action_num, config=config)
    critic = Critic(observation_size, action_num, config=config)

    device = hlp.get_device()
    agent = TQC(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_LAPSAC(observation_size, action_num, config: acf.LAPSACConfig):
    from cares_reinforcement_learning.algorithm.policy import LAPSAC
    from cares_reinforcement_learning.networks.LAPSAC import Actor, Critic

    actor = Actor(observation_size, action_num, config=config)
    critic = Critic(observation_size, action_num, config=config)

    device = hlp.get_device()
    agent = LAPSAC(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_LA3PSAC(observation_size, action_num, config: acf.LA3PSACConfig):
    from cares_reinforcement_learning.algorithm.policy import LA3PSAC
    from cares_reinforcement_learning.networks.LA3PSAC import Actor, Critic

    actor = Actor(observation_size, action_num, config=config)
    critic = Critic(observation_size, action_num, config=config)

    device = hlp.get_device()
    agent = LA3PSAC(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_MAPERSAC(observation_size, action_num, config: acf.MAPERSACConfig):
    from cares_reinforcement_learning.algorithm.policy import MAPERSAC
    from cares_reinforcement_learning.networks.MAPERSAC import Actor, Critic

    actor = Actor(observation_size, action_num, config=config)
    critic = Critic(observation_size, action_num, config=config)

    device = hlp.get_device()
    agent = MAPERSAC(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_RDSAC(observation_size, action_num, config: acf.RDSACConfig):
    from cares_reinforcement_learning.algorithm.policy import RDSAC
    from cares_reinforcement_learning.networks.RDSAC import Actor, Critic

    actor = Actor(observation_size, action_num, config=config)
    critic = Critic(observation_size, action_num, config=config)

    device = hlp.get_device()
    agent = RDSAC(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_DroQ(observation_size, action_num, config: acf.DroQConfig):
    from cares_reinforcement_learning.algorithm.policy import DroQ
    from cares_reinforcement_learning.networks.DroQ import Actor, Critic

    actor = Actor(observation_size, action_num, config=config)
    critic = Critic(observation_size, action_num, config=config)

    device = hlp.get_device()
    agent = DroQ(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_CrossQ(observation_size, action_num, config: acf.CrossQConfig):
    from cares_reinforcement_learning.algorithm.policy import CrossQ
    from cares_reinforcement_learning.networks.CrossQ import Actor, Critic

    actor = Actor(observation_size, action_num, config=config)
    critic = Critic(observation_size, action_num, config=config)

    device = hlp.get_device()
    agent = CrossQ(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_SDAR(observation_size, action_num, config: acf.SDARConfig):
    from cares_reinforcement_learning.algorithm.policy import SDAR
    from cares_reinforcement_learning.networks.SDAR import Actor, Critic

    actor = Actor(observation_size, action_num, config=config)
    critic = Critic(observation_size, action_num, config=config)

    device = hlp.get_device()
    agent = SDAR(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_DynaSAC(observation_size, action_num, config: acf.DynaSACConfig):
    """
    Create networks for model-based SAC agent. The Actor and Critic is same.
    An extra world model is added.
    """
    from cares_reinforcement_learning.algorithm.mbrl import DynaSAC
    from cares_reinforcement_learning.networks.DynaSAC import Actor, Critic
    from cares_reinforcement_learning.networks.world_models import EnsembleWorldReward

    actor = Actor(observation_size, action_num, config=config)
    critic = Critic(observation_size, action_num, config=config)

    device = hlp.get_device()

    world_model = EnsembleWorldReward(
        observation_size=observation_size,
        num_actions=action_num,
        num_models=config.num_models,
        lr=config.world_model_lr,
        device=device,
    )

    agent = DynaSAC(
        actor_network=actor,
        critic_network=critic,
        world_network=world_model,
        config=config,
        device=device,
    )
    return agent


def create_SACD(observation_size, action_num, config: acf.SACDConfig):
    from cares_reinforcement_learning.algorithm.policy import SACD
    from cares_reinforcement_learning.networks.SACD import Actor, Critic

    actor = Actor(observation_size, action_num, config=config)
    critic = Critic(observation_size, action_num, config=config)

    device = hlp.get_device()
    agent = SACD(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_DIAYN(observation_size, action_num, config: acf.DIAYNConfig):
    from cares_reinforcement_learning.algorithm.usd import DIAYN
    from cares_reinforcement_learning.networks.DIAYN import Discriminator

    agent = create_SAC(observation_size + config.num_skills, action_num, config=config)

    discriminator = Discriminator(
        observation_size, num_skills=config.num_skills, config=config
    )

    device = hlp.get_device()
    agent = DIAYN(
        skills_agent=agent,
        discriminator_network=discriminator,
        config=config,
        device=device,
    )
    return agent


def create_DADS(observation_size, action_num, config: acf.DADSConfig):
    from cares_reinforcement_learning.algorithm.usd import DADS
    from cares_reinforcement_learning.networks.DADS import SkillDynamicsModel

    agent = create_SAC(observation_size + config.num_skills, action_num, config=config)

    discriminator = SkillDynamicsModel(
        observation_size=observation_size, num_skills=config.num_skills, config=config
    )

    device = hlp.get_device()
    agent = DADS(
        skills_agent=agent,
        discriminator_network=discriminator,
        config=config,
        device=device,
    )
    return agent


###################################
#         TD3 Algorithms          #
###################################


def create_DDPG(observation_size, action_num, config: acf.DDPGConfig):
    from cares_reinforcement_learning.algorithm.policy import DDPG
    from cares_reinforcement_learning.networks.DDPG import Actor, Critic

    actor = Actor(observation_size, action_num, config=config)
    critic = Critic(observation_size, action_num, config=config)

    device = hlp.get_device()
    agent = DDPG(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_TD3(observation_size, action_num, config: acf.TD3Config):
    from cares_reinforcement_learning.algorithm.policy import TD3
    from cares_reinforcement_learning.networks.TD3 import Actor, Critic

    actor = Actor(observation_size, action_num, config=config)
    critic = Critic(observation_size, action_num, config=config)

    device = hlp.get_device()
    agent = TD3(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_TD3AE(observation_size, action_num, config: acf.TD3AEConfig):
    from cares_reinforcement_learning.algorithm.policy import TD3AE
    from cares_reinforcement_learning.encoders.vanilla_autoencoder import Decoder
    from cares_reinforcement_learning.networks.TD3AE import Actor, Critic

    actor = Actor(observation_size, action_num, config=config)
    critic = Critic(observation_size, action_num, config=config)

    ae_config = config.autoencoder_config
    decoder = Decoder(
        observation_size["image"],
        out_dim=actor.encoder.out_dim,
        latent_dim=ae_config.latent_dim,
        num_layers=ae_config.num_layers,
        num_filters=ae_config.num_filters,
        kernel_size=ae_config.kernel_size,
    )

    device = hlp.get_device()
    agent = TD3AE(
        actor_network=actor,
        critic_network=critic,
        decoder_network=decoder,
        config=config,
        device=device,
    )
    return agent


def create_NaSATD3(observation_size, action_num, config: acf.NaSATD3Config):
    from cares_reinforcement_learning.algorithm.policy import NaSATD3
    from cares_reinforcement_learning.networks.NaSATD3 import Actor, Critic

    actor = Actor(observation_size, action_num, config=config)
    critic = Critic(observation_size, action_num, config=config)

    device = hlp.get_device()
    agent = NaSATD3(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_PERTD3(observation_size, action_num, config: acf.PERTD3Config):
    from cares_reinforcement_learning.algorithm.policy import PERTD3
    from cares_reinforcement_learning.networks.PERTD3 import Actor, Critic

    actor = Actor(observation_size, action_num, config=config)
    critic = Critic(observation_size, action_num, config=config)

    device = hlp.get_device()
    agent = PERTD3(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_LAPTD3(observation_size, action_num, config: acf.LAPTD3Config):
    from cares_reinforcement_learning.algorithm.policy import LAPTD3
    from cares_reinforcement_learning.networks.LAPTD3 import Actor, Critic

    actor = Actor(observation_size, action_num, config=config)
    critic = Critic(observation_size, action_num, config=config)

    device = hlp.get_device()
    agent = LAPTD3(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_PALTD3(observation_size, action_num, config: acf.PALTD3Config):
    from cares_reinforcement_learning.algorithm.policy import PALTD3
    from cares_reinforcement_learning.networks.PALTD3 import Actor, Critic

    actor = Actor(observation_size, action_num, config=config)
    critic = Critic(observation_size, action_num, config=config)

    device = hlp.get_device()
    agent = PALTD3(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_LA3PTD3(observation_size, action_num, config: acf.LA3PTD3Config):
    from cares_reinforcement_learning.algorithm.policy import LA3PTD3
    from cares_reinforcement_learning.networks.LA3PTD3 import Actor, Critic

    actor = Actor(observation_size, action_num, config=config)
    critic = Critic(observation_size, action_num, config=config)

    device = hlp.get_device()
    agent = LA3PTD3(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_MAPERTD3(observation_size, action_num, config: acf.MAPERTD3Config):
    from cares_reinforcement_learning.algorithm.policy import MAPERTD3
    from cares_reinforcement_learning.networks.MAPERTD3 import Actor, Critic

    actor = Actor(observation_size, action_num, config=config)
    critic = Critic(observation_size, action_num, config=config)

    device = hlp.get_device()
    agent = MAPERTD3(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_RDTD3(observation_size, action_num, config: acf.RDTD3Config):
    from cares_reinforcement_learning.algorithm.policy import RDTD3
    from cares_reinforcement_learning.networks.RDTD3 import Actor, Critic

    actor = Actor(observation_size, action_num, config=config)
    critic = Critic(observation_size, action_num, config=config)

    device = hlp.get_device()
    agent = RDTD3(
        actor_network=actor,
        critic_network=critic,
        config=config,
        device=device,
    )
    return agent


def create_CTD4(observation_size, action_num, config: acf.CTD4Config):
    from cares_reinforcement_learning.algorithm.policy import CTD4
    from cares_reinforcement_learning.networks.CTD4 import Actor, Critic

    device = hlp.get_device()

    actor = Actor(observation_size, action_num, config=config)
    ensemble_critic = Critic(observation_size, action_num, config=config)

    agent = CTD4(
        actor_network=actor,
        ensemble_critic=ensemble_critic,
        config=config,
        device=device,
    )

    return agent


def create_TD7(observation_size, action_num, config: acf.TD7Config):
    from cares_reinforcement_learning.algorithm.policy import TD7
    from cares_reinforcement_learning.networks.TD7 import Actor, Critic, Encoder

    device = hlp.get_device()

    actor = Actor(observation_size, action_num, config=config)
    critic = Critic(observation_size, action_num, config=config)
    encoder = Encoder(observation_size, action_num, config=config)

    agent = TD7(
        actor_network=actor,
        critic_network=critic,
        encoder_network=encoder,
        config=config,
        device=device,
    )

    return agent


def _compare_mlp_parts(obj1: acf.AlgorithmConfig, obj2: acf.AlgorithmConfig) -> bool:
    # Extract fields where the value is of type mlp_type
    def get_mlp_fields(obj):
        return {
            name: value.dict()
            for name, value in obj.__dict__.items()
            if isinstance(value, acf.MLPConfig)
        }

    mlp_fields1 = get_mlp_fields(obj1)
    mlp_fields2 = get_mlp_fields(obj2)

    return mlp_fields1 == mlp_fields2


class NetworkFactory:
    def create_network(
        self,
        observation_size,
        action_num: int,
        config: acf.AlgorithmConfig,
    ):
        algorithm = config.algorithm

        agent = None
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isfunction(obj):
                if name == f"create_{algorithm}":
                    agent = obj(observation_size, action_num, config)

        if agent is None:
            logging.warning(f"Unknown {agent} algorithm.")
        else:
            if config.model_path is not None:
                logging.info(f"Loading model weights from {config.model_path}")
                agent.load_models(filepath=config.model_path, filename=config.algorithm)

            if not _compare_mlp_parts(
                type(config)(algorithm=config.algorithm, gamma=config.gamma), config
            ):
                logging.warning(
                    "The network architecture has changed from the default configuration."
                )

        return agent
