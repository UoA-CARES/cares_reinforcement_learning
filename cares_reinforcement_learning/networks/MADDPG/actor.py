from cares_reinforcement_learning.networks.DDPG import Actor as DDPGActor
from cares_reinforcement_learning.networks.DDPG import DefaultActor as DDPGDefaultActor
from cares_reinforcement_learning.util.configurations import MADDPGConfig


class DefaultActor(DDPGDefaultActor):
    # pylint: disable=super-init-not-called
    def __init__(
        self, observation_size: dict, num_actions: int, agent_id: str | None = None
    ):
        obs_dict = observation_size["obs"]
        key = agent_id if agent_id is not None else next(iter(obs_dict))
        input_size = obs_dict[key]

        super().__init__(
            observation_size=input_size,
            num_actions=num_actions,
            hidden_sizes=[256, 256],
        )


class Actor(DDPGActor):
    def __init__(
        self,
        observation_size: dict,
        num_actions: int,
        config: MADDPGConfig,
        agent_id: str | None = None,
    ):

        obs_dict = observation_size["obs"]
        key = agent_id if agent_id is not None else next(iter(obs_dict))
        input_size = obs_dict[key]

        super().__init__(
            observation_size=input_size,
            num_actions=num_actions,
            config=config,
        )
