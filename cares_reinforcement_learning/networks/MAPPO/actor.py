from cares_reinforcement_learning.networks.PPO import Actor as PPOActor
from cares_reinforcement_learning.networks.PPO import DefaultActor as PPODefaultActor
from cares_reinforcement_learning.util.configurations import MAPPOConfig


class DefaultActor(PPODefaultActor):
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


class Actor(PPOActor):
    def __init__(
        self,
        observation_size: dict,
        num_actions: int,
        config: MAPPOConfig,
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
