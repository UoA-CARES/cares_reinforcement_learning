from cares_reinforcement_learning.networks.SAC import Actor as SACActor
from cares_reinforcement_learning.networks.SAC import DefaultActor as SACDefaultActor
from cares_reinforcement_learning.util.configurations import MASACConfig


class DefaultActor(SACDefaultActor):
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


class Actor(SACActor):
    def __init__(
        self,
        observation_size: dict,
        num_actions: int,
        config: MASACConfig,
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
