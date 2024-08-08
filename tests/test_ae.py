import inspect

import pytest

from cares_reinforcement_learning.encoders import configurations
from cares_reinforcement_learning.encoders.autoencoder_factory import AEFactory
from cares_reinforcement_learning.encoders.configurations import AEConfig, BurgessConfig


def test_ae_factory():
    factory = AEFactory()

    ae_configurations = {}
    for name, cls in inspect.getmembers(configurations, inspect.isclass):
        if issubclass(cls, AEConfig) and cls != AEConfig and cls != BurgessConfig:
            name = name.replace("Config", "")
            ae_configurations[name] = cls

    for ae, config in ae_configurations.items():

        observation_size = (9, 84, 84)

        config = config(latent_dim=100)

        ae = factory.create_autoencoder(
            observation_size=observation_size, config=config
        )
        assert ae is not None, f"{ae} was not created successfully"
