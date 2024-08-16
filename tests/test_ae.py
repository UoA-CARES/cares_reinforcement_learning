import inspect
import os

import pytest
import torch

from cares_reinforcement_learning.encoders import configurations
from cares_reinforcement_learning.encoders.autoencoder_factory import AEFactory
from cares_reinforcement_learning.encoders.configurations import AEConfig, BurgessConfig

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(not IN_GITHUB_ACTIONS, reason="Running more complex test locally")
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


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_ae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    observation_size = (3, 32, 32)

    test_image = torch.randn(2, *observation_size)
    test_image = test_image.to(device)

    factory = AEFactory()

    ae_configurations = {}
    for name, cls in inspect.getmembers(configurations, inspect.isclass):
        if issubclass(cls, AEConfig) and cls != AEConfig and cls != BurgessConfig:
            name = name.replace("Config", "")
            ae_configurations[name] = cls

    for ae, config in ae_configurations.items():

        config = config(latent_dim=100)

        autoencoder = factory.create_autoencoder(
            observation_size=observation_size, config=config
        )
        assert autoencoder is not None, f"{ae} was not created successfully"

        autoencoder = autoencoder.to(device)

        out = autoencoder(test_image)
        assert (
            out["reconstructed_observation"].shape
            == test_image.shape
            == (2, *observation_size)
        )

        loss = autoencoder.update_autoencoder(test_image)
        assert loss is not None
