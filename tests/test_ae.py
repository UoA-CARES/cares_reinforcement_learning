import inspect

import pytest
import torch

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.encoders import configurations
from cares_reinforcement_learning.encoders.autoencoder_factory import AEFactory
from cares_reinforcement_learning.encoders.configurations import AEConfig, BurgessConfig


def test_ae():
    device = hlp.get_device()

    observation_size = (3, 32, 32)

    test_image = torch.rand(2, *observation_size)
    test_image = test_image.to(device)

    factory = AEFactory()

    ae_configurations = {}
    for name, cls in inspect.getmembers(configurations, inspect.isclass):
        if issubclass(cls, AEConfig) and cls != AEConfig and cls != BurgessConfig:
            name = name.replace("Config", "")
            ae_configurations[name] = cls

    for ae, config in ae_configurations.items():

        config = config(latent_dim=100)

        try:
            autoencoder = factory.create_autoencoder(
                observation_size=observation_size, config=config
            )
        except Exception as e:
            pytest.fail(f"Exception making autoencoder: {ae} {e}")

        autoencoder = autoencoder.to(device)

        out = autoencoder(test_image)
        assert (
            out["reconstructed_observation"].shape
            == test_image.shape
            == (2, *observation_size)
        )

        loss = autoencoder.update_autoencoder(test_image)
        assert loss is not None
