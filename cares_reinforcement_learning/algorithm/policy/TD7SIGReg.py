"""
TD7SIGReg (TD7 + SIGReg representation regularisation)
-------------------------------------------------------

TD7 Paper: https://arxiv.org/pdf/2306.02451
SIGReg (LeJEPA) Paper: https://arxiv.org/abs/2511.08544

TD7SIGReg is identical to TD7 except for how the SALE state
embedding z_s is kept well-conditioned:

TD7:
    z_s = AvgL1Norm(f(s))
    L_enc = || g(z_s, a) - stopgrad(z_s') ||^2

TD7SIGReg:
    z_s = f(s)                          (no AvgL1Norm)
    L_pred = || g(z_s, a) - stopgrad(z_s') ||^2
    L_enc  = (1 - λ) * L_pred + λ * SIGReg(z_s)

SIGReg (Sketched Isotropic Gaussian Regularisation) pushes the
batch distribution of z_s towards an isotropic Gaussian by matching
the empirical characteristic function of random 1D projections to
that of N(0, 1). This replaces the hard scale normalisation with a
soft distributional constraint and prevents representation collapse.

λ = sigreg_lambda (default 0.05, as in LeJEPA).

The actor/critic feature-layer AvgL1Norm is unchanged from TD7.
"""

from typing import Any

import torch
import torch.nn.functional as F

from cares_reinforcement_learning.algorithm.configurations import TD7SIGRegConfig
from cares_reinforcement_learning.algorithm.policy.TD7 import TD7
from cares_reinforcement_learning.algorithm.sigreg import SIGReg
from cares_reinforcement_learning.networks.TD7SIGReg import Actor, Critic, Encoder


class TD7SIGReg(TD7):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        encoder_network: Encoder,
        config: TD7SIGRegConfig,
        device: torch.device,
    ):
        super().__init__(
            actor_network=actor_network,
            critic_network=critic_network,
            encoder_network=encoder_network,
            config=config,
            device=device,
        )

        self.sigreg_lambda = config.sigreg_lambda
        self.sigreg = SIGReg(
            knots=config.sigreg_knots,
            t_max=config.sigreg_t_max,
            num_slices=config.sigreg_num_slices,
        ).to(device)

    def _update_encoder(
        self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor
    ) -> dict[str, Any]:
        info: dict[str, Any] = {}

        with torch.no_grad():
            next_zs = self.encoder_net.zs(next_states)

        zs = self.encoder_net.zs(states)
        pred_zs = self.encoder_net.zsa(zs, actions)

        pred_loss = F.mse_loss(pred_zs, next_zs)
        sigreg_loss = self.sigreg(zs)

        encoder_loss = (
            1.0 - self.sigreg_lambda
        ) * pred_loss + self.sigreg_lambda * sigreg_loss

        self.encoder_net_optimiser.zero_grad()
        encoder_loss.backward()
        self.encoder_net_optimiser.step()

        with torch.no_grad():
            # --- SALE / representation health ---
            info["encoder_loss"] = encoder_loss.item()
            info["sale_pred_loss"] = pred_loss.item()
            info["sigreg_loss"] = sigreg_loss.item()
            info["sigreg_lambda"] = float(self.sigreg_lambda)

            info["zs_abs_mean"] = zs.abs().mean().item()
            info["zs_std"] = zs.std(unbiased=False).item()
            info["pred_zs_abs_mean"] = pred_zs.abs().mean().item()
            info["pred_zs_std"] = pred_zs.std(unbiased=False).item()

            # Cosine similarity: are we predicting the right *direction* in latent space?
            eps = 1e-12
            cos = (pred_zs * next_zs).sum(dim=1) / (
                pred_zs.norm(dim=1) * next_zs.norm(dim=1) + eps
            )
            info["sale_cos_mean"] = cos.mean().item()
            info["sale_cos_p05"] = cos.quantile(0.05).item()

            # Tail risk: max per-sample latent prediction error
            per_sample_mse = (pred_zs - next_zs).pow(2).mean(dim=1)
            info["sale_mse_mean"] = per_sample_mse.mean().item()
            info["sale_mse_p95"] = per_sample_mse.quantile(0.95).item()

        return info
