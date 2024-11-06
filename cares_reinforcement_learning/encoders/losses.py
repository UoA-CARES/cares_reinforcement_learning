import abc
import math
from typing import Any

import torch
from torch import optim
from torch.nn import functional as F

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.encoders.configurations import BurgessConfig
from cares_reinforcement_learning.encoders.constants import Losses, ReconDist
from cares_reinforcement_learning.encoders.discriminator import Discriminator


class AELoss:
    """
    Autoencoder loss function.

    This class represents the loss function used for training an autoencoder model.
    It calculates the reconstruction loss and adds an L2 penalty on the latent representation.

    Args:
        latent_lambda (float): The weight for the L2 penalty on the latent representation.

    """

    def __init__(self, latent_lambda: float = 1e-6) -> None:
        self.latent_lambda = latent_lambda

    def calculate_loss(
        self,
        data: torch.Tensor,
        reconstructed_data: torch.Tensor,
        latent_sample: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the loss for the autoencoder model.

        Args:
            data: The input data.
            reconstructed_data: The reconstructed data.
            latent_sample: The latent representation of the input data.

        Returns:
            The calculated loss.

        """
        rec_loss = _reconstruction_loss(
            data, reconstructed_data, distribution=ReconDist.GAUSSIAN
        )

        # add L2 penalty on latent representation
        latent_loss = (0.5 * latent_sample.pow(2).sum(1)).mean()

        loss = rec_loss + self.latent_lambda * latent_loss

        return loss

    def update_autoencoder(self, data: torch.Tensor, autoencoder) -> torch.Tensor:
        latent_observation = autoencoder.encoder(data)

        reconstructed_observation = autoencoder.decoder(latent_observation)

        loss = self.calculate_loss(
            data=data,
            reconstructed_data=reconstructed_observation,
            latent_sample=reconstructed_observation,
        )

        autoencoder.encoder_optimizer.zero_grad()
        autoencoder.decoder_optimizer.zero_grad()
        loss.backward()
        autoencoder.encoder_optimizer.step()
        autoencoder.decoder_optimizer.step()

        return loss


def get_burgess_loss_function(config: BurgessConfig):
    loss_name = config.loss_function_type

    if loss_name == Losses.VAE:
        return VAELoss(rec_dist=config.rec_dist, steps_anneal=config.steps_anneal)

    if loss_name == Losses.BETA_H:
        return BetaHLoss(
            rec_dist=config.rec_dist, steps_anneal=config.steps_anneal, beta=config.beta
        )

    if loss_name == Losses.BETA_B:
        return BetaBLoss(
            rec_dist=config.rec_dist,
            steps_anneal=config.steps_anneal,
            c_init=config.c_init,
            c_fin=config.c_fin,
            gamma=config.gamma,
        )
    if loss_name == Losses.FACTOR:
        device = hlp.get_device()

        disc_kwargs = {} if config.disc_kwargs is None else config.disc_kwargs
        disc_kwargs["latent_dim"] = config.latent_dim

        return FactorKLoss(
            device=device,
            rec_dist=config.rec_dist,
            steps_anneal=config.steps_anneal,
            gamma=config.gamma,
            disc_kwargs=disc_kwargs,
            optim_kwargs=config.optim_kwargs,
        )

    if loss_name == Losses.BTCVAE:
        # TODO n_data should be passed as argument
        return BtcvaeLoss(
            n_data=1000,
            rec_dist=config.rec_dist,
            steps_anneal=config.steps_anneal,
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
            is_mss=config.is_mss,
        )

    raise ValueError(f"Unknown loss function: {loss_name}")


class BaseBurgessLoss(metaclass=abc.ABCMeta):
    """
    Base class for losses.

    Parameters
    ----------
    rec_dist: {"bernoulli", "gaussian", "laplace"}, optional
        Reconstruction distribution of the likelihood on each pixel.
        Implicitly defines the reconstruction loss. Bernoulli corresponds to a
        binary cross entropy (bse), Gaussian corresponds to MSE, Laplace
        corresponds to L1.

    steps_anneal: nool, optional
        Number of annealing steps where gradually adding the regularisation.
    """

    def __init__(self, rec_dist=ReconDist.BERNOULLI, steps_anneal: int = 0):
        self.n_train_steps = 0
        self.rec_dist = rec_dist
        self.steps_anneal = steps_anneal

    @abc.abstractmethod
    def calculate_loss(
        self,
        data,
        reconstructed_data,
        latent_dist,
        is_train,
    ):
        """
        Calculates loss for a batch of data.

        Parameters
        ----------
        data : torch.Tensor
            Input data (e.g. batch of images). Shape : (batch_size, n_chan,
            height, width).

        reconstructed_data : torch.Tensor
            Reconstructed data. Shape : (batch_size, n_chan, height, width).

        latent_dist : tuple of torch.tensor
            sufficient statistics of the latent dimension. E.g. for gaussian
            (mean, log_var) each of shape : (batch_size, latent_dim).

        is_train : bool
            Whether currently in train mode.

        kwargs:
            Loss specific arguments
        """
        raise NotImplementedError("Method must be implemented in subclass")

    def _pre_call(self):
        self.n_train_steps += 1

    def update_autoencoder(self, data, autoencoder):
        self._pre_call()

        latent_dist = autoencoder.encoder(data)

        _, _, latent_sample = latent_dist

        reconstructed_observation = autoencoder.decoder(latent_sample)

        loss = self.calculate_loss(
            data=data,
            reconstructed_data=reconstructed_observation,
            latent_dist=latent_dist,
            is_train=True,
        )

        autoencoder.encoder_optimizer.zero_grad()
        autoencoder.decoder_optimizer.zero_grad()
        loss.backward()
        autoencoder.encoder_optimizer.step()
        autoencoder.decoder_optimizer.step()

        return loss


class BetaHLoss(BaseBurgessLoss):
    """
    Compute the Beta-VAE loss as in [1]

    Parameters
    ----------
    beta : float, optional
        Weight of the kl divergence.

    References
    ----------
        [1] Higgins, Irina, et al. "beta-vae: Learning basic visual concepts with
        a constrained variational framework." (2016).
    """

    def __init__(
        self,
        rec_dist: ReconDist = ReconDist.BERNOULLI,
        steps_anneal: int = 0,
        beta: int = 4,
    ):
        super().__init__(rec_dist=rec_dist, steps_anneal=steps_anneal)
        self.beta = beta

    def calculate_loss(
        self,
        data,
        reconstructed_data,
        latent_dist,
        is_train,
    ):
        rec_loss = _reconstruction_loss(
            data, reconstructed_data, reduction="sum", distribution=self.rec_dist
        )

        mean, logvar, _ = latent_dist
        kl_loss = _kl_normal_loss(mean, logvar)

        anneal_reg = (
            _linear_annealing(0, 1, self.n_train_steps, self.steps_anneal)
            if is_train
            else 1
        )
        loss = rec_loss + anneal_reg * (self.beta * kl_loss)

        return loss


# The vanilla vae is just betaH with beta = 1 so there's no extra function in the code
class VAELoss(BetaHLoss):
    """
    A class representing the vanilla loss for the variational encoder network.

    Parameters:
    - rec_dist (ReconDist): The reconstruction distribution to use. Default is ReconDist.BERNOULLI.
    - steps_anneal (int): The number of steps to anneal the loss. Default is 0.
    """

    def __init__(
        self,
        rec_dist: ReconDist = ReconDist.BERNOULLI,
        steps_anneal: int = 0,
    ):
        super().__init__(rec_dist=rec_dist, steps_anneal=steps_anneal, beta=1)


class FactorKLoss(BaseBurgessLoss):
    """
    Compute the Factor-VAE loss as per Algorithm 2 of [1]

    Parameters
    ----------
    device : torch.device

    gamma : float, optional
        Weight of the TC loss term. `gamma` in the paper.

    discriminator : disvae.discriminator.Discriminator

    optimizer_d : torch.optim

    disc_kwargs:
        Additional arguments for the discriminator - e.g. latent_dim.

    optim_kwargs:
        Additional arguments for the discriminator optimizer - e.g. lr or betas.

    References
    ----------
        [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
        arXiv preprint arXiv:1802.05983 (2018).
    """

    def __init__(
        self,
        device: torch.device,
        rec_dist: ReconDist = ReconDist.BERNOULLI,
        steps_anneal: int = 0,
        gamma: float = 10.0,
        disc_kwargs: dict[str, Any] | None = None,
        optim_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(rec_dist=rec_dist, steps_anneal=steps_anneal)

        if disc_kwargs is None:
            disc_kwargs = {}
        if optim_kwargs is None:
            optim_kwargs = {"lr": 5e-5, "betas": (0.5, 0.9)}

        self.gamma = gamma
        self.device = device

        self.discriminator = Discriminator(**disc_kwargs)
        self.discriminator.to(self.device)

        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(), **optim_kwargs
        )

    def _calculate_loss(self, data, reconstructed_data, latent_dist, is_train):
        rec_loss = _reconstruction_loss(
            data, reconstructed_data, reduction="sum", distribution=self.rec_dist
        )

        mean, logvar, latent_sample = latent_dist
        kl_loss = _kl_normal_loss(mean, logvar)

        d_z = self.discriminator(latent_sample)
        tc_loss = (d_z[:, 0] - d_z[:, 1]).mean()

        anneal_reg = (
            _linear_annealing(0, 1, self.n_train_steps, self.steps_anneal)
            if is_train
            else 1
        )
        vae_loss = rec_loss + (kl_loss + anneal_reg * self.gamma * tc_loss)

        return d_z, vae_loss

    def update_autoencoder(self, data, autoencoder):
        self._pre_call()

        # factor-vae split data into two batches. In the paper they sample 2 batches
        batch_size = data.size(dim=0)
        half_batch_size = batch_size // 2
        data = data.split(half_batch_size)
        data_one = data[0]
        data_two = data[1]

        latent_dist_one = autoencoder.encoder(data_one)

        _, _, latent_sample_one = latent_dist_one

        recon_batch_one = autoencoder.decoder(latent_sample_one)

        d_z, vae_loss = self._calculate_loss(
            data_one,
            recon_batch_one,
            latent_dist_one,
            autoencoder.training,
        )

        # Compute VAE gradients
        autoencoder.encoder_optimizer.zero_grad()
        autoencoder.decoder_optimizer.zero_grad()
        vae_loss.backward(retain_graph=True)

        # Discriminator Loss
        # Get second sample of latent distribution
        # works because only the BurgessEncoder can be used here
        latent_dist_two = autoencoder.encoder(data_two)

        _, _, latent_sample_two = latent_dist_two

        z_perm = _permute_dims(latent_sample_two).detach()

        d_z_perm = self.discriminator(z_perm)

        # Calculate total correlation loss
        # for cross entropy the target is the index => need to be long and says
        # that it's first output for d_z and second for perm
        ones = torch.ones(half_batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros_like(ones)
        d_tc_loss = 0.5 * (
            F.cross_entropy(d_z, zeros) + F.cross_entropy(d_z_perm, ones)
        )

        # with sigmoid would be :
        # d_tc_loss = 0.5 * (self.bce(d_z.flatten(), ones) + self.bce(d_z_perm.flatten(), 1 - ones))

        # d_tc_loss = anneal_reg * d_tc_loss

        # Compute discriminator gradients
        self.discriminator_optimizer.zero_grad()
        d_tc_loss.backward()

        # Update at the end (since pytorch 1.5. complains if update before)
        autoencoder.encoder_optimizer.step()
        autoencoder.decoder_optimizer.step()
        self.discriminator_optimizer.step()

        return vae_loss

    def calculate_loss(
        self,
        data,
        reconstructed_data,
        latent_dist,
        is_train,
    ):
        _, vae_loss = self._calculate_loss(
            data, reconstructed_data, latent_dist, is_train
        )

        return vae_loss


class BetaBLoss(BaseBurgessLoss):
    """
    Compute the Beta-VAE loss as in [1]

    Parameters
    ----------
    C_init : float, optional
        Starting annealed capacity C.

    C_fin : float, optional
        Final annealed capacity C.

    gamma : float, optional
        Weight of the KL divergence term.

    References
    ----------
        [1] Burgess, Christopher P., et al. "Understanding disentangling in
        $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
    """

    def __init__(
        self,
        rec_dist: ReconDist = ReconDist.BERNOULLI,
        steps_anneal: int = 0,
        c_init: float = 0.0,
        c_fin: float = 20.0,
        gamma: float = 100.0,
    ):
        super().__init__(rec_dist=rec_dist, steps_anneal=steps_anneal)
        self.gamma = gamma
        self.c_init = c_init
        self.c_fin = c_fin

    def calculate_loss(
        self,
        data,
        reconstructed_data,
        latent_dist,
        is_train,
    ):
        rec_loss = _reconstruction_loss(
            data, reconstructed_data, reduction="sum", distribution=self.rec_dist
        )

        mean, logvar, _ = latent_dist
        kl_loss = _kl_normal_loss(mean, logvar)

        capacity = (
            _linear_annealing(
                self.c_init, self.c_fin, self.n_train_steps, self.steps_anneal
            )
            if is_train
            else self.c_fin
        )

        loss = rec_loss + self.gamma * (kl_loss - capacity).abs()

        return loss


class BtcvaeLoss(BaseBurgessLoss):
    """
    Compute the decomposed KL loss with either minibatch weighted sampling or
    minibatch stratified sampling according to [1]

    Parameters
    ----------
    n_data: int
        Number of data in the training set

    alpha : float
        Weight of the mutual information term.

    beta : float
        Weight of the total correlation term.

    gamma : float
        Weight of the dimension-wise KL term.

    is_mss : bool
        Whether to use minibatch stratified sampling instead of minibatch
        weighted sampling.

    References
    ----------
       [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
       autoencoders." Advances in Neural Information Processing Systems. 2018.
    """

    def __init__(
        self,
        n_data: int,
        rec_dist: ReconDist = ReconDist.BERNOULLI,
        steps_anneal: int = 0,
        alpha: float = 1.0,
        beta: float = 6.0,
        gamma: float = 1.0,
        is_mss: bool = True,
    ):
        super().__init__(rec_dist=rec_dist, steps_anneal=steps_anneal)

        self.n_data = n_data
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.is_mss = is_mss  # minibatch stratified sampling

    def calculate_loss(
        self,
        data,
        reconstructed_data,
        latent_dist,
        is_train,
    ):
        rec_loss = _reconstruction_loss(
            data,
            reconstructed_data,
            reduction="sum",
            distribution=self.rec_dist,
        )

        _, _, latent_sample = latent_dist
        log_pz, log_qz, log_prod_qzi, log_q_zcx = _get_log_pz_qz_prodzi_qzcx(
            latent_sample, latent_dist, self.n_data, is_mss=self.is_mss
        )

        # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
        mi_loss = (log_q_zcx - log_qz).mean()
        # TC[z] = KL[q(z)||\prod_i z_i]
        tc_loss = (log_qz - log_prod_qzi).mean()
        # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
        dw_kl_loss = (log_prod_qzi - log_pz).mean()

        anneal_reg = (
            _linear_annealing(0, 1, self.n_train_steps, self.steps_anneal)
            if is_train
            else 1
        )

        # total loss
        loss = rec_loss + (
            self.alpha * mi_loss
            + self.beta * tc_loss
            + anneal_reg * self.gamma * dw_kl_loss
        )

        return loss


def _reconstruction_loss(
    data: torch.Tensor,
    reconstructed_data: torch.Tensor,
    reduction: str = "none",
    distribution: ReconDist = ReconDist.GAUSSIAN,
) -> torch.Tensor:
    """
    Calculates the per image reconstruction loss for a batch of data. I.e. negative
    log likelihood.

    Parameters
    ----------
    data : torch.Tensor
        Input data (e.g. batch of images). Shape : (batch_size, n_chan,
        height, width).

    reconstructed_data : torch.Tensor
        Reconstructed data. Shape : (batch_size, n_chan, height, width).

    distribution : {"bernoulli", "gaussian", "laplace"}
        Distribution of the likelihood on the each pixel. Implicitely defines the
        loss Bernoulli corresponds to a binary cross entropy (bse) loss and is the
        most commonly used. It has the issue that it doesn't penalize the same
        way (0.1,0.2) and (0.4,0.5), which might not be optimal. Gaussian
        distribution corresponds to MSE, and is sometimes used, but hard to train
        ecause it ends up focusing only a few pixels that are very wrong. Laplace
        distribution corresponds to L1 solves partially the issue of MSE.

    storer : dict
        Dictionary in which to store important variables for vizualisation.

    Returns
    -------
    loss : torch.Tensor
        Per image cross entropy (i.e. normalized per batch but not pixel and
        channel)
    """
    batch_size, _, _, _ = reconstructed_data.size()

    if distribution == ReconDist.BERNOULLI:
        loss = (
            F.binary_cross_entropy(reconstructed_data, data)
            if reduction != "sum"
            else F.binary_cross_entropy(reconstructed_data, data, reduction=reduction)
        )
    elif distribution == ReconDist.GAUSSIAN:
        loss = (
            F.mse_loss(reconstructed_data, data)
            if reduction != "sum"
            else F.mse_loss(reconstructed_data, data, reduction=reduction)
        )
    elif distribution == ReconDist.LAPLACE:
        loss = (
            F.l1_loss(reconstructed_data, data)
            if reduction != "sum"
            else F.l1_loss(reconstructed_data, data, reduction=reduction)
        )
        # emperical value to give similar values than bernoulli => use same hyperparam
        loss = loss * 3
        loss = loss * (loss != 0)  # masking to avoid nan
    else:
        assert distribution not in iter(ReconDist)
        raise ValueError(f"Unknown distribution: {distribution}")

    if reduction == "sum":
        loss = loss / batch_size

    return loss


def _kl_normal_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Calculates the KL divergence between a normal distribution
    with diagonal covariance and a unit normal distribution.

    Parameters
    ----------
    mean : torch.Tensor
        Mean of the normal distribution. Shape (batch_size, latent_dim) where
        D is dimension of distribution.

    logvar : torch.Tensor
        Diagonal log variance of the normal distribution. Shape (batch_size,
        latent_dim)

    storer : dict
        Dictionary in which to store important variables for vizualisation.
    """
    # batch mean of kl for each latent dimension
    latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=0)
    total_kl = latent_kl.sum()

    return total_kl


def _permute_dims(latent_sample: torch.Tensor) -> torch.Tensor:
    """
    Implementation of Algorithm 1 in ref [1]. Randomly permutes the sample from
    q(z) (latent_dist) across the batch for each of the latent dimensions (mean
    and log_var).

    Parameters
    ----------
    latent_sample: torch.Tensor
        sample from the latent dimension using the reparameterisation trick

    Returns
    -------
    perm: torch.Tensor
        permuted latent sample

    References
    ----------
        [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
        arXiv preprint arXiv:1802.05983 (2018).

    """
    perm = torch.zeros_like(latent_sample)
    batch_size, dim_z = perm.size()

    for z in range(dim_z):
        pi = torch.randperm(batch_size).to(latent_sample.device)
        perm[:, z] = latent_sample[pi, z]

    return perm


def _linear_annealing(
    init: float, fin: float, step: int, annealing_steps: int
) -> float:
    """
    Linear annealing of a parameter.

    Parameters
    ----------
    init : float
        Initial value of the parameter.

    fin : float
        Final value of the parameter.

    step : int
        Current step in the annealing process.

    annealing_steps : int
        Total number of annealing steps.

    Returns
    -------
    annealed : float
        Annealed value of the parameter.
    """
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed


# Batch TC specific
def _get_log_pz_qz_prodzi_qzcx(
    latent_sample: torch.Tensor,
    latent_dist: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    n_data: int,
    is_mss: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the log probabilities for p(z), q(z), prod(q(z_i)), and q(z|x).

    Parameters
    ----------
    latent_sample : torch.Tensor
        Sample from the latent dimension using the reparameterization trick.
        Shape: (batch_size, latent_dim).

    latent_dist : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Tuple containing the mean, log variance, and standard deviation of the
        latent distribution. Shape: (batch_size, latent_dim).

    n_data : int
        Number of data in the training set.

    is_mss : bool, optional
        Whether to use minibatch stratified sampling instead of minibatch weighted
        sampling. Default is True.

    Returns
    -------
    log_pz : torch.Tensor
        Log probability of p(z). Shape: (batch_size,).

    log_qz : torch.Tensor
        Log probability of q(z). Shape: (batch_size,).

    log_prod_qzi : torch.Tensor
        Log probability of prod(q(z_i)). Shape: (batch_size,).

    log_q_zCx : torch.Tensor
        Log probability of q(z|x). Shape: (batch_size,).
    """
    batch_size, _ = latent_sample.shape

    # calculate log q(z|x)
    mean, logvar, _ = latent_dist
    log_q_zCx = _log_density_gaussian(latent_sample, mean, logvar).sum(dim=1)

    # calculate log p(z)
    # mean and log var is 0
    zeros = torch.zeros_like(latent_sample)
    log_pz = _log_density_gaussian(latent_sample, zeros, zeros).sum(1)

    mat_log_qz = _matrix_log_density_gaussian(latent_sample, mean, logvar)

    if is_mss:
        # use stratification
        log_iw_mat = _log_importance_weight_matrix(batch_size, n_data).to(
            latent_sample.device
        )
        mat_log_qz = mat_log_qz + log_iw_mat.view(batch_size, batch_size, 1)

    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
    log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(1)

    return log_pz, log_qz, log_prod_qzi, log_q_zCx


def _matrix_log_density_gaussian(
    x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    """Calculates log density of a Gaussian for all combination of batch pairs of
    `x` and `mu`. I.e. return tensor of shape `(batch_size, batch_size, dim)`
    instead of (batch_size, dim) in the usual log density.

    Parameters
    ----------
    x: torch.Tensor
        Value at which to compute the density. Shape: (batch_size, dim).

    mu: torch.Tensor
        Mean. Shape: (batch_size, dim).

    logvar: torch.Tensor
        Log variance. Shape: (batch_size, dim).

    Returns
    -------
    log_density: torch.Tensor
        Log density of a Gaussian. Shape: (batch_size, batch_size, dim).
    """
    batch_size, dim = x.shape
    x = x.view(batch_size, 1, dim)
    mu = mu.view(1, batch_size, dim)
    logvar = logvar.view(1, batch_size, dim)
    return _log_density_gaussian(x, mu, logvar)


def _log_density_gaussian(
    x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    """Calculates log density of a Gaussian.

    Parameters
    ----------
    x: torch.Tensor
        Value at which to compute the density.

    mu: torch.Tensor
        Mean.

    logvar: torch.Tensor
        Log variance.
    """
    normalization = -0.5 * (math.log(2 * math.pi) + logvar)
    inv_var = torch.exp(-logvar)
    log_density = normalization - 0.5 * ((x - mu) ** 2 * inv_var)
    return log_density


def _log_importance_weight_matrix(batch_size: int, dataset_size: int) -> torch.Tensor:
    """
    Calculates a log importance weight matrix

    Parameters
    ----------
    batch_size: int
        number of training images in the batch

    dataset_size: int
        number of training images in the dataset
    """
    N = dataset_size
    M = batch_size - 1 if batch_size > 1 else 1
    strat_weight = (N - M) / (N * M)
    W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[:: M + 1] = 1 / N
    W.view(-1)[1 :: M + 1] = strat_weight
    W[M - 1, 0] = strat_weight
    return W.log()
