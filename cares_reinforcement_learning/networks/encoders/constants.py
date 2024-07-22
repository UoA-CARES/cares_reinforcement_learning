from enum import Enum


class Autoencoders(str, Enum):
    BURGESS = "Burgess"
    SQVAE = "SQVAE"
    AE = "AE"


class Losses(str, Enum):
    VAE = "vae"
    BETA_H = "betaH"
    BETA_B = "betaB"
    FACTOR = "factor"
    BTCVAE = "btcVAE"
    AE = "ae"
    SQVAE = "SQVAE"


class ReconDist(str, Enum):
    BERNOULLI = "bernoulli"
    LAPLACE = "laplace"
    GAUSSIAN = "gaussian"


class ParamVarQ(str, Enum):
    GAUSSIAN_1 = "gaussian_1"
    GAUSSIAN_2 = "gaussian_2"
    GAUSSIAN_3 = "gaussian_3"
    GAUSSIAN_4 = "gaussian_4"
