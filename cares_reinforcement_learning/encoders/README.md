# Autoencoders
This folder contains the implementation of AEs and VAEs for the purposes of training RL algorithms.
All autoencoder implementations are generalised for the image input size. 

# Implementations
The autoencoder family encompasses a range of neural network architectures designed for unsupervised learning tasks, particularly dimensionality reduction and feature learning. At its core, an autoencoder consists of an encoder that compresses input data into a latent representation and a decoder that reconstructs the original data from this compressed form. Variants of autoencoders, such as Variational Autoencoders (VAEs) and Beta-VAEs, introduce probabilistic elements and regularization techniques to enhance the quality and interpretability of the latent space. While standard autoencoders focus on reconstruction accuracy, advanced variants like Beta-VAE and Squared VAE (SqVAE) aim to improve latent space disentanglement and sparsity, making them valuable for generating more meaningful and structured representations.

## Code Interface
The autoencoder implementations are designed to be generally pieced together for a varietey of uses cases.
The basic structure and usage of the autoencoders is laid out below.

### Base Autoencoder (autoencoder.py)
All autoencoders derive from the autoencoder class that defines the required base implementation.
This class defines the base requirements for general use throughout the RL package. 

### Loss Functions (losses.py)
The losses module contains the general loss functions for all of the autoencoder implementations. 
This allows the use of these loss functions in more general situations beyond the specific autoencoder class (e.g. SACAE/TD3AE as examples of this).

### Autoencoder Factory (autoencoder_factory.py)
The autoencoder factory is the best means of creating the specific autoencoder implementations detailed below. 
This relies on creating the specific configurations (configurations.py) for each autoencoder and passing it through to the factory to assemble the autoencoder. 

# Vanilla Autoencoder
A basic Autoencoder is a type of neural network designed to learn efficient representations of data by encoding it into a lower-dimensional space and then decoding it back to its original form. The vanilla autoencoder is a basic autoencoder implementation utilising the L2 loss function.
There is an implementation of a 1D and 2D version for use on vector and image based inputs. 

# Variational Autoencoder
A Variational Autoencoder (VAE) is an advanced type of autoencoder that introduces probabilistic elements to learn a latent space with meaningful structure. Unlike basic autoencoders, VAEs model the latent representation as a probability distribution rather than a fixed vector. The encoder outputs the parameters of this distribution, while the decoder reconstructs the input from samples drawn from it. VAEs are trained to maximize a lower bound on the likelihood of the data, balancing between reconstruction accuracy and the regularization of the latent space through KL divergence. This approach enables VAEs to generate new, similar data samples and learn robust, interpretable features in the latent space.

The variational autoencoder ([VAE](https://arxiv.org/abs/1312.6114)) is implemented as a varient of the Burgess Autoencoder (BetaHLoss) where α=β=ɣ=1. Each term is computed exactly by a closed form solution (KL between the prior and the posterior). Tightest lower bound.

## Burgess (Beta-VAE) Autoencoders
The Burgess Autoencoder is an advanced variant of the Variational Autoencoder (VAE) designed to enhance disentanglement in latent space representations. It introduces a hyperparameter, β, which scales the KL divergence term in the loss function, allowing for greater control over the trade-off between reconstruction fidelity and latent space regularization. By gradually increasing β during training through linear annealing this encourages the model to learn more interpretable and independent features in the latent space, making it particularly useful for tasks that benefit from disentangled representations.

- [BetaH](https://openreview.net/pdf?id=Sy2fzU9gl): α=β=ɣ>1. Each term is computed exactly by a closed form solution. Simply adds a hyper-parameter (β in the paper) before the KL.

- [BetaB](https://arxiv.org/abs/1804.03599): α=β=ɣ>1. Same as β-VAEH but only penalizes the 3 terms once they deviate from a capacity C which increases during training.

- [Factor](https://arxiv.org/abs/1802.05983): α=ɣ=1, β>1. Each term is computed exactly by a closed form solution. Simply adds a hyper-parameter (β in the paper) before the KL. Adds a weighted Total Correlation term to the standard VAE loss. The total correlation is estimated using a classifier and the density-ratio trick.

- [BTCVAE](https://arxiv.org/abs/1802.04942): α=ɣ=1 (although can be modified), β>1. Conceptually equivalent to FactorVAE, but each term is estimated separately using minibatch stratified sampling.

The original implementations that have been refactored into our workspace can be found here: [disentangling-vae](https://github.com/YannDubs/disentangling-vae/tree/master) all credit to the original authors/developers for these methods.

## SQ-VAE (WIP)
SQ-VAE (Stochastic Quantization Variational Autoencoder) is a variant of the Variational Autoencoder that incorporates stochastic quantization into its architecture. It discretizes the latent space by mapping continuous latent variables to discrete values with added randomness. This approach helps in learning more structured and interpretable latent representations while maintaining the flexibility of probabilistic modeling. The stochastic quantization process improves the model’s ability to capture complex data distributions and generate diverse samples.

The original implementation that has been refactored into our workspace can be found here: [sqvae](https://github.com/sony/sqvae) all credit to the original authors/developers for this method.