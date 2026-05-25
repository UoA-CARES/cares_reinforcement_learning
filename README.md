<p align="center">
<img src="./media/logo.png" alt="CARES reinforcement learning package logo" style="width: 80%;"/>
</p>

# CARES Reinforcement Learning

Branch:

```text
feature/fractional-swish-gelu
```

The CARES Reinforcement Learning package provides a modular reinforcement learning (RL) framework used as the foundation for RL-related research projects within the CARES research group.

This branch extends the CARES RL framework with:

- fractional-order activation functions
- Grünwald-Letnikov-inspired nonlinearities
- residual fractional GELU activations
- adaptive residual fractional activations
- smooth fractional Swish variants
- RL-safe fractional activations
- configurable actor/critic activation placement

for systematic evaluation in modern actor-critic reinforcement learning algorithms such as TD3 and SAC.

---

# Motivation

**Reinforcement Learning Algorithms** (that is to say, *how* the neural networks are updated) remain fundamentally similar across applications.

This package is designed so that RL algorithms are implemented once and can then be:

- reused
- extended
- configured
- evaluated

across many environments and research settings.

The framework supports modular integration of:

- neural network architectures
- replay buffers
- encoders
- activation functions
- optimisation methods

without modifying the underlying RL algorithm implementation.

---

# Fractional Activation Research Motivation

This branch investigates whether fractional-order nonlinear transformations can improve:

- optimisation stability
- gradient propagation
- representation smoothness
- actor-critic learning dynamics
- continuous-control RL performance
- RL sample efficiency

compared to standard activations such as:

- ReLU
- GELU
- Swish / SiLU

The branch focuses particularly on smooth nonlinear and fractional-inspired activations designed for off-policy actor-critic RL.

---

# Fractional Activation References and Inspiration

Several activations implemented in this branch are inspired by prior work in:

- fractional calculus
- fractional neural networks
- smooth nonlinear activation functions
- residual learning
- adaptive activation functions

while introducing RL-oriented stabilisation and residual extensions for actor-critic reinforcement learning.

---

# Fractional Neural Network Inspiration

This branch was primarily inspired by:

Z. Alijani and V. Molek,  
"Fractional concepts in neural networks: Enhancing activation and loss functions",  
arXiv preprint arXiv:2310.11875, 2023.

Paper:

https://arxiv.org/abs/2310.11875

Reference implementation:

https://gitlab.com/irafm-ai/frac_calc_ann

The paper explores the use of fractional concepts in neural networks through:

- fractional activation functions
- fractional-order nonlinearities
- Grünwald-Letnikov approximations
- fractional modifications of learning dynamics

Several activations implemented in this branch adapt and extend these ideas for reinforcement learning and actor-critic optimisation.

---

# Residual Learning Inspiration

The residual fractional activations are conceptually related to residual learning introduced in:

K. He et al.,  
"Deep Residual Learning for Image Recognition",  
CVPR 2016.

Paper:

https://arxiv.org/abs/1512.03385

The residual fractional activations preserve a stable baseline activation while injecting a controlled fractional residual correction:

$$
f(x)=\mathrm{BaseActivation}(x)+\mathrm{FractionalResidual}(x)
$$

This residual formulation was motivated by the observation that residual structures often improve:

- optimisation stability
- gradient propagation
- deep network training dynamics

particularly in deep actor-critic reinforcement learning.

---

# Adaptive Activation Inspiration

The adaptive residual activations are additionally inspired by learnable activation function research including:

- PReLU
- Swish
- adaptive gating activations
- learnable nonlinear scaling

Related references include:

K. He et al.,  
"Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification",  
2015.

https://arxiv.org/abs/1502.01852

P. Ramachandran et al.,  
"Searching for Activation Functions",  
2017.

https://arxiv.org/abs/1710.05941

A. Elfwing et al.,  
"Sigmoid-weighted linear units for neural network function approximation in reinforcement learning",  
2017.

https://arxiv.org/abs/1702.03118

The `AdaptiveResidualFractionalGELU` activation extends these ideas through learnable fractional residual balancing:

$$
f(x)=
\mathrm{GELU}(x)
+
\beta
\mathrm{FractionalTerm}(x)
$$

where the fractional contribution is adaptively learned during training.

---

# Smooth Activation Inspiration

Several activations implemented in this branch build upon modern smooth nonlinear activations including GELU, Swish, and Mish.

Related references:

D. Hendrycks and K. Gimpel,  
"Gaussian Error Linear Units (GELUs)",  
2016.

https://arxiv.org/abs/1606.08415

D. Misra,  
"Mish: A Self Regularized Non-Monotonic Activation Function",  
2019.

https://arxiv.org/abs/1908.08681

These smooth activations are used as stable nonlinear baselines before introducing fractional nonlinear modulation.

---

# Fractional Activation Implementations

Implemented in:

```text
cares_reinforcement_learning/networks/fractional_activations.py
```

Dynamically loaded through:

```text
cares_reinforcement_learning/networks/common.py
```

The activations implemented in this branch fall into several different mathematical categories.

---

# 1. Fractional Derivative Inspired Activations

These activations are inspired by Grünwald-Letnikov fractional derivative approximations.

They use:

- Gamma-function coefficients
- shifted finite summations
- fractional finite differences
- fractional accumulation dynamics

These are the activations most closely aligned with classical fractional calculus.

---

## FractionalGELU

Fractional derivative inspired GELU activation.

Base activation:

$$
\mathrm{GELU}(x)
$$

Uses Grünwald-Letnikov-style finite fractional summation:

$$
\sum_{i=0}^{n}(-1)^i
\frac{\Gamma(\alpha+1)}
{\Gamma(i+1)\Gamma(\alpha-i+1)}
f(x-ih)
$$

Characteristics:

- closest implementation to true fractional calculus
- smooth nonlinear behaviour
- fractional accumulation effects
- expressive nonlinear transitions
- higher computational complexity

---

## FractionalMish

Fractional derivative inspired Mish activation.

Base activation:

$$
\mathrm{Mish}(x)=x\tanh(\mathrm{softplus}(x))
$$

Characteristics:

- smooth non-monotonic activation
- fractional finite-difference behaviour
- expressive smooth transitions
- fractional nonlinear modulation

---

## SafeGLFractionalGELU

RL-safe version of `FractionalGELU`.

Adds:

- stable Gamma handling
- alpha stabilisation
- safe bounded behaviour
- `nan_to_num`

Designed specifically for stable actor-critic RL training.

---

## SafeGLFractionalMish

RL-safe version of `FractionalMish`.

Adds:

- stable fractional computation
- bounded numerical behaviour
- safer optimisation dynamics

---

# 2. Fractional Power-Law Activations

These activations use fractional-order power-law scaling.

Core formulation:

$$
|x|^{1-a}
$$

These activations are simpler and often more numerically stable than full Grünwald-Letnikov formulations.

---

## FractionalReLUPositive

Positive-only fractional ReLU variant.

Formulation:

$$
f(x)=x^{1-a}, \quad x>0
$$

Characteristics:

- positive fractional rectification
- sparse activation behaviour
- simple fractional scaling
- ReLU-inspired dynamics

---

## FLReLU

Fractional Leaky ReLU.

Characteristics:

- fractional positive and negative scaling
- fixed negative leakage
- stable rectifier dynamics

---

## FLReLU2

Learnable fractional Leaky ReLU.

Adds:

- learnable leakage
- adaptive negative response
- trainable nonlinear asymmetry

---

## FractionalPReLU

Fractional parametric ReLU.

Characteristics:

- fractional-order rectification
- trainable negative branch
- adaptive nonlinear response

---

## FPReLU2

Learnable fractional PReLU variant.

Characteristics:

- adaptive negative scaling
- learnable leakage dynamics
- trainable rectifier behaviour

---

## FParReLU

Fractional parametric ReLU with learnable slope parameters.

Characteristics:

- learnable negative branch
- stable bounded scaling
- adaptive fractional rectification

---

## SafeFractionalGELU

Fractionally-scaled GELU.

Base activation:

$$
\mathrm{GELU}(x)
$$

Fractional scaling:

$$
f(x)=
\mathrm{GELU}(x)
\frac{
(|x|+\epsilon)^{1-a}
}{
\Gamma(2-a)
}
$$

Characteristics:

- RL-safe fractional modulation
- stable magnitude scaling
- smoother optimisation behaviour
- lower instability than full GL approximations

---

## SafeFractionalMish

Fractionally-scaled Mish activation.

Characteristics:

- smooth fractional scaling
- RL-safe behaviour
- stable nonlinear modulation

---

# 3. Fractional-Inspired Smooth Gated Activations

These activations extend smooth nonlinearities such as Swish and SiLU using fractional-inspired modulation.

These are not strict fractional derivatives.

Instead, they inject adaptive fractional-style nonlinear behaviour into smooth gating mechanisms.

---

## FractionalSwish

Fractional-inspired Swish activation.

Base activation:

$$
\mathrm{Swish}(x)=x\sigma(x)
$$

Fractional-inspired modulation:

$$
f(x)=
\mathrm{Swish}(x)
+
\alpha\sigma(x)(1-\mathrm{Swish}(x))
$$

Characteristics:

- smooth gated behaviour
- adaptive nonlinear modulation
- Swish-family extension
- smooth actor optimisation
- stable gradients

---

## FractionalSwishBeta

Learnable-beta Fractional Swish.

Adds:

- trainable gating strength
- adaptive smoothness
- learnable nonlinear response

Characteristics:

- stronger flexibility
- adaptive gating dynamics
- RL-friendly smooth transitions

---

## FALU

Fractional Adaptive Linear Unit.

Core structure:

$$
g(x)=x\sigma(\beta x)
$$

Characteristics:

- adaptive smooth gating
- learnable nonlinear behaviour
- higher-order fractional-inspired modulation
- smooth actor-critic optimisation

---

## SafeFractionalSwish

RL-safe version of `FractionalSwish`.

Adds:

- bounded numerical behaviour
- `nan_to_num`
- RL-oriented stabilisation

---

## SafeFALU

RL-safe version of `FALU`.

Characteristics:

- stable smooth gating
- bounded optimisation dynamics
- safer actor-critic training

---

# 4. Residual Fractional Activations

These activations preserve the baseline activation while injecting a controlled fractional residual correction.

These were designed specifically for RL stability.

---

## ResidualFractionalGELU

Residual fractional GELU activation.

Base activation:

$$
\mathrm{GELU}(x)
$$

Residual fractional augmentation:

$$
f(x)=
\mathrm{GELU}(x)
+
\beta
\frac{
sign(x)|x|^{1-a}
}{
\Gamma(2-a)
}
$$

Characteristics:

- preserves baseline GELU geometry
- injects controlled fractional dynamics
- stable optimisation behaviour
- smoother critic learning
- improved gradient propagation
- RL-oriented residual nonlinear modulation

---

## AdaptiveResidualFractionalGELU

Adaptive residual fractional GELU.

Extends `ResidualFractionalGELU` with learnable residual scaling.

Adds:

- adaptive residual strength
- trainable fractional correction
- learnable nonlinear balancing

Characteristics:

- dynamically learns fractional influence
- preserves stable baseline activation
- adaptive RL optimisation behaviour

---

# Usage

Consult the repository wiki for usage examples and documentation:

https://github.com/UoA-CARES/cares_reinforcement_learning/wiki

---

# Installation Instructions

![Python](https://img.shields.io/badge/python-3.10--3.12-blue.svg)

Clone repository:

```bash
git clone https://github.com/UoA-CARES/cares_reinforcement_learning.git
```

Checkout fractional activation branch:

```bash
git checkout feature/fractional-swish-gelu
```

Install requirements:

```bash
pip3 install -r requirements.txt
```

Install editable package:

```bash
pip3 install --editable .
```

---

# Related Training Framework

Training orchestration and experiment execution are handled through:

https://github.com/UoA-CARES/gymnasium_envrionments

That repository provides:

- OpenAI Gymnasium integration
- DeepMind Control Suite integration
- batch execution
- plotting
- experiment management
- multi-seed orchestration

---

# Running Fractional Activation Experiments

Example:

```bash
ACTIVATION=ResidualFractionalGELU \
ALPHAS=0.1,0.2,0.3,0.4,0.5 \
ALGORITHM=TD3 \
LAYERS=1 \
PLACEMENT=all_both \
python3 run.py train cli \
--gym openai \
--task HalfCheetah-v4 \
--batch 1 \
TD3 \
--seeds 10 20 30 40 50 \
--max_workers 5
```

---

# Package Structure

```text
cares_reinforcement_learning/
├─ algorithm/
├─ encoders/
├─ memory/
├─ networks/
│  ├─ common.py
│  ├─ fractional_activations.py
├─ util/
```

---

# Supported Algorithms

The framework supports a broad range of reinforcement learning algorithms including:

## Q-Learning

- DQN
- DoubleDQN
- Rainbow
- QRDQN
- PERDQN
- NoisyNet
- C51

---

## Actor-Critic

- PPO
- DDPG
- TD3
- SAC
- REDQ
- TQC
- CrossQ
- TD7
- CTD4
- DroQ
- PALTD3
- LAPTD3
- LA3PTD3
- MAPERTD3
- MAPERSAC
- SDAR

including image-based variants such as:

- TD3AE
- SACAE
- NaSATD3

---

## Multi-Agent RL

- QMIX
- MADDPG
- M3DDPG

---

## Unsupervised Skill Discovery

- DIAYN
- DADS

---

# Citation

```text
@misc{cares_reinforcement_learning,
  title = {CARES Reinforcement Learning},
  author = {CARES},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/UoA-CARES/cares_reinforcement_learning}
}
```
