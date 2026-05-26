<p align="center">
<img src="./media/logo.png" alt="CARES reinforcement learning package logo" style="width: 80%;"/>
</p>

# CARES Reinforcement Learning
### Fractional Activation Research Branch

Branch:

```text
feature/fractional-swish-gelu
```

The CARES Reinforcement Learning package provides a modular reinforcement learning (RL) framework used for RL research within the CARES research group.

This branch extends the CARES RL framework with:

- fractional-inspired activation functions
- Grünwald-Letnikov-inspired nonlinearities
- residual fractional activations
- adaptive residual fractional activations
- RL-safe stabilised activations
- smooth Swish/GELU/Mish nonlinearities
- configurable actor/critic activation placement

for systematic evaluation in modern actor-critic RL algorithms such as:

- TD3
- SAC

---

# Motivation

Modern RL algorithms are highly sensitive to optimisation stability and gradient flow.

Small changes in activation functions can significantly affect:

- exploration behaviour
- critic stability
- representation learning
- convergence speed
- optimisation smoothness
- sample efficiency

This branch investigates whether fractional-inspired nonlinearities can improve these properties compared to standard activations such as:

- ReLU
- LeakyReLU
- PReLU
- GELU
- Swish / SiLU

with a particular focus on off-policy actor-critic RL.

---

# Research Direction

This branch explores three core ideas:

| Idea | Goal |
|---|---|
| Fractional activations | Introduce richer nonlinear behaviour |
| Safe activations | Improve RL numerical stability |
| Residual/adaptive residual activations | Preserve stable baseline activations while injecting fractional behaviour |

---

# Main Fractional Inspiration

This work was primarily inspired by:

Z. Alijani and V. Molek,  
**"Fractional concepts in neural networks: Enhancing activation and loss functions"**  
arXiv preprint arXiv:2310.11875, 2023.

Paper:

https://arxiv.org/abs/2310.11875

Reference implementation:

https://gitlab.com/irafm-ai/frac_calc_ann

The paper explores the use of:

- fractional activation functions
- fractional-order nonlinearities
- Grünwald-Letnikov approximations
- fractional modifications of neural network learning dynamics

Several activations implemented in this branch adapt and extend these ideas specifically for RL and actor-critic optimisation.

---

# Residual and Adaptive Residual Inspiration

The residual fractional activations implemented in this branch are conceptually related to residual learning introduced in:

K. He et al.,  
**"Deep Residual Learning for Image Recognition"**  
CVPR 2016.

https://arxiv.org/abs/1512.03385

Residual learning introduced the idea of preserving a stable baseline transformation while learning an additional residual correction:

$$
f(x)=x+\mathcal{F}(x)
$$

The residual fractional activations in this branch follow a similar philosophy:

$$
f(x)=
\mathrm{BaseActivation}(x)
+
\mathrm{FractionalResidual}(x)
$$

where a standard activation such as GELU, Swish, or PReLU is preserved while a fractional nonlinear correction is added.

The motivation is to preserve the stable optimisation behaviour of modern activations while introducing additional fractional nonlinear expressiveness in a controlled way.

---

## Adaptive Residual Inspiration

The adaptive residual activations are additionally inspired by learnable activation and adaptive scaling methods including:

### PReLU

K. He et al.,  
**"Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"**  
2015.

https://arxiv.org/abs/1502.01852

which introduced learnable activation parameters.

---

### Highway Networks

R. K. Srivastava et al.,  
**"Highway Networks"**  
2015.

https://arxiv.org/abs/1505.00387

which introduced adaptive gating mechanisms controlling information flow through neural layers.

---

### Swish

P. Ramachandran et al.,  
**"Searching for Activation Functions"**  
2017.

https://arxiv.org/abs/1710.05941

which explored smooth adaptive gating behaviour in activations.

---

The adaptive residual activations implemented in this branch extend these ideas by learning the strength of the fractional residual contribution during training:

$$
f(x)=
\mathrm{BaseActivation}(x)
+
\beta
\mathrm{FractionalResidual}(x)
$$

where:

- the base activation preserves stable optimisation behaviour
- the learnable parameter $\beta$ determines how much fractional behaviour should be added

This allows the network to automatically increase or suppress the fractional contribution depending on its usefulness for the RL objective.

---

# Additional Activation References

---

## ReLU

V. Nair and G. Hinton,  
**"Rectified Linear Units Improve Restricted Boltzmann Machines"**  
ICML 2010.

https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf

---

## LeakyReLU

A. L. Maas et al.,  
**"Rectifier Nonlinearities Improve Neural Network Acoustic Models"**  
2013.

https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf

---

## GELU

D. Hendrycks and K. Gimpel,  
**"Gaussian Error Linear Units (GELUs)"**  
2016.

https://arxiv.org/abs/1606.08415

---

## Mish

D. Misra,  
**"Mish: A Self Regularized Non-Monotonic Activation Function"**  
2019.

https://arxiv.org/abs/1908.08681

---

## SiLU / Swish in RL

A. Elfwing et al.,  
**"Sigmoid-weighted linear units for neural network function approximation in reinforcement learning"**  
2017.

https://arxiv.org/abs/1702.03118

---

# Core Concepts

---

## 1. Fractional Activations

These activations introduce fractional-order nonlinear behaviour inspired by fractional calculus.

The goal is to create richer nonlinear transformations than standard activations.

Examples:

- `FractionalGELU`
- `FractionalSwish`
- `FractionalPReLU`
- `FLReLU`

---

## 2. Safe Activations

RL training can become unstable when fractional nonlinearities produce:

- exploding values
- NaNs
- unstable gradients

Safe versions introduce stabilisation mechanisms such as:

- clipping
- epsilon stabilisation
- bounded scaling
- `nan_to_num`

Examples:

- `SafeFractionalGELU`
- `SafeFractionalSwish`
- `SafeGLFractionalGELU`
- `SafeFALU`

### Intuition

```text
Fractional activation
+ RL-oriented stabilisation
```

---

## 3. Residual Fractional Activations

Pure fractional activations sometimes modify the activation behaviour too aggressively.

Residual versions preserve the original activation and only add a controlled fractional correction.

General idea:

```text
BaseActivation(x)
+ FractionalResidual(x)
```

Examples:

- `ResidualFractionalGELU`
- `ResidualFractionalSwish`
- `ResidualFLReLU`
- `ResidualFractionalPReLU`

### Intuition

```text
Stable original activation
+ controlled fractional correction
```

This often produces smoother optimisation behaviour in RL.

---

## 4. Adaptive Residual Activations

Adaptive residual activations extend the residual idea further.

Instead of manually selecting the residual strength, the network learns it automatically during training.

General idea:

```text
BaseActivation(x)
+ learned_beta * FractionalResidual(x)
```

Examples:

- `AdaptiveResidualFractionalGELU`
- `AdaptiveResidualFractionalSwish`
- `AdaptiveResidualFLReLU`
- `AdaptiveResidualFractionalPReLU`

### Intuition

If the fractional component helps learning:

- the network increases its influence

If it hurts optimisation:

- the network reduces it automatically

This allows the model to learn the best balance between standard and fractional behaviour.

---

# Activation Categories

---

# 1. Grünwald-Letnikov Fractional Activations

These are the activations most closely related to classical fractional calculus.

They use fractional finite-difference approximations inspired by Grünwald-Letnikov derivatives.

## Implemented

- `FractionalGELU`
- `FractionalMish`
- `SafeGLFractionalGELU`
- `SafeGLFractionalMish`

### Characteristics

- strongest connection to fractional calculus
- expressive nonlinear behaviour
- more computationally complex
- potentially less stable without safeguards

---

# 2. Fractional Power-Law Activations

These use fractional power scaling:

```text
|x|^(1-a)
```

instead of full fractional derivative approximations.

## Implemented

- `FractionalReLUPositive`
- `FLReLU`
- `FLReLU2`
- `FractionalPReLU`
- `FPReLU2`
- `FParReLU`

### Characteristics

- simpler than GL activations
- easier to stabilise
- rectifier-style behaviour
- fractional nonlinear scaling

---

# 3. Smooth Fractional Activations

These extend smooth nonlinearities such as Swish, GELU, and Mish.

## Implemented

- `FractionalSwish`
- `FractionalSwishBeta`
- `FALU`
- `SafeFractionalSwish`
- `SafeFractionalSwishBeta`
- `SafeFALU`
- `SafeFractionalGELU`
- `SafeFractionalMish`

### Characteristics

- smooth gradients
- smoother actor optimisation
- RL-friendly optimisation behaviour
- fractional-inspired gating dynamics

---

# 4. Residual Fractional Smooth Activations

These preserve the original activation while injecting fractional corrections.

## GELU Family

- `ResidualFractionalGELU`
- `AdaptiveResidualFractionalGELU`

## Swish Family

- `ResidualFractionalSwish`
- `AdaptiveResidualFractionalSwish`
- `ResidualFractionalSwishBeta`
- `AdaptiveResidualFractionalSwishBeta`

### Characteristics

- preserve smooth baseline activations
- inject controlled fractional behaviour
- smoother optimisation
- adaptive residual balancing
- more RL-stable than pure fractional replacements

---

# 5. Residual Fractional Rectifier Activations

These apply the residual/adaptive residual idea to ReLU-style activations.

## Implemented

- `ResidualFractionalReLUPositive`
- `AdaptiveResidualFractionalReLUPositive`
- `ResidualFLReLU`
- `AdaptiveResidualFLReLU`
- `ResidualFLReLU2`
- `AdaptiveResidualFLReLU2`
- `ResidualFractionalPReLU`
- `AdaptiveResidualFractionalPReLU`

### Characteristics

- preserve stable rectifier behaviour
- inject controlled fractional corrections
- maintain stable negative leakage behaviour
- allow adaptive residual balancing

---

# Adaptive Residual vs PReLU

These are conceptually different.

## PReLU

PReLU learns:

```text
the negative slope of the activation
```

## Adaptive Residual Activations

Adaptive residual activations learn:

```text
how much fractional behaviour should be added
```

So:

| Activation | Learns |
|---|---|
| PReLU | negative slope |
| Adaptive residual activation | fractional residual strength |

---

# File Locations

Fractional activations are implemented in:

```text
cares_reinforcement_learning/networks/fractional_activations.py
```

Dynamic loading occurs through:

```text
cares_reinforcement_learning/networks/common.py
```

---

# Installation

Clone repository:

```bash
git clone https://github.com/UoA-CARES/cares_reinforcement_learning.git
```

Checkout branch:

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

Training orchestration is handled through:

https://github.com/UoA-CARES/gymnasium_envrionments

That repository provides:

- environment wrappers
- experiment management
- batch execution
- plotting utilities
- multi-seed training
- OpenAI Gymnasium integration
- DeepMind Control Suite integration

---

# Example Experiment

```bash
ACTIVATION=AdaptiveResidualFractionalGELU \
ALPHA=0.1 \
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

# Supported Placement Strategies

## 1-Layer Networks

```text
all_both
```

## 2-Layer Networks

```text
all_both
all_actor
all_critic
first_both
first_actor
first_critic
```

---

# Package Structure

```text
cares_reinforcement_learning/
├── algorithm/
├── encoders/
├── memory/
├── networks/
│   ├── common.py
│   ├── fractional_activations.py
│   └── ...
├── policy/
├── util/
└── ...
```

---

# Notes

- Activation names must exactly match class names
- Activations are dynamically loaded through `common.py`
- Residual activations preserve baseline nonlinear behaviour
- Adaptive residual activations learn fractional strength during training
- Safe activations are specifically designed for RL stability

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
