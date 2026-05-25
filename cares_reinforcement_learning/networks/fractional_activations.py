import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math


# ============================================================
# Fractional ReLU positive branch only
# ============================================================

class FractionalReLUPositive(nn.Module):
    """
    Fractional ReLU with only the positive branch.

    Base activation:
        ReLU(x) = max(0, x)

    Fractional form:
        f(x) = x^(1-a), x > 0
        f(x) = 0,      x <= 0

    Notes:
    - This is a positive-only fractional rectifier.
    - No gamma normalization is used here.
    """

    def __init__(self, a=0.1, epsilon=1e-6, clip_value=1.0):
        super().__init__()
        self.a = a
        self.epsilon = epsilon
        self.clip_value = clip_value

    def forward(self, x):
        epsilon = 1e-6

        # Clamp for numerical stability before fractional power
        x_clamped = torch.clamp(x, min=epsilon)

        # Positive fractional power
        positive_part = torch.pow(x_clamped, 1 - self.a)

        # Negative side is zero, same as ReLU
        negative_part = 0

        output = torch.where(x > 0, positive_part, negative_part)

        return output


# ============================================================
# Fractional Parametric ReLU with learnable negative parameter
# ============================================================

class FPReLU2(nn.Module):
    """
    Fractional PReLU-style activation.

    Positive branch:
        x^(1-a) / Gamma(2-a)

    Negative branch:
        -p * (-x)^(1-a) / Gamma(2-a)

    Notes:
    - a is fixed.
    - p is learnable and forced positive through softplus.
    - This is suitable for your FPReLU experiments.
    """

    def __init__(self, a=0.2, parameter_init=1.5):
        super().__init__()

        # Fixed fractional order
        self.a = nn.Parameter(torch.tensor(a), requires_grad=False)

        # Learnable negative-side parameter
        self.p_raw = nn.Parameter(torch.tensor(parameter_init))

        # Gamma normalization
        self.g = float(np.math.gamma(2 - float(a)))

    def p(self):
        # Ensures p is positive and smooth
        return F.softplus(self.p_raw)

    def forward(self, x):
        # Positive side
        pos = (x > 0).float() * (x.clamp_min(1e-8) ** (1 - self.a)) / self.g

        # Negative side
        neg = (
            (x < 0).float()
            * (-(self.p()) / self.g)
            * ((-x).clamp_min(1e-8) ** (1 - self.a))
        )

        return pos + neg


# ============================================================
# Fractional Parametric ReLU with learnable slope k
# ============================================================

class FParReLU(nn.Module):
    """
    Fractional parametric ReLU.

    Positive branch:
        x^(1-a) / Gamma(2-a)

    Negative branch:
        -k * (-x)^(1-a) / Gamma(2-a)

    Notes:
    - a is fixed.
    - k is learnable.
    - k is clipped to [0.5, 3.0].
    """

    def __init__(self, a=0.1, k_init=1.5):
        super().__init__()

        # Fixed fractional order
        self.a = nn.Parameter(torch.tensor(a), requires_grad=False)

        # Learnable negative slope
        self.k = nn.Parameter(torch.tensor(k_init), requires_grad=True)

        # Gamma normalization factor
        self.g = np.math.gamma(2 - float(a))

    def forward(self, x):
        eps = 1e-8

        pos_mask = (x > 0).float()
        neg_mask = (x < 0).float()

        # Positive side
        positive_side = (
            pos_mask
            * (1.0 / self.g)
            * (torch.clamp(x, min=eps) ** (1 - self.a))
        )

        # Negative side with learnable clipped slope
        negative_side = neg_mask * (
            -(torch.clamp(self.k, 0.5, 3.0) / self.g)
            * (torch.clamp(-x, min=eps) ** (1 - self.a))
        )

        return positive_side + negative_side


# ============================================================
# Fractional Leaky ReLU with learnable small negative slope
# ============================================================

class FLReLU2(nn.Module):
    """
    Fractional Leaky ReLU with learnable negative slope.

    Positive branch:
        x^(1-a) / Gamma(2-a)

    Negative branch:
        -k * (-x)^(1-a) / Gamma(2-a)

    Notes:
    - a is fixed.
    - k is learnable.
    - k is clipped to [0.01, 0.3].
    """

    def __init__(self, a=0.1, k_init=0.1):
        super().__init__()

        # Fixed fractional order
        self.a = nn.Parameter(torch.tensor(a), requires_grad=False)

        # Learnable leaky slope
        self.k = nn.Parameter(torch.tensor(k_init), requires_grad=True)

        # Gamma normalization
        self.g = np.math.gamma(2 - float(a))

    def forward(self, x):
        pos_mask = (x > 0).float()
        neg_mask = (x < 0).float()

        # Positive side
        positive_side = (
            pos_mask
            * (1.0 / self.g)
            * (torch.clamp(x, min=1e-8) ** (1 - self.a))
        )

        # Negative side
        negative_side = neg_mask * (
            -(torch.clamp(self.k, 0.01, 0.3) / self.g)
            * (torch.clamp(-x, min=1e-8) ** (1 - self.a))
        )

        return positive_side + negative_side


# ============================================================
# Fractional Leaky ReLU
# ============================================================

class FractionaLLeakyReLU(nn.Module):
    """
    Fractional Leaky ReLU.

    Positive branch:
        x^(1-a) / Gamma(2-a)

    Negative branch:
        0.1 * |x|^(1-a) / Gamma(2-a)

    Notes:
    - This is one of your earlier versions.
    - Kept unchanged to preserve previous experiments.
    """

    def __init__(self, a=0.5, epsilon=1e-6, clip_value=1.0):
        super().__init__()
        self.a = a
        self.epsilon = epsilon
        self.clip_value = clip_value

    def forward(self, x):
        epsilon = 1e-6

        # Original version clamps x to positive values
        x_clamped = torch.clamp(x, min=epsilon)

        # Positive side
        positive_side = (1 / np.math.gamma(2 - self.a)) * torch.where(
            x_clamped > 0,
            x_clamped ** (1 - self.a),
            torch.tensor(self.epsilon, device=x.device),
        )

        # Negative side
        negative_side = (0.1 / np.math.gamma(2 - self.a)) * torch.where(
            x_clamped <= 0,
            torch.abs(x_clamped) ** (1 - self.a),
            torch.tensor(self.epsilon, device=x.device),
        )

        return positive_side + negative_side


# ============================================================
# Fractional Leaky ReLU with fixed k
# ============================================================

class FLReLU(nn.Module):
    """
    Fractional Leaky ReLU with fixed negative slope k.

    Positive branch:
        x^(1-a) / Gamma(2-a)

    Negative branch:
        -k * (-x)^(1-a) / Gamma(2-a)

    Notes:
    - k is fixed to 1.5.
    - Kept unchanged from your previous file.
    """

    def __init__(self, a=0.1, epsilon=1e-6, clip_value=1.0):
        super().__init__()
        self.a = a
        self.epsilon = epsilon
        self.clip_value = clip_value
        self.k = 1.5
        self.g = np.math.gamma(2 - a)

    def forward(self, x):
        # Positive side
        positive_side = torch.where(
            x > 0,
            (1.0 / self.g) * x ** (1 - self.a),
            (1.0 / self.g) * torch.full_like(x, self.epsilon) * (x == 0),
        )

        # Negative side
        negative_side = torch.where(
            x < 0,
            -(self.k / self.g) * (-x) ** (1 - self.a),
            -(self.k / self.g) * torch.full_like(x, self.epsilon) * (x == 0),
        )

        return positive_side + negative_side


# ============================================================
# Fractional PReLU
# ============================================================

class FractionalPReLU(nn.Module):
    """
    Fractional PReLU.

    Positive branch:
        x^(1-alpha) / Gamma(2-alpha)

    Negative branch:
        -a * (-x)^(1-alpha) / Gamma(2-alpha)

    Notes:
    - alpha is the fractional order.
    - self.a is the fixed negative-side slope.
    """

    def __init__(self, alpha=0.5, epsilon=1e-6):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.a = 1.5
        self.g = np.math.gamma(2 - alpha)

    def forward(self, x):
        # Positive side
        positive_side = torch.where(
            x > 0,
            (1.0 / self.g) * x ** (1 - self.alpha),
            (1.0 / self.g) * torch.full_like(x, self.epsilon) * (x == 0),
        )

        # Negative side
        negative_side = torch.where(
            x < 0,
            -(self.a / self.g) * (-x) ** (1 - self.alpha),
            -(self.a / self.g) * torch.full_like(x, self.epsilon) * (x == 0),
        )

        return positive_side + negative_side


# ============================================================
# ReLU power version
# ============================================================

class ReLU(nn.Module):
    """
    Power ReLU variant.

    Base activation:
        ReLU(x)

    Form:
        ReLU(x)^a

    Notes:
    - This prints the input in forward.
    - Keep the print only if you need debugging.
    """

    def __init__(self, a: float = 0.5):
        super().__init__()
        self.a = a
        self.counter = 0

    def forward(self, x):
        output = nn.functional.relu(x).pow(self.a)

        # Debug print from your original version
        print(f"input: {x}")

        return output


# ============================================================
# Fractional ReLU with torch Gamma
# ============================================================

class FractionalReLUGamma(nn.Module):
    """
    Fractional ReLU with Gamma normalization.

    Positive branch:
        x^(1-a) / Gamma(2-a)

    Negative branch:
        0

    Notes:
    - Uses torch.special.gamma.
    """

    def __init__(self, a: float = 0.5):
        super().__init__()
        self.a = a

    def forward(self, x):
        gamma_factor = 1 / torch.special.gamma(
            torch.tensor(2 - self.a, dtype=x.dtype, device=x.device)
        )

        return torch.where(
            x > 0,
            gamma_factor * torch.pow(torch.abs(x), 1 - self.a),
            torch.zeros_like(x),
        )


# ============================================================
# Fractional ReLU custom polynomial-style version
# ============================================================

class FractionalReLUCustom(nn.Module):
    """
    Custom fractional ReLU version.

    Notes:
    - Kept from your previous file.
    - This is not the standard Gamma-normalized fractional ReLU.
    """

    def __init__(self, a: float = 0.5):
        super().__init__()
        self.a = a

    def forward(self, x):
        return torch.where(
            x > 0,
            (2 - self.a) + torch.pow(x, 1 - self.a) * (self.a - 2) * (1 - self.a),
            torch.zeros_like(x),
        )


# ============================================================
# Fractional Tanh
# ============================================================

class FractionalTanh(nn.Module):
    """
    Fractional-order Tanh activation.

    Base activation:
        tanh(x)

    Fractional modulation:
        tanh(x) * |x|^(1-a)

    Notes:
    - epsilon avoids zero-power instability.
    """

    def __init__(self, a: float = 0.5):
        super().__init__()
        self.a = a

    def forward(self, x):
        return torch.tanh(x) * torch.pow(torch.abs(x) + 1e-6, 1 - self.a)


# ============================================================
# Helper for smooth fractional activations
# ============================================================

def _gamma_float(value: float) -> float:
    """
    Helper function for scalar Gamma computation.

    Used to precompute Gamma constants for finite
    Grünwald-Letnikov-style fractional sums.
    """
    return float(math.gamma(value))


# ============================================================
# Fractional Swish / SiLU
# ============================================================

class FractionalSwish(nn.Module):
    """
    Fractional version of Swish / SiLU.

    Base activation:
        Swish(x) = x * sigmoid(x)

    Fractional formulation:
        f(x) = Swish(x) + alpha * sigmoid(x) * (1 - Swish(x))

    Notes:
    - Smooth non-monotonic activation.
    - Fractional correction controlled by alpha.
    - Recommended alpha range: [0, 1].
    - Closest non-fractional baseline: Swish / SiLU.
    """

    def __init__(self, a=0.1):
        super().__init__()
        self.a = a

    def forward(self, x):
        alpha = torch.clamp(
            torch.tensor(self.a, device=x.device, dtype=x.dtype),
            0.0,
            1.0,
        )

        sig = torch.sigmoid(x)

        # Standard Swish / SiLU
        swish = x * sig

        # Fractional correction
        output = swish + alpha * sig * (1.0 - swish)

        return output


# ============================================================
# Fractional Swish with learnable beta
# ============================================================

class FractionalSwishBeta(nn.Module):
    """
    Fractional Swish with learnable beta.

    Base activation:
        Swish_beta(x) = x * sigmoid(beta * x)

    Fractional formulation:
        f(x) = Swish_beta(x)
             + alpha * sigmoid(beta*x) * (1 - Swish_beta(x))

    Notes:
    - beta controls smoothness and gating strength.
    - alpha controls the fractional correction.
    - alpha is fixed.
    - beta is learnable.
    """

    def __init__(self, a=0.1, beta_init=1.0):
        super().__init__()

        self.a = a

        # Learnable beta parameter
        self.beta_raw = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, x):
        alpha = torch.clamp(
            torch.tensor(self.a, device=x.device, dtype=x.dtype),
            0.0,
            1.0,
        )

        # Softplus keeps beta positive
        beta = torch.clamp(F.softplus(self.beta_raw), 0.1, 10.0)

        sig = torch.sigmoid(beta * x)

        # Beta-Swish
        swish = x * sig

        # Fractional correction
        output = swish + alpha * sig * (1.0 - swish)

        return output


# ============================================================
# Fractional Adaptive Linear Unit
# ============================================================

class FALU(nn.Module):
    """
    Fractional Adaptive Linear Unit.

    Interpretation:
        Generalized adaptive fractional Swish-family activation.

    Core gated structure:
        g(x) = x * sigmoid(beta*x)

    For alpha in [0, 1]:
        f(x) = g(x) + alpha * sigmoid(beta*x) * (1 - g(x))

    For alpha in [1, 2]:
        a higher-order adaptive correction is applied.

    Notes:
    - There is no standard non-fractional FALU baseline.
    - Closest non-fractional baseline: Swish / SiLU.
    - beta is learnable.
    - alpha is fixed here for fair comparison with your fixed-alpha
      fractional rectifier activations.
    """

    def __init__(self, a=0.1, beta_init=1.0):
        super().__init__()

        self.a = a

        # Learnable beta parameter
        self.beta_raw = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, x):
        alpha = torch.clamp(
            torch.tensor(self.a, device=x.device, dtype=x.dtype),
            0.0,
            2.0,
        )

        beta = torch.clamp(F.softplus(self.beta_raw), 0.1, 10.0)

        # Swish-like gated structure
        sig_beta = torch.sigmoid(beta * x)
        g = x * sig_beta

        # Fractional region alpha in [0, 1]
        if alpha.item() <= 1.0:
            output = g + alpha * sig_beta * (1.0 - g)
            return output

        # Higher-order adaptive region alpha in [1, 2]
        sig = torch.sigmoid(x)
        h = g + sig * (1.0 - g)

        output = h + (alpha - 1.0) * sig_beta * (1.0 - 2.0 * h)

        return output


# ============================================================
# Fractional GELU
# ============================================================

class FractionalGELU(nn.Module):
    """
    Grünwald-Letnikov-style fractional GELU.

    Base activation:
        GELU(x)

    Tanh approximation used:
        GELU(x) ≈ 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))

    Fractional formulation:
        finite Grünwald-Letnikov fractional sum over shifted GELU terms.

    Notes:
    - Closest non-fractional baseline: GELU.
    - alpha range: [0, 2].
    - h controls finite-difference step size.
    - n_iter controls approximation depth.
    """

    def __init__(self, a=0.1, h=0.5, n_iter=3):
        super().__init__()

        self.a = a
        self.h = h
        self.n_iter = n_iter

        # Use register_buffer so this moves correctly with .to(device)
        self.register_buffer(
            "gamma_consts",
            torch.tensor(
                [_gamma_float(i + 1) for i in range(n_iter)],
                dtype=torch.float32,
            ),
        )

    def _gelu_tanh_part(self, z):
        """
        Returns z * (1 + tanh(...)).
        The final 0.5 factor is applied outside the summation.
        """
        sqrt_2_over_pi = 0.7978845608028654

        return z * (
            1.0
            + torch.tanh(
                sqrt_2_over_pi * (z + 0.044715 * z ** 3)
            )
        )

    def forward(self, x):
        alpha = torch.clamp(
            torch.tensor(self.a, device=x.device, dtype=x.dtype),
            0.0,
            2.0,
        )

        h = torch.tensor(self.h, device=x.device, dtype=x.dtype)

        gamma_a_plus_one = torch.exp(torch.lgamma(1.0 + alpha))

        gamma_consts = self.gamma_consts.to(
            device=x.device,
            dtype=x.dtype,
        )

        out = torch.zeros_like(x)

        # Finite Grünwald-Letnikov fractional summation
        for i in range(self.n_iter):
            z = x - i * h

            denom = gamma_consts[i] * torch.exp(
                torch.lgamma(1.0 - i + alpha)
            )

            coeff = gamma_a_plus_one / (denom + 1e-8)

            sign = -1.0 if i % 2 == 1 else 1.0

            out = out + sign * coeff * self._gelu_tanh_part(z)

        # Apply h^alpha scaling and GELU 0.5 factor
        output = out / ((h ** alpha) * 2.0)

        return output


# ============================================================
# Fractional Mish
# ============================================================

class FractionalMish(nn.Module):
    """
    Grünwald-Letnikov-style fractional Mish.

    Base activation:
        Mish(x) = x * tanh(softplus(x))

    Fractional formulation:
        finite Grünwald-Letnikov fractional sum over shifted Mish terms.

    Notes:
    - Closest non-fractional baseline: Mish.
    - alpha range: [0, 2].
    - h controls finite-difference step size.
    - n_iter controls approximation depth.
    """

    def __init__(self, a=0.1, h=0.5, n_iter=3):
        super().__init__()

        self.a = a
        self.h = h
        self.n_iter = n_iter

        # Use register_buffer so this moves correctly with .to(device)
        self.register_buffer(
            "gamma_consts",
            torch.tensor(
                [_gamma_float(i + 1) for i in range(n_iter)],
                dtype=torch.float32,
            ),
        )

    def _mish(self, z):
        """
        Standard Mish activation.
        """
        return z * torch.tanh(F.softplus(z))

    def forward(self, x):
        alpha = torch.clamp(
            torch.tensor(self.a, device=x.device, dtype=x.dtype),
            0.0,
            2.0,
        )

        h = torch.tensor(self.h, device=x.device, dtype=x.dtype)

        gamma_a_plus_one = torch.exp(torch.lgamma(1.0 + alpha))

        gamma_consts = self.gamma_consts.to(
            device=x.device,
            dtype=x.dtype,
        )

        out = torch.zeros_like(x)

        # Finite Grünwald-Letnikov fractional summation
        for i in range(self.n_iter):
            z = x - i * h

            denom = gamma_consts[i] * torch.exp(
                torch.lgamma(1.0 - i + alpha)
            )

            coeff = gamma_a_plus_one / (denom + 1e-8)

            sign = -1.0 if i % 2 == 1 else 1.0

            out = out + sign * coeff * self._mish(z)

        # Apply h^alpha scaling
        output = out / (h ** alpha)

        return output


# ============================================================
# Safe RL-friendly Fractional Swish / SiLU
# ============================================================

class SafeFractionalSwish(nn.Module):
    """
    Safe fractional Swish / SiLU for RL.

    Base activation:
        Swish(x) = x * sigmoid(x)

    Fractional formulation:
        f(x) = Swish(x) + alpha * sigmoid(x) * (1 - Swish(x))

    Notes:
    - Same formulation as FractionalSwish.
    - Adds nan_to_num protection.
    - Does not use fractional powers, so negative inputs are safe.
    - Recommended alpha range: [0, 1].
    """

    def __init__(self, a=0.1):
        super().__init__()
        self.a = a

    def forward(self, x):
        alpha = torch.clamp(
            torch.tensor(self.a, device=x.device, dtype=x.dtype),
            0.0,
            1.0,
        )

        sig = torch.sigmoid(x)
        swish = x * sig

        output = swish + alpha * sig * (1.0 - swish)

        return torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)


# ============================================================
# Safe RL-friendly Fractional Swish with learnable beta
# ============================================================

class SafeFractionalSwishBeta(nn.Module):
    """
    Safe fractional Swish with learnable beta for RL.

    Base activation:
        Swish_beta(x) = x * sigmoid(beta*x)

    Notes:
    - Same idea as FractionalSwishBeta.
    - Adds nan_to_num protection.
    - beta is learnable and clipped for stability.
    """

    def __init__(self, a=0.1, beta_init=1.0):
        super().__init__()
        self.a = a
        self.beta_raw = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, x):
        alpha = torch.clamp(
            torch.tensor(self.a, device=x.device, dtype=x.dtype),
            0.0,
            1.0,
        )

        beta = torch.clamp(F.softplus(self.beta_raw), 0.1, 10.0)

        sig = torch.sigmoid(beta * x)
        swish = x * sig

        output = swish + alpha * sig * (1.0 - swish)

        return torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)


# ============================================================
# Safe RL-friendly FALU
# ============================================================

class SafeFALU(nn.Module):
    """
    Safe Fractional Adaptive Linear Unit for RL.

    Core gated structure:
        g(x) = x * sigmoid(beta*x)

    Notes:
    - Same structure as FALU.
    - Adds nan_to_num protection.
    - beta is learnable and clipped for stability.
    - Closest non-fractional baseline: Swish / SiLU.
    """

    def __init__(self, a=0.1, beta_init=1.0):
        super().__init__()

        self.a = a
        self.beta_raw = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, x):
        alpha = torch.clamp(
            torch.tensor(self.a, device=x.device, dtype=x.dtype),
            0.0,
            2.0,
        )

        beta = torch.clamp(F.softplus(self.beta_raw), 0.1, 10.0)

        sig_beta = torch.sigmoid(beta * x)
        g = x * sig_beta

        if alpha.item() <= 1.0:
            output = g + alpha * sig_beta * (1.0 - g)
        else:
            sig = torch.sigmoid(x)
            h = g + sig * (1.0 - g)
            output = h + (alpha - 1.0) * sig_beta * (1.0 - 2.0 * h)

        return torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)


# ============================================================
# Safe RL-friendly Fractional GELU-style activation
# ============================================================

class SafeFractionalGELU(nn.Module):
    """
    Safe fractional GELU-style activation for RL.

    Base activation:
        GELU(x)

    Stable fractional-style modulation:
        f(x) = GELU(x) * (|x| + epsilon)^(1-a) / Gamma(2-a)

    Notes:
    - This is not the same as the GL-style FractionalGELU above.
    - It is closer to your rectifier fractional design.
    - It avoids gamma singularities from the GL finite-sum form.
    - It avoids fractional powers of negative values by using |x|.
    - Good for RL stability testing.
    """

    def __init__(self, a=0.1, epsilon=1e-6):
        super().__init__()
        self.a = a
        self.epsilon = epsilon
        self.g = np.math.gamma(2 - a)

    def forward(self, x):
        gelu = F.gelu(x)

        # Safe magnitude-based fractional scaling
        scale = torch.pow(torch.abs(x) + self.epsilon, 1.0 - self.a)

        output = (gelu * scale) / self.g

        return torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)


# ============================================================
# Safe RL-friendly Fractional Mish-style activation
# ============================================================

class SafeFractionalMish(nn.Module):
    """
    Safe fractional Mish-style activation for RL.

    Base activation:
        Mish(x) = x * tanh(softplus(x))

    Stable fractional-style modulation:
        f(x) = Mish(x) * (|x| + epsilon)^(1-a) / Gamma(2-a)

    Notes:
    - This is not the same as the GL-style FractionalMish above.
    - It is closer to your rectifier fractional design.
    - It avoids gamma singularities from the GL finite-sum form.
    - It avoids fractional powers of negative values by using |x|.
    - Good for RL stability testing.
    """

    def __init__(self, a=0.1, epsilon=1e-6):
        super().__init__()
        self.a = a
        self.epsilon = epsilon
        self.g = np.math.gamma(2 - a)

    def forward(self, x):
        mish = x * torch.tanh(F.softplus(x))

        # Safe magnitude-based fractional scaling
        scale = torch.pow(torch.abs(x) + self.epsilon, 1.0 - self.a)

        output = (mish * scale) / self.g

        return torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)


# ============================================================
# Safe GL-style Fractional GELU
# ============================================================

class SafeGLFractionalGELU(nn.Module):
    """
    Safe version of the GL-style FractionalGELU.

    Notes:
    - Keeps the finite Grünwald-Letnikov fractional sum.
    - Adds nan_to_num protection.
    - Clamps alpha to avoid exact singular edge cases.
    - Use this if you want to stay closer to the external repo formulation.
    """

    def __init__(self, a=0.1, h=0.5, n_iter=3):
        super().__init__()

        self.a = a
        self.h = h
        self.n_iter = n_iter

        self.register_buffer(
            "gamma_consts",
            torch.tensor(
                [_gamma_float(i + 1) for i in range(n_iter)],
                dtype=torch.float32,
            ),
        )

    def _gelu_tanh_part(self, z):
        sqrt_2_over_pi = 0.7978845608028654

        return z * (
            1.0
            + torch.tanh(
                sqrt_2_over_pi * (z + 0.044715 * z ** 3)
            )
        )

    def forward(self, x):
        # Avoid exact 0 and 2 edge cases
        alpha = torch.clamp(
            torch.tensor(self.a, device=x.device, dtype=x.dtype),
            1e-4,
            2.0 - 1e-4,
        )

        h = torch.tensor(self.h, device=x.device, dtype=x.dtype)

        gamma_a_plus_one = torch.exp(torch.lgamma(1.0 + alpha))

        gamma_consts = self.gamma_consts.to(
            device=x.device,
            dtype=x.dtype,
        )

        out = torch.zeros_like(x)

        for i in range(self.n_iter):
            z = x - i * h

            denom = gamma_consts[i] * torch.exp(
                torch.lgamma(1.0 - i + alpha)
            )

            coeff = gamma_a_plus_one / (denom + 1e-8)

            sign = -1.0 if i % 2 == 1 else 1.0

            out = out + sign * coeff * self._gelu_tanh_part(z)

        output = out / ((h ** alpha) * 2.0)

        return torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)


# ============================================================
# Safe GL-style Fractional Mish
# ============================================================

class SafeGLFractionalMish(nn.Module):
    """
    Safe version of the GL-style FractionalMish.

    Notes:
    - Keeps the finite Grünwald-Letnikov fractional sum.
    - Adds nan_to_num protection.
    - Clamps alpha to avoid exact singular edge cases.
    - Use this if you want to stay closer to the external repo formulation.
    """

    def __init__(self, a=0.1, h=0.5, n_iter=3):
        super().__init__()

        self.a = a
        self.h = h
        self.n_iter = n_iter

        self.register_buffer(
            "gamma_consts",
            torch.tensor(
                [_gamma_float(i + 1) for i in range(n_iter)],
                dtype=torch.float32,
            ),
        )

    def _mish(self, z):
        return z * torch.tanh(F.softplus(z))

    def forward(self, x):
        # Avoid exact 0 and 2 edge cases
        alpha = torch.clamp(
            torch.tensor(self.a, device=x.device, dtype=x.dtype),
            1e-4,
            2.0 - 1e-4,
        )

        h = torch.tensor(self.h, device=x.device, dtype=x.dtype)

        gamma_a_plus_one = torch.exp(torch.lgamma(1.0 + alpha))

        gamma_consts = self.gamma_consts.to(
            device=x.device,
            dtype=x.dtype,
        )

        out = torch.zeros_like(x)

        for i in range(self.n_iter):
            z = x - i * h

            denom = gamma_consts[i] * torch.exp(
                torch.lgamma(1.0 - i + alpha)
            )

            coeff = gamma_a_plus_one / (denom + 1e-8)

            sign = -1.0 if i % 2 == 1 else 1.0

            out = out + sign * coeff * self._mish(z)

        output = out / (h ** alpha)

        return torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)

# ============================================================
# Residual Fractional GELU
# ============================================================

class ResidualFractionalGELU(nn.Module):

    def __init__(self, a=0.1, beta=0.05, epsilon=1e-6, clip_value=5.0):
        super().__init__()

        self.a = a
        self.beta = beta
        self.epsilon = epsilon
        self.clip_value = clip_value
        self.g = np.math.gamma(2 - a)

    def forward(self, x):
        gelu = F.gelu(x)

        frac = (
            torch.sign(x)
            * torch.pow(torch.abs(x) + self.epsilon, 1.0 - self.a)
            / self.g
        )

        frac = torch.clamp(frac, -self.clip_value, self.clip_value)

        output = gelu + self.beta * frac

        return torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)


# ============================================================
# Adaptive Residual Fractional GELU
# ============================================================

class AdaptiveResidualFractionalGELU(nn.Module):

    def __init__(self, a=0.1, beta_init=0.05, epsilon=1e-6, clip_value=5.0):
        super().__init__()

        self.a = a
        self.epsilon = epsilon
        self.clip_value = clip_value
        self.g = np.math.gamma(2 - a)

        self.beta_raw = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, x):
        gelu = F.gelu(x)

        frac = (
            torch.sign(x)
            * torch.pow(torch.abs(x) + self.epsilon, 1.0 - self.a)
            / self.g
        )

        frac = torch.clamp(frac, -self.clip_value, self.clip_value)

        beta = torch.tanh(self.beta_raw)

        output = gelu + beta * frac

        return torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)

