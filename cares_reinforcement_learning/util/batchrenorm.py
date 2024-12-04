import torch

# from https://github.com/ludvb/batchrenorm

__all__ = ["BatchRenorm1d", "BatchRenorm2d", "BatchRenorm3d"]


class BatchRenorm(torch.jit.ScriptModule):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-3,
        momentum: float = 0.01,
        affine: bool = True,
        warmup_steps: int = 10000,
    ):
        super().__init__()
        self.register_buffer(
            "running_mean", torch.zeros(num_features, dtype=torch.float)
        )
        self.register_buffer("running_var", torch.ones(num_features, dtype=torch.float))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        self.weight = torch.nn.Parameter(torch.ones(num_features, dtype=torch.float))
        self.bias = torch.nn.Parameter(torch.zeros(num_features, dtype=torch.float))
        self.affine = affine
        self.eps = eps
        self.step = 0
        self.momentum = momentum
        self.warmup_steps = warmup_steps

    def _check_input_dim(self, x: torch.Tensor) -> None:
        raise NotImplementedError()  # pragma: no cover

    @property
    def rmax(self) -> torch.Tensor:
        """
        Scales standard deviation
        """
        return (2 / 35000 * self.num_batches_tracked + 25 / 35).clamp_(1.0, 3.0)

    @property
    def dmax(self) -> torch.Tensor:
        """
        Scales mean
        """
        return (5 / 20000 * self.num_batches_tracked - 25 / 20).clamp_(0.0, 5.0)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Mask is a boolean tensor used for indexing, where True values are padded
        i.e for 3D input, mask should be of shape (batch_size, seq_len)
        mask is used to prevent padded values from affecting the batch statistics
        """
        self._check_input_dim(x)
        if x.dim() > 2:
            x = x.transpose(1, -1)
        if self.training:
            # x=r(x^−μ)/σ+d # changing input with running mean, std, dynamic upper limit r, dynamic shift limit d
            # μ, σ, r, d updated as:
            # -> μ = μ + momentum * (input.mean(0))
            # -> σ = σ + momentum * (input.std(0) + eps)
            # -> r = clip(input.std(0)/σ, !/rmax, rmax)
            # -> d = clip((input.mean(0) - μ)/σ, -dmax, dmax)
            # Also: optional masking
            # Also: counter "num_batches_tracked"
            # Note: The introduction of r and d mitigates some of the issues of BN, especially with small BZ or significant shifts in the input distribution.
            dims = [i for i in range(x.dim() - 1)]
            if mask is not None:
                z = x[~mask]
                batch_mean = z.mean(0)
                batch_var = z.var(0, unbiased=False)
            else:
                batch_mean = x.mean(dims)
                batch_var = x.var(dims, unbiased=False)

            # Adding warm up
            warmed_up_factor = (self.num_batches_tracked >= self.warmup_steps).float()

            running_std = torch.sqrt(self.running_var.view_as(batch_var) + self.eps)
            r = ((batch_var / running_std).clamp_(1 / self.rmax, self.rmax)).detach()
            d = (
                (
                    (batch_mean - self.running_mean.view_as(batch_mean)) / running_std
                ).clamp_(-self.dmax, self.dmax)
            ).detach()
            if warmed_up_factor:
                x = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
            else:
                x = r * ((x - batch_mean) / torch.sqrt(batch_var + self.eps)) + d
            # Pytorch convention (1-beta)*estimated + beta*observed
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * batch_mean.detach()
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * batch_var.detach()
            self.num_batches_tracked += 1
        else:  # x=r(x^−μpop​ )/σpop​ +d # running mean and std
            x = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        if self.affine:  # Step 3 affine transform: y=γx+β
            x = self.weight * x + self.bias
        if x.dim() > 2:
            x = x.transpose(1, -1)
        return x


class BatchRenorm1d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() not in [2, 3]:
            raise ValueError("expected 2D or 3D input (got {x.dim()}D input)")


class BatchRenorm2d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError("expected 4D input (got {x.dim()}D input)")


class BatchRenorm3d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 5:
            raise ValueError("expected 5D input (got {x.dim()}D input)")
