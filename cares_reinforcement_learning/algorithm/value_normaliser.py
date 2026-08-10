import torch


class ValueNormaliser:
    """
    Running return/value normalization for PPO-style critic targets.

    The critic predicts normalized values while GAE and logging operate
    in the original return scale.

    MAPPO-style value normalization is especially important for
    stabilizing critic learning in cooperative MARL tasks.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        epsilon: float = 1e-4,
        device: torch.device | None = None,
    ):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = torch.tensor(epsilon, device=device)

    def update(self, returns: torch.Tensor) -> None:
        returns = returns.detach()
        batch_mean = returns.mean(dim=0)
        batch_var = returns.var(dim=0, unbiased=False)
        batch_count = torch.tensor(
            returns.shape[0], device=returns.device, dtype=torch.float32
        )

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = total_count

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var + 1e-8)

    def normalise(self, returns: torch.Tensor) -> torch.Tensor:
        return (returns - self.mean) / self.std

    def denormalise(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std + self.mean
