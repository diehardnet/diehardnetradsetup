import time

import torch


class Timer:
    time_measure = 0

    def tic(self): self.time_measure = time.time()

    def toc(self): self.time_measure = time.time() - self.time_measure

    @property
    def diff_time(self): return self.time_measure

    @property
    def diff_time_str(self): return str(self)

    def __str__(self): return f"{self.time_measure:.4f}s"

    def __repr__(self): return str(self)


def equal(rhs: torch.Tensor, lhs: torch.Tensor, threshold: float = 0) -> bool:
    """ Compare based or not in a threshold, if threshold is none then it is equal comparison    """
    if threshold > 0:
        return bool(torch.all(torch.le(torch.abs(torch.subtract(rhs, lhs)), threshold)))
    else:
        return bool(torch.equal(rhs, lhs))
