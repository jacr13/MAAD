from typing import Any, Dict, Optional, Union

import numpy as np
import scipy.signal
import torch


def is_number(number: Any) -> bool:
    if type(number) in [int, float, np.int64, np.float64]:
        return True
    return False


def discount_cumsum(x, gamma):
    """Compute discounted sum along the 0-th dimension of the `x` ndarray
    Return an ndarray `y` with the same shape as x, satisfying:
        y[t] = x[t] + gamma * x[t+1] + gamma^2 * x[t+2] + ... + gamma^k * x[t+k],
            where k = len(x) - t - 1

    Args:
        x (np.ndarray): 2-D array of floats, time x features
        gamma (float): Discount factor
    """
    assert x.ndim >= 1
    return scipy.signal.lfilter([1], [1, float(-gamma)], x[::-1], axis=0)[::-1]


def standardized(array):
    """Normalize the values in an array.

    Args:
        array (np.ndarray): Array of values to normalize.

    Returns:
        array with zero mean and unit standard deviation.
    """
    return (array - array.mean()) / max(1e-4, array.std())


def gradients_norm(
    model: torch.nn.Module,
    norm_type: Union[float, int, str] = 2,
    reduction: Optional[str] = "norm",
    precision: int = 4,
) -> Union[Dict[str, float], float]:
    if norm_type not in ["fro", "nuc"]:
        norm_type = float(norm_type)

    norms = {
        f"grad_{norm_type}_norm/{name}": p.grad.detach().data.norm(norm_type)
        for name, p in model.named_parameters()
        if p.grad is not None
    }

    if norms:
        tensor_norms = torch.tensor(list(norms.values()))
        if reduction == "mean":
            return round(tensor_norms.mean().item(), precision)
        elif reduction == "sum":
            return round(tensor_norms.sum().item(), precision)
        elif reduction == "norm":
            return round(tensor_norms.norm(norm_type).item(), precision)
        elif reduction == "none" or reduction is None:
            norms[f"grad_{norm_type}_norm/total"] = round(
                tensor_norms.norm(norm_type).item(), precision
            )
            norms = {k: round(v.item(), precision) for k, v in norms.items()}
            return norms
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
    return {}


def huber_quant_reg_loss(td_errors, quantile, kappa=1.0):
    """Huber regression loss (introduced in 1964) following the definition
    in section 2.3 in the IQN paper (https://arxiv.org/abs/1806.06923).
    The loss involves a disjunction of 2 cases:
        case one: |td_errors| <= kappa
        case two: |td_errors| > kappa
    """
    aux = (
        0.5 * td_errors**2 * (torch.abs(td_errors) <= kappa).float()
        + kappa
        * (torch.abs(td_errors) - (0.5 * kappa))
        * (torch.abs(td_errors) > kappa).float()
    )
    return torch.abs(quantile - ((td_errors.le(0.0)).float())) * aux / kappa


class LRScheduler(object):
    def __init__(
        self,
        optimizer,
        initial_lr,
        lr_schedule,
        total_num_steps,
        kwargs=None,
    ):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.lr_schedule = lr_schedule
        self.total_num_steps = total_num_steps

    def step(self, steps_so_far):
        """Return the next lr in accordance with the initial lr,
        the desired schedule, and the learning progress.
        """
        if self.lr_schedule == "linear":
            next_lr = self.initial_lr * max(
                1.0 - (float(steps_so_far) / float(self.total_num_steps)),
                0.0,
            )
        elif self.lr_schedule == "constant":
            next_lr = self.initial_lr * 1.0
        else:
            raise NotImplementedError("invalid lr schedule.")
        for g in self.optimizer.param_groups:
            g["lr"] = next_lr
        return next_lr


# Set up the learning rate schedule
# LambdaLR will raise automatically an unavoidable warning to save also the optimizer
class LRDecay(object):
    def __init__(self, hps, lr):
        self.hps = hps
        self.lr_init = lr

    def __call__(self, t):
        if self.hps.lr_schedule == "constant":
            return 1
        elif self.hps.lr_schedule == "linear":
            return max(
                1.0 - ((t - 1.0) / (self.hps.num_timesteps // self.hps.rollout_len)),
                0.0,
            )
        elif self.hps.lr_schedule == "step":
            drop = 0.5  # drop factor
            epochs_drop = 100  # drops every epochs_drop
            return np.power(drop, np.floor((1 + t) / epochs_drop))
        else:
            raise ValueError(f"lr_schedule {self.hps.lr_schedule} is not supported.")
        return 1
