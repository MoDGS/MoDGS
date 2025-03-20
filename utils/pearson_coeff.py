import math
from typing import Tuple

import torch
from torch import Tensor

from torchmetrics.functional.regression.utils import _check_data_shape_to_num_outputs
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.checks import _check_same_shape
def _check_same_shape(preds: Tensor, target: Tensor) -> None:
    """Check that predictions and target have the same shape, else raise error."""
    if preds.shape != target.shape:
        raise RuntimeError(
            f"Predictions and targets are expected to have the same shape, but got {preds.shape} and {target.shape}."
        )
def _check_data_shape_to_num_outputs(
    preds: Tensor, target: Tensor, num_outputs: int, allow_1d_reshape: bool = False
) -> None:
    """Check that predictions and target have the correct shape, else raise error.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        num_outputs: Number of outputs in multioutput setting
        allow_1d_reshape: Allow that for num_outputs=1 that preds and target does not need to be 1d tensors. Instead
            code that follows are expected to reshape the tensors to 1d.

    """
    if preds.ndim > 2 or target.ndim > 2:
        raise ValueError(
            f"Expected both predictions and target to be either 1- or 2-dimensional tensors,"
            f" but got {target.ndim} and {preds.ndim}."
        )
    cond1 = False
    if not allow_1d_reshape:
        cond1 = num_outputs == 1 and not (preds.ndim == 1 or preds.shape[1] == 1)
    cond2 = num_outputs > 1 and preds.ndim > 1 and num_outputs != preds.shape[1]
    if cond1 or cond2:
        raise ValueError(
            f"Expected argument `num_outputs` to match the second dimension of input, but got {num_outputs}"
            f" and {preds.shape[1]}."
        )

def pearson_corrcoef(preds: Tensor, target: Tensor) -> Tensor:
    """Compute pearson correlation coefficient.

    Args:
        preds: estimated scores
        target: ground truth scores

    Example (single output regression):
        >>> from torchmetrics.functional.regression import pearson_corrcoef
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> pearson_corrcoef(preds, target)
        tensor(0.9849)

    Example (multi output regression):
        >>> from torchmetrics.functional.regression import pearson_corrcoef
        >>> target = torch.tensor([[3, -0.5], [2, 7]])
        >>> preds = torch.tensor([[2.5, 0.0], [2, 8]])
        >>> pearson_corrcoef(preds, target)
        tensor([1., 1.])

    """
    d = preds.shape[1] if preds.ndim == 2 else 1
    _temp = torch.zeros(d, dtype=preds.dtype, device=preds.device)
    mean_x, mean_y, var_x = _temp.clone(), _temp.clone(), _temp.clone()
    var_y, corr_xy, nb = _temp.clone(), _temp.clone(), _temp.clone()
    _, _, var_x, var_y, corr_xy, nb = _pearson_corrcoef_update(
        preds, target, mean_x, mean_y, var_x, var_y, corr_xy, nb, num_outputs=1 if preds.ndim == 1 else preds.shape[-1]
    )
    return _pearson_corrcoef_compute(var_x, var_y, corr_xy, nb)
def _pearson_corrcoef_compute(
    var_x: Tensor,
    var_y: Tensor,
    corr_xy: Tensor,
    nb: Tensor,
) -> Tensor:
    """Compute the final pearson correlation based on accumulated statistics.

    Args:
        var_x: variance estimate of x tensor
        var_y: variance estimate of y tensor
        corr_xy: covariance estimate between x and y tensor
        nb: number of observations

    """
    var_x /= nb - 1
    var_y /= nb - 1
    corr_xy /= nb - 1
    # if var_x, var_y is float16 and on cpu, make it bfloat16 as sqrt is not supported for float16
    # on cpu, remove this after https://github.com/pytorch/pytorch/issues/54774 is fixed
    if var_x.dtype == torch.float16 and var_x.device == torch.device("cpu"):
        var_x = var_x.bfloat16()
        var_y = var_y.bfloat16()

    bound = math.sqrt(torch.finfo(var_x.dtype).eps)
    if (var_x < bound).any() or (var_y < bound).any():
        rank_zero_warn(
            "The variance of predictions or target is close to zero. This can cause instability in Pearson correlation"
            "coefficient, leading to wrong results. Consider re-scaling the input if possible or computing using a"
            f"larger dtype (currently using {var_x.dtype}).",
            UserWarning,
        )

    corrcoef = (corr_xy / (var_x * var_y).sqrt()).squeeze()
    return torch.clamp(corrcoef, -1.0, 1.0)
def _pearson_corrcoef_update(
    preds: Tensor,
    target: Tensor,
    mean_x: Tensor,
    mean_y: Tensor,
    var_x: Tensor,
    var_y: Tensor,
    corr_xy: Tensor,
    num_prior: Tensor,
    num_outputs: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Update and returns variables required to compute Pearson Correlation Coefficient.

    Check for same shape of input tensors.

    Args:
        preds: estimated scores
        target: ground truth scores
        mean_x: current mean estimate of x tensor
        mean_y: current mean estimate of y tensor
        var_x: current variance estimate of x tensor
        var_y: current variance estimate of y tensor
        corr_xy: current covariance estimate between x and y tensor
        num_prior: current number of observed observations
        num_outputs: Number of outputs in multioutput setting

    """
    # Data checking
    _check_same_shape(preds, target)
    _check_data_shape_to_num_outputs(preds, target, num_outputs)
    num_obs = preds.shape[0]
    cond = num_prior.mean() > 0 or num_obs == 1

    if cond:
        mx_new = (num_prior * mean_x + preds.sum(0)) / (num_prior + num_obs)
        my_new = (num_prior * mean_y + target.sum(0)) / (num_prior + num_obs)
    else:
        mx_new = preds.mean(0)
        my_new = target.mean(0)

    num_prior += num_obs

    if cond:
        var_x += ((preds - mx_new) * (preds - mean_x)).sum(0)
        var_y += ((target - my_new) * (target - mean_y)).sum(0)
    else:
        var_x += preds.var(0) * (num_obs - 1)
        var_y += target.var(0) * (num_obs - 1)
    corr_xy += ((preds - mx_new) * (target - mean_y)).sum(0)
    mean_x = mx_new
    mean_y = my_new

    return mean_x, mean_y, var_x, var_y, corr_xy, num_prior