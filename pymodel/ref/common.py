"""OP-03 and OP-04: Vision LayerNorm and same-shape residual addition."""

import torch
import torch.nn.functional as F


def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Last-axis affine LayerNorm, matching the pinned nn.LayerNorm path."""
    if x.ndim < 1 or weight.shape != (x.shape[-1],) or bias.shape != weight.shape:
        raise ValueError("LayerNorm weight/bias must match the last input dimension")
    if eps <= 0:
        raise ValueError("eps must be positive")
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


def residual_add(residual: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
    """Add without broadcasting, implicit dtype promotion or input mutation."""
    if residual.shape != branch.shape:
        raise ValueError("Residual inputs must have the same shape")
    if residual.dtype != branch.dtype or residual.device != branch.device:
        raise ValueError("Residual inputs must have the same dtype and device")
    return residual + branch
