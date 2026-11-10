"""BF16 Vision operator references, using the pinned eager attention path.

Source: Transformers f90e93c4f839, modeling_qwen3_5_moe.py and vision_utils.py.
Processor Patch order and position metadata follow the IF-04 contract.
"""

import torch
import torch.nn.functional as F

from pymodel.ref.common import layer_norm


def vision_patch_embedding(patches: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """OP-08: flattened processor patches [N,C*T*P*P] -> [N,D]."""
    if weight.ndim != 5:
        raise ValueError("Patch weight must have shape [D,C,T,P,P]")
    width = weight[0].numel()
    if patches.ndim != 2 or patches.shape[1] != width:
        raise ValueError("Each processor patch must contain C*T*P*P values")
    shape = (patches.shape[0], *weight.shape[1:])
    out = F.conv3d(patches.reshape(shape).to(weight.dtype), weight, bias, stride=weight.shape[2:])
    return out.reshape(patches.shape[0], weight.shape[0])


def vision_position_interpolation(grid_thw: torch.Tensor, table_side: int = 48,
                                   spatial_merge_size: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate int64 indices [N,4] and FP32 bilinear weights [N,4].

    grid_thw [M,3] contains temporal/height/width Patch counts per image/video;
    N=sum(t*h*w). Output rows follow the processor's spatial-merge block order,
    repeated for each temporal slice. The model uses align_corners=True and
    clamps neighbours to the learned table's border. Results stay on grid_thw.device.
    """
    if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
        raise ValueError("grid_thw must have shape [M,3] with temporal/height/width Patch counts")
    if grid_thw.dtype not in (torch.int32, torch.int64):
        raise TypeError("grid_thw must contain int32 or int64 Patch counts")
    if type(table_side) is not int or table_side < 1:
        raise ValueError("table_side must be a positive integer")
    if type(spatial_merge_size) is not int or spatial_merge_size < 1:
        raise ValueError("spatial_merge_size must be a positive integer")
    if bool((grid_thw <= 0).any()):
        raise ValueError("Temporal/height/width Patch counts must be positive")
    if bool((grid_thw[:, 1:] % spatial_merge_size != 0).any()):
        raise ValueError("Patch height and width must be divisible by spatial_merge_size")
    if grid_thw.shape[0] == 0:
        return (torch.empty((0, 4), dtype=torch.int64, device=grid_thw.device),
                torch.empty((0, 4), dtype=torch.float32, device=grid_thw.device))

    # Associate every packed Patch row with its image/video grid.
    grid = grid_thw.to(torch.int64)
    counts = grid[:, 0] * grid[:, 1] * grid[:, 2]
    heights = torch.repeat_interleave(grid[:, 1], counts)
    widths = torch.repeat_interleave(grid[:, 2], counts)
    starts = torch.repeat_interleave(F.pad(counts.cumsum(0)[:-1], (1, 0)), counts)
    within_frame = (torch.arange(counts.sum(), device=grid.device) - starts) % (heights * widths)

    # Decode the contiguous m*m Patch groups back into spatial row/column coordinates.
    m = spatial_merge_size
    blocks_w = widths // m
    patch_row = (within_frame // (m * m * blocks_w)) * m + (within_frame // m) % m
    patch_col = ((within_frame // (m * m)) % blocks_w) * m + within_frame % m

    # Map the current Patch grid onto the fixed learned grid in FP32.
    table_row = patch_row.float() * (table_side - 1) / (heights - 1).clamp(min=1)
    table_col = patch_col.float() * (table_side - 1) / (widths - 1).clamp(min=1)
    row_floor, col_floor = torch.floor(table_row), torch.floor(table_col)
    offsets = torch.arange(2, device=grid.device)
    row_indices = (row_floor.long()[:, None] + offsets).clamp(0, table_side - 1)
    col_indices = (col_floor.long()[:, None] + offsets).clamp(0, table_side - 1)

    # Preserve upstream's FP32 distance/weight order for each axis, then take the outer product.
    row_weights = (1 - (table_row[:, None] - row_floor[:, None] - offsets).abs()).clamp(min=0)
    col_weights = (1 - (table_col[:, None] - col_floor[:, None] - offsets).abs()).clamp(min=0)
    interp_indices = (row_indices[:, :, None] * table_side + col_indices[:, None, :]).reshape(-1, 4)
    interp_weights = (row_weights[:, :, None] * col_weights[:, None, :]).reshape(-1, 4)
    return interp_indices, interp_weights


def vision_position_embedding(x: torch.Tensor, position_table: torch.Tensor,
                              interp_indices: torch.Tensor, interp_weights: torch.Tensor) -> torch.Tensor:
    """OP-11: four bilinear neighbours per row, supplied in processor order.

    vision_position_interpolation generates the IF-04 indices/weights with
    bilinear/align_corners=True and merge-compatible ordering.
    FP32 interpolation weights promote the weighted sum before its BF16 cast.
    """
    if x.ndim != 2 or position_table.ndim != 2 or position_table.shape[1] != x.shape[1]:
        raise ValueError("Position table and hidden features must have the same width")
    if interp_indices.shape != (x.shape[0], 4) or interp_weights.shape != interp_indices.shape:
        raise ValueError("Expected four interpolation indices and weights per Patch")
    if interp_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("Interpolation indices must be int32 or int64")
    if interp_weights.dtype != torch.float32:
        raise TypeError("Interpolation weights must retain the reference's FP32 dtype")
    if interp_indices.numel() and (int(interp_indices.min()) < 0 or int(interp_indices.max()) >= len(position_table)):
        raise ValueError("Interpolation index outside position table")
    neighbours = F.embedding(interp_indices, position_table)
    position = (neighbours * interp_weights.unsqueeze(-1)).sum(1)
    return x + position.to(x.dtype)


def vision_axial_cos_sin(position_ids: torch.Tensor, head_dim: int = 72,
                         theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 axial frequencies for IF-04 spatial positions [N,2] = (h,w)."""
    if position_ids.ndim != 2 or position_ids.shape[1] != 2 or head_dim <= 0 or head_dim % 4:
        raise ValueError("Expected [N,2] positions and a head_dim divisible by four")
    spatial_dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, spatial_dim, 2, device=position_ids.device).float() / spatial_dim))
    angles = position_ids.unsqueeze(-1).float() * inv_freq
    frequencies = torch.cat((angles[:, 0], angles[:, 1]), dim=-1)
    frequencies = torch.cat((frequencies, frequencies), dim=-1)
    return frequencies.cos(), frequencies.sin()


def vision_axial_rope(q: torch.Tensor, k: torch.Tensor,
                      cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate whole heads in FP32, then restore the original Q/K dtypes."""
    if q.shape != k.shape or q.ndim != 3 or q.shape[-1] % 2:
        raise ValueError("Expected same-shaped Q/K [N,H,even head_dim]")
    if cos.shape != (q.shape[0], q.shape[-1]) or sin.shape != cos.shape:
        raise ValueError("cos/sin must have shape [N,head_dim]")
    q32, k32 = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_half = torch.cat((-q32[..., q.shape[-1] // 2:], q32[..., :q.shape[-1] // 2]), dim=-1)
    k_half = torch.cat((-k32[..., k.shape[-1] // 2:], k32[..., :k.shape[-1] // 2]), dim=-1)
    return (q32 * cos + q_half * sin).to(q.dtype), (k32 * cos + k_half * sin).to(k.dtype)


def vision_attention(x: torch.Tensor, qkv_weight: torch.Tensor, qkv_bias: torch.Tensor,
                      proj_weight: torch.Tensor, proj_bias: torch.Tensor, *, cos: torch.Tensor,
                      sin: torch.Tensor, cu_seqlens: torch.Tensor, num_heads: int = 16) -> torch.Tensor:
    """OP-09: noncausal MHA, independently within each packed visual segment."""
    if x.ndim != 2 or num_heads <= 0 or x.shape[1] % num_heads:
        raise ValueError("Vision hidden width must be divisible by num_heads")
    if cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (torch.int32, torch.int64) or len(cu_seqlens) < 2:
        raise ValueError("cu_seqlens must be a cumulative integer vector")
    boundaries = cu_seqlens.tolist()
    if boundaries[0] != 0 or boundaries[-1] != x.shape[0] or any(a >= b for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("Visual segments must be positive and partition all Patch rows")
    n, d = x.shape
    head_dim = d // num_heads
    q, k, v = F.linear(x, qkv_weight, qkv_bias).reshape(n, 3, num_heads, head_dim).unbind(1)
    q, k = vision_axial_rope(q, k, cos, sin)
    parts = []
    for begin, end in zip(boundaries, boundaries[1:]):
        qh, kh, vh = (t[begin:end].transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        scores = torch.matmul(qh, kh.transpose(-1, -2)) * head_dim**-0.5
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        parts.append(torch.matmul(probabilities, vh).transpose(1, 2).contiguous())
    context = torch.cat(parts, dim=1).reshape(n, d)
    return F.linear(context, proj_weight, proj_bias)


def dense_ffn(x: torch.Tensor, fc1_weight: torch.Tensor, fc1_bias: torch.Tensor,
              fc2_weight: torch.Tensor, fc2_bias: torch.Tensor) -> torch.Tensor:
    """OP-10: Vision block Linear -> GELU(tanh) -> Linear, including biases."""
    hidden = F.linear(x, fc1_weight, fc1_bias)
    return F.linear(F.gelu(hidden, approximate="tanh"), fc2_weight, fc2_bias)


def spatial_merger(x: torch.Tensor, norm_weight: torch.Tensor, norm_bias: torch.Tensor,
                   fc1_weight: torch.Tensor, fc1_bias: torch.Tensor, fc2_weight: torch.Tensor,
                   fc2_bias: torch.Tensor, *, spatial_merge_size: int = 2, eps: float = 1e-6) -> torch.Tensor:
    """OP-12: Norm before grouping, then Linear -> exact GELU -> Linear.

    Incoming rows already follow the processor's contiguous m*m Patch groups.
    """
    group = spatial_merge_size**2
    if x.ndim != 2 or spatial_merge_size < 1 or x.shape[0] % group:
        raise ValueError("Patch rows must contain complete spatial merge groups")
    normalized = layer_norm(x, norm_weight, norm_bias, eps)
    grouped = normalized.reshape(-1, group * x.shape[-1])
    hidden = F.gelu(F.linear(grouped, fc1_weight, fc1_bias), approximate="none")
    return F.linear(hidden, fc2_weight, fc2_bias)
