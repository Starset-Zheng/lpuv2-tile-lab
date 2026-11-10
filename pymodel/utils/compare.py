"""Compare named outputs without changing the execution arithmetic."""

from collections.abc import Mapping
import math

import torch

from pymodel.common.case import OutputCheck


INDEX_DTYPES = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.uint16, torch.uint32}


def tensor_info(tensor: torch.Tensor) -> dict:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "stride": list(tensor.stride()),
        "device": str(tensor.device),
    }


def check_tensors(values: Mapping[str, torch.Tensor]) -> None:
    if not isinstance(values, Mapping) or not values:
        raise ValueError("Expected a nonempty mapping of named tensors")
    for name, value in values.items():
        if not isinstance(name, str) or not name or not isinstance(value, torch.Tensor):
            raise TypeError("Inputs and outputs must map nonempty names to torch.Tensor objects")
        if value.layout != torch.strided or value.is_quantized or value.device.type == "meta":
            raise TypeError(f"{name}: expected a materialized, dense strided tensor")
        if value.is_complex() or value.requires_grad:
            raise TypeError(f"{name}: expected a real inference tensor without gradients")


def _coordinate(flat_index: int, shape: torch.Size) -> list[int]:
    result = []
    for size in reversed(shape):
        result.append(flat_index % size)
        flat_index //= size
    return list(reversed(result))


def _json_number(value: float) -> float | str:
    return value if math.isfinite(value) else "inf"


def _compare_tensor(actual: torch.Tensor, expected: torch.Tensor, rule: OutputCheck) -> dict:
    result = {
        "passed": False,
        "actual": tensor_info(actual),
        "expected": tensor_info(expected),
        "rule": {"atol": rule.atol, "rtol": rule.rtol, "logical_indices": rule.logical_indices},
        "reasons": [],
    }
    floating = expected.is_floating_point()
    if floating and (rule.atol is None or rule.logical_indices):
        raise ValueError("Floating outputs need explicit atol/rtol and cannot use logical_indices")
    if not floating and rule.atol is not None:
        raise ValueError("Integer and boolean outputs require exact comparison")
    if rule.logical_indices and expected.dtype not in INDEX_DTYPES:
        raise ValueError("logical_indices requires integer indices representable as int64")
    if actual.shape != expected.shape:
        result["reasons"].append("shape_mismatch")
        return result
    if actual.dtype != expected.dtype and not rule.logical_indices:
        result["reasons"].append("dtype_mismatch")
    if actual.is_floating_point() != floating or (rule.logical_indices and actual.dtype not in INDEX_DTYPES):
        result["reasons"].append("incompatible_dtype")
        return result

    if floating:
        a = actual.detach().to(device="cpu", dtype=torch.float64)
        e = expected.detach().to(device="cpu", dtype=torch.float64)
        finite = torch.isfinite(a) & torch.isfinite(e)
        diff = torch.where(finite, (a - e).abs(), torch.inf)
        matched = finite & (diff <= rule.atol + rule.rtol * e.abs())
        relative = torch.where(e == 0, torch.where(diff == 0, 0.0, torch.inf), diff / e.abs())
        relative = torch.where(finite, relative, torch.inf)
        result.update(
            max_abs_error=_json_number(diff.max().item()) if diff.numel() else None,
            max_rel_error=_json_number(relative.max().item()) if relative.numel() else None,
            worst_index=_coordinate(int(diff.argmax()), expected.shape) if diff.numel() else None,
            actual_nonfinite_count=int((~torch.isfinite(a)).sum()),
            expected_nonfinite_count=int((~torch.isfinite(e)).sum()),
        )
        if not bool(finite.all()):
            result["reasons"].append("nonfinite_value")
    else:
        a, e = actual.detach().cpu(), expected.detach().cpu()
        if rule.logical_indices:
            a, e = a.to(torch.int64), e.to(torch.int64)
        elif actual.dtype != expected.dtype:
            return result
        matched = a == e
        bad = (~matched).reshape(-1).nonzero()
        result.update(
            max_abs_error=None,
            max_rel_error=None,
            worst_index=_coordinate(int(bad[0, 0]), expected.shape) if bad.numel() else None,
        )
    result["mismatch_count"] = int((~matched).sum())
    result["mismatch_fraction"] = result["mismatch_count"] / matched.numel() if matched.numel() else 0.0
    if result["mismatch_count"]:
        result["reasons"].append("value_mismatch")
    result["passed"] = not result["reasons"]
    return result


def compare_outputs(actual: Mapping[str, torch.Tensor], expected: Mapping[str, torch.Tensor],
                    checks: Mapping[str, OutputCheck]) -> dict:
    check_tensors(actual)
    check_tensors(expected)
    if set(checks) != set(expected):
        raise ValueError("Comparison rules must cover exactly the reference's named outputs")
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    outputs = {name: _compare_tensor(actual[name], expected[name], checks[name])
               for name in expected if name in actual}
    return {
        "passed": not missing and not unexpected and all(item["passed"] for item in outputs.values()),
        "missing_outputs": missing,
        "unexpected_outputs": unexpected,
        "outputs": outputs,
    }
