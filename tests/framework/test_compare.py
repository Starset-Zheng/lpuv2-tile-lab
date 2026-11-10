"""Framework checks; these fixtures do not validate any model OP or PTO kernel."""

import json

import pytest
import torch

from pymodel.common.case import OutputCheck
from pymodel.utils.compare import compare_outputs


def compare(actual, expected, **kwargs):
    return compare_outputs({"out": actual}, {"out": expected}, {"out": OutputCheck(**kwargs)})


def test_tiny_reference_values_are_still_checked():
    report = compare(torch.tensor([1.0, 0.01]), torch.tensor([1.0, 0.0]), atol=1e-5, rtol=0.1)
    assert not report["passed"]
    assert report["outputs"]["out"]["worst_index"] == [1]
    assert report["outputs"]["out"]["max_rel_error"] == "inf"
    json.dumps(report, allow_nan=False)


def test_one_bad_element_is_not_hidden_by_an_average():
    expected = torch.ones(1000, dtype=torch.bfloat16)
    actual = expected.clone()
    actual[-1] = 2
    report = compare(actual, expected, atol=0.01, rtol=0.01)
    assert not report["passed"]
    assert report["outputs"]["out"]["mismatch_count"] == 1


def test_each_output_has_its_own_tolerance():
    expected = {"out": torch.zeros(2, dtype=torch.bfloat16), "state_out": torch.zeros(2)}
    actual = {"out": torch.full((2,), 0.01, dtype=torch.bfloat16), "state_out": torch.full((2,), 0.01)}
    report = compare_outputs(actual, expected, {
        "out": OutputCheck(atol=0.02, rtol=0), "state_out": OutputCheck(atol=1e-6, rtol=0),
    })
    assert report["outputs"]["out"]["passed"]
    assert not report["outputs"]["state_out"]["passed"]
    assert not report["passed"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_values_fail_even_if_present_in_both_results(value):
    report = compare(torch.tensor([value]), torch.tensor([value]), atol=0, rtol=0)
    assert not report["passed"]
    assert report["outputs"]["out"]["actual_nonfinite_count"] == 1
    assert "nonfinite_value" in report["outputs"]["out"]["reasons"]
    json.dumps(report, allow_nan=False)


def test_equal_values_do_not_hide_wrong_output_dtype():
    report = compare(torch.ones(2), torch.ones(2, dtype=torch.bfloat16), atol=0, rtol=0)
    assert not report["passed"]
    assert report["outputs"]["out"]["reasons"] == ["dtype_mismatch"]


def test_shapes_are_not_broadcast_for_comparison():
    report = compare(torch.ones(2, 1), torch.ones(2), atol=0, rtol=0)
    assert not report["passed"]
    assert report["outputs"]["out"]["reasons"] == ["shape_mismatch"]


def test_all_named_outputs_are_required_and_extra_outputs_are_reported():
    value = torch.ones(1)
    report = compare_outputs({"out": value, "extra": value}, {"out": value, "state_out": value}, {
        "out": OutputCheck(atol=0, rtol=0), "state_out": OutputCheck(atol=0, rtol=0),
    })
    assert not report["passed"]
    assert report["missing_outputs"] == ["state_out"]
    assert report["unexpected_outputs"] == ["extra"]


def test_int64_indices_are_compared_exactly_above_float64_integer_precision():
    expected = torch.tensor([2**53, 2**53 + 1], dtype=torch.int64)
    actual = expected.flip(0)
    report = compare(actual, expected)
    assert not report["passed"]
    assert report["outputs"]["out"]["mismatch_count"] == 2


def test_checked_width_index_representation_preserves_uint32_values():
    expected = torch.tensor([0, 2**31, 2**32 - 1], dtype=torch.int64)
    actual = expected.to(torch.uint32)
    assert compare(actual, expected, logical_indices=True)["passed"]
    assert not compare(actual, expected)["passed"]
    bad = torch.tensor([0, 2**31, 0], dtype=torch.uint32)
    assert not compare(bad, expected, logical_indices=True)["passed"]


def test_index_wraparound_is_not_treated_as_matching():
    report = compare(torch.tensor([2**32 - 1], dtype=torch.uint32), torch.tensor([-1]), logical_indices=True)
    assert not report["passed"]


def test_scalar_and_empty_tensors_have_defined_diagnostics():
    scalar = compare(torch.tensor(2.0), torch.tensor(1.0), atol=0, rtol=0)
    assert scalar["outputs"]["out"]["worst_index"] == []
    empty = compare(torch.empty(0, 3), torch.empty(0, 3), atol=0, rtol=0)
    assert empty["passed"]
    assert empty["outputs"]["out"]["max_abs_error"] is None


def test_missing_float_tolerance_and_incomplete_output_rules_are_errors():
    with pytest.raises(ValueError, match="explicit atol/rtol"):
        compare(torch.ones(1), torch.ones(1))
    with pytest.raises(ValueError, match="exactly"):
        compare_outputs({"out": torch.ones(1)}, {"out": torch.ones(1)}, {})


@pytest.mark.parametrize("kwargs", [
    {"atol": -1, "rtol": 0}, {"atol": float("nan"), "rtol": 0},
    {"atol": 0}, {"atol": True, "rtol": 0}, {"logical_indices": "yes"},
    {"logical_indices": True, "atol": 0, "rtol": 0},
])
def test_invalid_comparison_rules_are_rejected(kwargs):
    with pytest.raises(ValueError):
        OutputCheck(**kwargs)
