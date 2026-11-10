"""Exercise runner failures, state isolation and exact artifact serialization."""

import json
from pathlib import Path

import pytest
import torch

from pymodel.common.case import Case, OutputCheck, load_case
from pymodel.task_gen.validate import run_case


def make_case():
    return Case(case_id="framework-fixture", seed=17, numerical_profile="framework_fixture", outputs={
        "out": OutputCheck(atol=0, rtol=0), "state_out": OutputCheck(atol=0, rtol=0),
        "expert_ids": OutputCheck(logical_indices=True),
    })


def make_inputs(case):
    return {
        "x": torch.randn(3, 17, generator=case.generator(), dtype=torch.bfloat16),
        "state": torch.zeros(2, 3, dtype=torch.float32),
        "expert_ids": torch.tensor([0, 255], dtype=torch.int64),
    }


def fixture_reference(inputs, parameters):
    return {"out": inputs["x"] + 0.25, "state_out": inputs["state"] + 2,
            "expert_ids": inputs["expert_ids"]}


def fixture_implementation(inputs, parameters):
    inputs["x"].add_(0.25)
    inputs["state"].add_(2)
    return {"out": inputs["x"], "state_out": inputs["state"],
            "expert_ids": inputs["expert_ids"].to(torch.uint32)}


def test_input_generator_is_reproducible_and_does_not_reset_global_rng():
    case = make_case()
    state = torch.random.get_rng_state()
    assert torch.equal(make_inputs(case)["x"], make_inputs(case)["x"])
    assert torch.equal(state, torch.random.get_rng_state())


def test_runner_isolates_inputs_and_preserves_bf16_state_and_indices(tmp_path):
    case = make_case()
    inputs = make_inputs(case)
    original_x = inputs["x"].clone()
    result = run_case(case, inputs=inputs, reference=fixture_reference,
                      implementation=fixture_implementation, backend="framework_fixture", artifact_root=tmp_path)
    assert result.passed
    assert torch.equal(inputs["x"], original_x)
    assert not torch.count_nonzero(inputs["state"])
    saved_inputs = torch.load(result.run_dir / "inputs.pt", weights_only=True)
    expected = torch.load(result.run_dir / "reference.pt", weights_only=True)
    actual = torch.load(result.run_dir / "actual.pt", weights_only=True)
    assert torch.equal(saved_inputs["x"].view(torch.uint16), original_x.view(torch.uint16))
    assert actual["out"].dtype == expected["out"].dtype == torch.bfloat16
    assert actual["state_out"].dtype == torch.float32
    assert actual["expert_ids"].dtype == torch.uint32
    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    assert manifest["status"] == "passed"
    assert manifest["backend"] == "framework_fixture"
    assert manifest["case"]["op_id"] is None
    assert manifest["reference"]["source_sha256"]
    assert all((result.run_dir / name).is_file() for name in manifest["files"])


def test_wrong_state_fails_even_when_primary_output_matches(tmp_path):
    def wrong_state(inputs, parameters):
        result = fixture_implementation(inputs, parameters)
        result["state_out"].add_(0.5)
        return result

    case = make_case()
    result = run_case(case, inputs=make_inputs(case), reference=fixture_reference,
                      implementation=wrong_state, backend="framework_fixture", artifact_root=tmp_path)
    assert not result.passed
    assert result.report["outputs"]["out"]["passed"]
    assert not result.report["outputs"]["state_out"]["passed"]
    assert json.loads((result.run_dir / "manifest.json").read_text())["status"] == "failed"


def test_missing_backend_is_an_error_and_never_creates_actual_output(tmp_path):
    case = make_case()
    with pytest.raises(RuntimeError, match="no implementation"):
        run_case(case, inputs=make_inputs(case), reference=fixture_reference,
                 implementation=None, backend="unconnected", artifact_root=tmp_path)
    run_dir, = (tmp_path / case.case_id).iterdir()
    assert not (run_dir / "actual.pt").exists()
    assert not (run_dir / "reference.pt").exists()
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "error"


def test_backend_exception_keeps_reference_and_traceback(tmp_path):
    def broken(inputs, parameters):
        raise RuntimeError("fixture backend failed")

    case = make_case()
    with pytest.raises(RuntimeError, match="fixture backend failed"):
        run_case(case, inputs=make_inputs(case), reference=fixture_reference,
                 implementation=broken, backend="framework_fixture", artifact_root=tmp_path)
    run_dir, = (tmp_path / case.case_id).iterdir()
    assert (run_dir / "reference.pt").is_file()
    assert not (run_dir / "actual.pt").exists()
    assert "fixture backend failed" in (run_dir / "run.log").read_text()
    assert json.loads((run_dir / "report.json").read_text())["stage"] == "implementation"


def test_repeated_runs_have_separate_artifact_directories(tmp_path):
    case = make_case()
    kwargs = dict(inputs=make_inputs(case), reference=fixture_reference, implementation=fixture_implementation,
                  backend="framework_fixture", artifact_root=tmp_path)
    first = run_case(case, **kwargs)
    second = run_case(case, **kwargs)
    assert first.passed and second.passed
    assert first.run_dir != second.run_dir


def test_input_views_keep_aliases_in_each_isolated_execution_and_saved_inputs(tmp_path):
    base = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    inputs = {"base": base, "view": base[:, 1:]}
    case = Case(case_id="alias-fixture", seed=1, numerical_profile="framework_fixture",
                outputs={"out": OutputCheck(atol=0, rtol=0)})

    def reference(values, parameters):
        expected = values["base"].clone()
        expected[:, 1:] += 1
        return {"out": expected}

    def implementation(values, parameters):
        values["view"].add_(1)
        return {"out": values["base"]}

    result = run_case(case, inputs=inputs, reference=reference, implementation=implementation,
                      backend="framework_fixture", artifact_root=tmp_path)
    assert result.passed
    assert torch.equal(base, torch.arange(6, dtype=torch.bfloat16).reshape(2, 3))
    loaded = torch.load(result.run_dir / "inputs.pt", weights_only=True)
    assert loaded["view"].stride() == (3, 1)
    assert loaded["base"].untyped_storage().data_ptr() == loaded["view"].untyped_storage().data_ptr()


def test_example_case_loads_without_claiming_an_operator_is_implemented():
    root = Path(__file__).resolve().parents[2]
    case = load_case(root / "cases/examples/op04-residual-bf16.json")
    assert case.op_id == "OP-04"
    assert case.parameters == {"rows": 3, "cols": 17}
    assert case.outputs["out"] == OutputCheck(atol=0, rtol=0)


def test_interactive_callbacks_work_without_a_source_file(tmp_path):
    namespace = {}
    exec(compile("def callback(inputs, parameters):\n    return {'out': inputs['x']}\n",
                 "<framework-fixture>", "exec"), namespace)
    case = Case(case_id="interactive-fixture", seed=0, numerical_profile="framework_fixture",
                outputs={"out": OutputCheck(atol=0, rtol=0)})
    result = run_case(case, inputs={"x": torch.ones(1, dtype=torch.bfloat16)},
                      reference=namespace["callback"], implementation=lambda x, p: {"out": x["x"].clone()},
                      backend="framework_fixture", artifact_root=tmp_path)
    assert result.passed
    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    assert manifest["reference"]["source_sha256"] is None


@pytest.mark.parametrize("case_id", ["../escape", "/tmp/escape", "", ".."])
def test_case_id_cannot_escape_artifact_directory(case_id):
    with pytest.raises(ValueError, match="safe directory"):
        Case(case_id=case_id, seed=0, numerical_profile="framework_fixture", outputs={"out": OutputCheck()})


def test_nonfinite_and_incomplete_case_rules_are_rejected_on_load(tmp_path):
    path = tmp_path / "case.json"
    data = make_case().to_dict()
    data["outputs"]["out"] = {"atol": 0}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="both"):
        load_case(path)
