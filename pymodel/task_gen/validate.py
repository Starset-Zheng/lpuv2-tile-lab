"""Run independent reference and implementation callbacks on isolated copies."""

from collections.abc import Callable, Mapping
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import platform
import subprocess
import traceback
from uuid import uuid4

import torch

from pymodel.common.case import Case
from pymodel.utils.compare import check_tensors, compare_outputs, tensor_info


Executor = Callable[[dict[str, torch.Tensor], dict], Mapping[str, torch.Tensor]]
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    run_dir: Path
    report: dict


def _git(path: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _executor_info(executor: Executor | None) -> dict | None:
    if executor is None:
        return None
    try:
        source = inspect.getsourcefile(executor)
    except TypeError:
        source = None
    return {
        "callable": f"{getattr(executor, '__module__', type(executor).__module__)}:"
                    f"{getattr(executor, '__qualname__', type(executor).__qualname__)}",
        "source_file": source,
        "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest()
        if source and Path(source).is_file() else None,
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def run_case(case: Case, *, inputs: Mapping[str, torch.Tensor], reference: Executor,
             implementation: Executor | None, backend: str, artifact_root: str | Path = "artifacts",
             backend_details: dict | None = None) -> ValidationResult:
    """Callbacks receive (named_inputs, parameters) and return every named output.

    A backend label records the requested execution path, not proof of native PTO
    coverage. Its adapter supplies build/binary/device details when implemented.
    Missing implementations and execution errors raise after saving diagnostics.
    """
    check_tensors(inputs)
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError("An explicit backend is required")
    if not callable(reference) or (implementation is not None and not callable(implementation)):
        raise TypeError("reference and implementation must be callable")
    details = backend_details if backend_details is not None else {}
    if not isinstance(details, dict):
        raise TypeError("backend_details must be a JSON object")
    json.dumps(details, allow_nan=False)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:12]
    run_dir = Path(artifact_root) / case.case_id / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "run_id": run_id,
        "case": case.to_dict(),
        "backend": backend,
        "backend_details": details,
        "reference": _executor_info(reference),
        "implementation": _executor_info(implementation),
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
        "repository": {
            "commit": _git(ROOT, "rev-parse", "HEAD"),
            "worktree_status": _git(ROOT, "status", "--porcelain"),
            "pto_checkout": _git(ROOT / "third_party/pto-isa", "rev-parse", "HEAD")
            if (ROOT / "third_party/pto-isa/.git").exists() else None,
        },
        "inputs": {name: tensor_info(tensor) for name, tensor in inputs.items()},
        "files": ["inputs.pt", "manifest.json", "report.json", "run.log"],
    }
    _write_json(run_dir / "manifest.json", manifest)
    stage = "prepare"
    try:
        torch.save(dict(inputs), run_dir / "inputs.pt")
        if implementation is None:
            raise RuntimeError(f"Backend {backend!r} has no implementation for this case")
        reference_inputs = copy.deepcopy(dict(inputs))
        actual_inputs = copy.deepcopy(dict(inputs))
        with torch.inference_mode():
            stage = "reference"
            expected = reference(reference_inputs, copy.deepcopy(case.parameters))
            check_tensors(expected)
            expected = copy.deepcopy(dict(expected))
            torch.save(expected, run_dir / "reference.pt")
            manifest["files"].append("reference.pt")
            stage = "implementation"
            actual = implementation(actual_inputs, copy.deepcopy(case.parameters))
            check_tensors(actual)
            torch.save(dict(actual), run_dir / "actual.pt")
            manifest["files"].append("actual.pt")
            stage = "comparison"
            report = compare_outputs(actual, expected, case.outputs)
        manifest["status"] = "passed" if report["passed"] else "failed"
        _write_json(run_dir / "report.json", report)
        (run_dir / "run.log").write_text(f"{case.case_id}: {manifest['status']} ({backend})\n")
    except Exception as exc:
        manifest["status"] = "error"
        report = {"passed": False, "stage": stage, "error_type": type(exc).__name__, "error": str(exc)}
        _write_json(run_dir / "report.json", report)
        (run_dir / "run.log").write_text(traceback.format_exc())
        raise
    finally:
        _write_json(run_dir / "manifest.json", manifest)
    return ValidationResult(passed=report["passed"], run_dir=run_dir, report=report)
