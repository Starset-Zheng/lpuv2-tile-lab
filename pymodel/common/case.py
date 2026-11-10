"""Small, backend-independent validation case definitions."""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re

import torch


@dataclass(frozen=True)
class OutputCheck:
    atol: float | None = None
    rtol: float | None = None
    logical_indices: bool = False

    def __post_init__(self):
        if (self.atol is None) != (self.rtol is None):
            raise ValueError("Specify both atol and rtol, or neither for an integer output")
        for value in (self.atol, self.rtol):
            if value is not None and (
                type(value) not in (int, float) or not math.isfinite(value) or value < 0
            ):
                raise ValueError("Tolerances must be finite, nonnegative numbers")
        if type(self.logical_indices) is not bool:
            raise ValueError("logical_indices must be a boolean")
        if self.logical_indices and self.atol is not None:
            raise ValueError("Logical indices require exact comparison, without floating tolerances")


@dataclass(frozen=True)
class Case:
    case_id: str
    seed: int
    numerical_profile: str
    outputs: dict[str, OutputCheck]
    parameters: dict = field(default_factory=dict)
    op_id: str | None = None
    kernel_id: str | None = None

    def __post_init__(self):
        if not isinstance(self.case_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.case_id):
            raise ValueError("case_id must be a safe directory name beginning with a letter or digit")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if not isinstance(self.numerical_profile, str) or not self.numerical_profile.strip():
            raise ValueError("A numerical_profile is required")
        if not isinstance(self.outputs, dict) or not self.outputs:
            raise ValueError("Declare every named output and its comparison rule")
        if any(not isinstance(k, str) or not k or not isinstance(v, OutputCheck) for k, v in self.outputs.items()):
            raise ValueError("outputs must map nonempty names to OutputCheck objects")
        if not isinstance(self.parameters, dict):
            raise ValueError("parameters must be a JSON object")
        for name in (self.op_id, self.kernel_id):
            if name is not None and (not isinstance(name, str) or not name):
                raise ValueError("OP/kernel IDs must be nonempty strings or null")
        json.dumps(self.to_dict(), allow_nan=False)

    def generator(self) -> torch.Generator:
        """Use this CPU generator when constructing reproducible input tensors."""
        return torch.Generator(device="cpu").manual_seed(self.seed)

    def to_dict(self) -> dict:
        return {"schema_version": 1, **asdict(self)}


def load_case(path: str | Path) -> Case:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("Expected a case object with schema_version=1")
    data.pop("schema_version")
    outputs = data.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("outputs must be an object")
    data["outputs"] = {name: OutputCheck(**rule) for name, rule in outputs.items()}
    return Case(**data)
