# Repository guidance

- This phase contains documentation, registries, validation helpers and an upstream submodule. Implement model algorithms only when a task authorizes implementation.
- Read the OP/kernel contract and independent reference before editing an operator. Keep `pymodel/ref` independent of `kernels/pto` and backend dispatch.
- Read `docs/work-allocation.md` for A/B ownership, shared interfaces and delivery order. Keep shared definitions and their consumer examples in focused changes; update affected callers and tests together.
- The shared model and numerical baseline is `configs/models/qwen3_6_35b_a3b_fp8/`; read `baseline.json`, `numerics.json` and `docs/model-numerical-baseline.md` before implementing a reference. Preserve source cast boundaries and checkpoint scales; the pinned Transformers main model does not implement MTP.
- Use the confirmed `functional-v0.json` profile for Python functional validation, including activation quantization, BF16/FP32 tolerances, exact-output checks, TopK ties and expert bucketing. Declare per-output checks explicitly in each case; preserve the distinction between model-source facts and project conventions. The readable contract is `docs/functional-validation-v0.md`.
- PTO is pinned by `data/upstream.lock.json`. Read `third_party/pto-isa/AGENTS.md` when working inside the submodule; do not change upstream as a workaround for local input/layout mistakes.
- `data/operators.json` and `data/kernels.json` track owners and separate implementation/validation states. A file, catalog entry or placeholder is not implementation evidence.
- The current functional phase permits Python Tile semantics and PyPTO-generated Torch execution against an independent golden. Record the actual path, versions, OP/K coverage and results. Results labelled PTO CPU/NPU must execute the corresponding upstream backend; missing implementations fail explicitly, and a golden result must never substitute for the tested program.
- Model outputs, routing indices, KV and GDN/Conv state are separately named and compared.
- LPUv2 instruction encoding, hardware buffer addresses and instruction width remain design inputs; do not copy an LPUv1 binary ABI into this project.
- Keep generated tensors, model weights, credentials, binaries and build outputs out of git.
- Run `python -m pytest tests/framework -q` when changing validation helpers. Framework fixtures do not validate model operators or PTO kernels; add their numerical and backend checks with the first actual implementation.
