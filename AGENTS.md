# Repository guidance

- This phase contains documentation, registries and an upstream submodule. Implement algorithms only when a task authorizes implementation.
- Read the OP/kernel contract and independent reference before editing an operator. Keep `pymodel/ref` independent of `kernels/pto` and backend dispatch.
- The shared model and numerical baseline is `configs/models/qwen3_6_35b_a3b_fp8/`; read `baseline.json`, `numerics.json` and `docs/model-numerical-baseline.md` before implementing a reference. Preserve source cast boundaries and checkpoint scales; the pinned Transformers main model does not implement MTP.
- PTO is pinned by `data/upstream.lock.json`. Read `third_party/pto-isa/AGENTS.md` when working inside the submodule; do not change upstream as a workaround for local input/layout mistakes.
- `data/operators.json` and `data/kernels.json` track owners and separate implementation/validation states. A file, catalog entry or placeholder is not implementation evidence.
- A PTO result must execute upstream PTO intrinsics. Missing backends fail explicitly; never substitute a golden result as an actual backend output.
- Model outputs, routing indices, KV and GDN/Conv state are separately named and compared.
- LPUv2 instruction encoding, hardware buffer addresses and instruction width remain design inputs; do not copy an LPUv1 binary ABI into this project.
- Keep generated tensors, model weights, credentials, binaries and build outputs out of git.
- At this framework stage, check registry coverage, source links and submodule pin. Add numerical and backend checks with the first actual implementation.
