# 验证约定

共同遵循 [模型与数值基线](model-numerical-baseline.md) 和 [数值契约](../configs/models/qwen3_6_35b_a3b_fp8/numerics.json)。基线锁定配置、权重revision及参考源码；具体FP8执行后端与浮点容差随实际case确认。

| 层级 | 比较对象 | 完成时记录 |
| --- | --- | --- |
| Golden | 固定参考函数/官方实现与独立reference | 来源revision、配置、dtype、公式及中间结果 |
| Kernel/OP | 同输入golden与实际PTO输出 | 后端、指令组合、Tile配置、有效范围、输出/索引/状态及容差 |
| 子图/层 | Vision、GDN/FullAttention+MoE、MTP | 节点与边界输出、残差、cache和状态交接 |
| 模型 | Prefill、连续Decode及后续多模态/MTP | 层输出、logits、生成结果及全部可写状态 |

部分Tile、多Tile尾块、零值/常量行、非零初始状态、空专家和多请求隔离随相关实现加入。比较每个命名输出；GDN State、Conv历史、KV和路由索引单独记录。

CPU功能验证、NPU验证及LPU功能模型验证分别登记。CPU模拟结果用于验证当前CPU路径的数值与边界语义；硬件同步、周期和资源结论随对应目标环境验证。

## 先运行公共框架检查

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/framework -q
```

当前测试覆盖比较器、用例配置、输入/状态隔离和复现产物。测试中的小型回调统一标记为 `framework_fixture`，模型OP与PTO kernel状态保持 `planned`。首个算子接入时增加真实reference、后端编译和对应测试。

## 公共入口与原pymodel的对应关系

| 原pymodel的职责 | 本仓库入口 |
| --- | --- |
| `test.py` 组织输入、算子、reference与断言 | `tests/operators/`、`tests/kernels/` 等pytest用例；公共执行入口为 `pymodel/task_gen/validate.py::run_case` |
| `utils.py` 生成参数、比较结果 | `pymodel/common/case.py::Case` 固定参数与seed；`pymodel/utils/compare.py::compare_outputs` 比较所有命名输出 |
| `gen_taskbin.py` 保存输入、输出及指令 | `run_case` 保存tensor、case、环境及比较报告；后端有真实指令格式后再接入指令导出 |

Reference和被测实现分别接收一份深拷贝的输入及参数，可以原地更新自己的状态。拷贝保留输入tensor之间的存储别名及stride；同一case的reference、被测实现和外部原始输入分别隔离。框架使用inference模式，保留各算子的dtype/cast，不开启统一autocast。

## Case与比较规则

[示例case](../cases/examples/op04-residual-bf16.json)定义OP/K编号、数值profile、seed、参数及输出检查规则。输入tensor由算子用例显式构造，随机数使用 `case.generator()`；JSON不承担通用计算图或Tensor生成DSL。OP-04示例中的零容差用于严格残差加法用例，其他算子按实际精度路径单独校准。

每个回调返回 `dict[str, torch.Tensor]`，例如 `{"out": ..., "state_out": ...}`。case必须声明reference返回的全部输出；比较器报告缺失或多余的输出。

- 浮点输出同时检查shape、dtype与每个元素的 `abs(actual-reference) <= atol + rtol * abs(reference)`。每个输出显式提供atol/rtol；接近零的参考值也参与比较。
- 整数和布尔输出精确比较，不使用浮点容差。专家索引需要比较int64 / uint32等不同存储类型时，显式设置 `logical_indices=true`，在不丢失整数值的int64域比较。
- 默认所有NaN/Inf都导致失败，包含双方同为Inf的情况。需比较含合法无穷值的中间mask时，再为该用例明确专门策略。
- 报告按输出记录最大绝对/相对误差、最坏坐标、不匹配数量/比例和非有限值数量。浮点分析转FP64，执行和保存的tensor保留原dtype；整数比较保留精确性。
- 相同输出形状及数值不能替代dtype检查。TopK同值顺序、有效区域和padding语义由对应算子用例显式验证。

## 首个算子的接入方式

下面展示pytest用例的组织方式。`residual_ref`与`residual_pto_cpu`由OP-04实现任务提供，目前仓库尚未提供这两个函数。

```python
import torch

from pymodel.common.case import load_case
from pymodel.task_gen.validate import run_case


def test_residual_bf16(tmp_path):
    case = load_case("cases/examples/op04-residual-bf16.json")
    p, generator = case.parameters, case.generator()
    shape = (p["rows"], p["cols"])
    inputs = {
        "x": torch.randn(shape, generator=generator, dtype=torch.bfloat16),
        "residual": torch.randn(shape, generator=generator, dtype=torch.bfloat16),
    }
    result = run_case(
        case, inputs=inputs,
        reference=lambda t, p: {"out": residual_ref(t["x"], t["residual"])},
        implementation=lambda t, p: {"out": residual_pto_cpu(t["x"], t["residual"])},
        backend="pto_cpu", artifact_root=tmp_path,
        backend_details={"target_family": "<actual target>", "binary_sha256": "<actual build hash>"},
    )
    assert result.passed, f"{result.report}\nartifacts: {result.run_dir}"
```

`backend`记录调用路径，实际PTO覆盖依据对应适配器和kernel源码审核，并在后端详情中记录实际编译器、binary hash、target和device。当前框架只提供回调接口，不提供PTO编译器或执行适配器；实现缺失会报错，运行异常保存诊断后重新抛出。

## 运行产物与复现

每次运行在 `artifact_root/<case_id>/<run_id>/` 建立独立目录。run_id含UTC时间和随机后缀，两人并行或重复运行时分别保存结果。默认artifact_root为当前目录的 `artifacts/`；pytest中可用 `tmp_path`，排错时改为需要保留的路径。

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | case、数值profile、backend及其详情、Python/Torch版本、repo/PTO版本、工作区状态、回调源码hash和输入shape/dtype/stride/device |
| `inputs.pt` | 本次输入，包含输入状态；在执行前保存 |
| `reference.pt` / `actual.pt` | 对应路径成功返回的全部命名tensor；保留BF16、FP32和整数dtype |
| `report.json` | 各输出检查结果；异常时记录失败阶段、异常类型和信息 |
| `run.log` | 成功/失败摘要，或异常traceback |

读取tensor文件使用 `torch.load(path, map_location="cpu", weights_only=True)`。callback执行失败时保留已完成阶段的产物；比较失败返回 `passed=false`，pytest用例必须断言该字段。框架本身不提升OP/K登记状态，实际通过的测试与后端证据由实现PR记录。
