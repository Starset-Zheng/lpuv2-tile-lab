# Qwen3.6-35B-A3B-FP8 基线

基线ID：`qwen3_6_35b_a3b_fp8-v1`。模型revision：`c653bba46dfa06e9658dcc0b0dc87fe4685f14a2`；Transformers源码commit：`f90e93c4f83974f634860e246cdfd2c68666d605`。

| 文件 | 职责 |
| --- | --- |
| [config.json](config.json) | 官方原始配置快照，保留完整FP8不量化清单 |
| [generation_config.json](generation_config.json) | 官方生成默认值快照 |
| [baseline.json](baseline.json) | 模型/权重/代码版本、结构尺寸、来源hash与执行范围 |
| [numerics.json](numerics.json) | dtype、Norm、Attention、GDN、MoE、Vision与量化数值规则 |
| [checkpoint_metadata.json](checkpoint_metadata.json) | 全42分片header的dtype/shape核对结果与代表张量 |

详细规则及后续执行记录见 [模型与数值基线](../../../docs/model-numerical-baseline.md)。权重文件保存在仓库外。当前已核对配置、来源源码和checkpoint header；golden、FP8执行环境及模型结果尚未实现/验证。
