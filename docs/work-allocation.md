# 两人分工

A主责GDN与视觉分支，B主责Full Attention、MoE、输入输出及MTP。按计算、状态和验证复杂度粗估分配：A为9条OP/17个kernel，B为7条OP/20个kernel。

A已由jcx认领，工作分支为 `operator_dev_jcx`。下表记录reference进度；PTO实现和模型验证独立登记。接口和验证范围见 [A组reference说明](reference-a.md)。

| 主责 | OP | 算子 | 当前状态 |
| --- | --- | --- | --- |
| B | OP-01 | Token Embedding | planned |
| A | OP-08 | Vision Patch Embedding | golden_ready |
| A | OP-11 | 视觉位置嵌入 | golden_ready |
| A | OP-03 | LayerNorm | golden_ready |
| A | OP-09 | Vision Attention | golden_ready |
| A | OP-04 | Residual Add | golden_ready |
| A | OP-10 | Dense FFN（GELU MLP） | golden_ready |
| A | OP-12 | Spatial Merger / Projector | golden_ready |
| A | OP-13 | Visual Feature Scatter | golden_ready |
| B | OP-02 | RMSNorm | planned |
| A | OP-06 | Gated DeltaNet 线性注意力 | in_progress：核心/状态图已验证，待目标FP8投影参考 |
| B | OP-05 | Full Attention | planned |
| B | OP-07 | MoE FFN | planned |
| B | OP-14 | LM Head | planned |
| B | OP-15 | Feature Concat | planned |
| B | OP-16 | MTP 输入投影 | planned |

## 公共接口与集成

A负责GDN/Conv状态、Vision/位置元数据及视觉接入；B负责KV、路由与MTP接口并牵头文本主干集成。MTP接受前缀与拒绝恢复由B组织，A提供GDN/Conv状态配合。

RMSNorm由B、LayerNorm/Residual由A维护；GEMM与量化接口由B牵头，参考版本及比较入口由A牵头。公共接口变更说明受影响的调用点，另一人交叉评审。

完整逐kernel归属见 [kernels.json](../data/kernels.json)，状态接口见 [interfaces.json](../data/interfaces.json)。首轮核心原型完成后按实际用时调整边界清晰的任务。
