# 两人分工

A主责GDN与视觉分支，B主责Full Attention、MoE、输入输出及MTP。按计算、状态和验证复杂度粗估分配：A为9条OP/17个kernel，B为7条OP/20个kernel。

| 主责 | OP | 算子 | 当前状态 |
| --- | --- | --- | --- |
| B | OP-01 | Token Embedding | planned |
| A | OP-08 | Vision Patch Embedding | planned |
| A | OP-11 | 视觉位置嵌入 | planned |
| A | OP-03 | LayerNorm | planned |
| A | OP-09 | Vision Attention | planned |
| A | OP-04 | Residual Add | planned |
| A | OP-10 | Dense FFN（GELU MLP） | planned |
| A | OP-12 | Spatial Merger / Projector | planned |
| A | OP-13 | Visual Feature Scatter | planned |
| B | OP-02 | RMSNorm | planned |
| A | OP-06 | Gated DeltaNet 线性注意力 | planned |
| B | OP-05 | Full Attention | planned |
| B | OP-07 | MoE FFN | planned |
| B | OP-14 | LM Head | planned |
| B | OP-15 | Feature Concat | planned |
| B | OP-16 | MTP 输入投影 | planned |

## 公共接口与集成

A负责GDN/Conv状态、Vision/位置元数据及视觉接入；B负责KV、路由与MTP接口并牵头文本主干集成。MTP接受前缀与拒绝恢复由B组织，A提供GDN/Conv状态配合。

RMSNorm由B、LayerNorm/Residual由A维护；GEMM与量化接口由B牵头，参考版本及比较入口由A牵头。公共接口变更说明受影响的调用点，另一人交叉评审。

完整逐kernel归属见 [kernels.json](../data/kernels.json)，状态接口见 [interfaces.json](../data/interfaces.json)。首轮核心原型完成后按实际用时调整边界清晰的任务。
