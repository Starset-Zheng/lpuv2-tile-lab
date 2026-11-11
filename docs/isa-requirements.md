# LPUv2 ISA与programming model需求

每条需求来自具体OP/K及可复现case，随程序验证逐步确认。当前尚无已接受的LPUv2指令或编码。

| 字段 | 内容 |
| --- | --- |
| 需求编号 / 状态 | REQ编号；candidate/accepted及评审结果 |
| 计算来源 | OP/K编号、场景、公式与case |
| 数据对象 | Tensor/Tile/Slice、dtype、布局、有效区域及多输出 |
| 行为 | 新建/部分更新/累加/别名、状态可见性与数值规则 |
| 依赖 | producer/consumer、完成条件、同步与生命周期 |
| PTO实现 | 所用指令与后端约束、组合或适配方式 |
| LPU选择 | 候选能力、替代组合、资源/搬运取舍及证据 |
| 验证 | 功能模型、程序路径、误差与回归结果 |

逻辑Tile与硬件tile register的映射、指令编码和buffer分配在候选规范中明确后再实现后端。

## B路线已验证的候选需求

| 编号 / 状态 | 来源及需求 | 当前表达与证据 | 待确定内容 |
| --- | --- | --- | --- |
| REQ-B-01 / candidate | OP-01、K-01 Embedding：根据Token ID读取BF16表行，按原Token位置输出，列尾块只读写有效元素 | `pl.read`读取ID，`pl.load`用该ID作行偏移，`pl.store`写回；H=64/67两例经`pypto_torch_codegen`在CPU与Golden精确一致，见[步骤2记录](golden-stage3-step2-embedding.md) | LPU的索引读取、地址计算与Tile搬运如何分工；32列只是本例分块选择，未固定指令宽度或Buffer位置 |

以上为Python功能证据，不表示已完成PTO CPU/NPU后端验证或已接受LPU指令。
