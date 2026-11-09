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
