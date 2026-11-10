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

当前框架未提供可执行测试。首次kernel落地时增加构建、运行、失败复现与CI命令。
