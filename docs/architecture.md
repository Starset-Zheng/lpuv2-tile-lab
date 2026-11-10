# 工程分层

保留pymodel中“公共定义—指令接口—算子—reference—测试—产物”的职责关系。各层从模型计算需求出发独立演进。

| 层 | 目录 | 职责与交接 |
| --- | --- | --- |
| 公共契约 | pymodel/common | Tensor/Tile的shape、dtype、布局、有效范围、索引及状态约定 |
| 指令适配 | pymodel/isa | 接入实际PTO调用与执行后端；后续接入LPU候选ISA |
| 候选语义 | pymodel/semantic | 后续独立定义LPU操作语义，服务功能模型与一致性检查 |
| 算子组织 | pymodel/ops | 一个OP对应一个完整逻辑kernel，场景与内部路径由入口组织 |
| Tile程序 | kernels/pto | 当前Python/PyPTO完整kernel及内部Tile步骤，后续接入实际后端 |
| Golden | pymodel/ref | 独立PyTorch reference，导出输出、索引与状态检查点 |
| 模型集成 | pymodel/models/qwen36 | Vision、Decoder、MTP、主干及生成状态交接 |
| 实验产物 | pymodel/task_gen | 后续组织case、输入输出和运行metadata；格式随执行后端确定 |
| 验证 | tests | 按指令、kernel、算子、子图、模型逐级验证 |

当前在Python层执行独立golden与Tile语义程序，展开完整OP内部的计算、分块和更新。PTO完整ISA说明独立维护，仓库保存固定版本索引与程序。LPUv2的Tile register、Slice、更新语义、依赖和生命周期由程序需求推动定义；当前没有预设硬件编码。

当前共16个逻辑kernel，对应16条OP。内部阶段和Prefill/Decode、recurrent/chunk路径分别验证中间结果，再检查完整OP输出；子图按模型计算流组合这些入口。当前编号与历史映射见 [kernel组织](kernel-organization.md)。

首次实现可从OP-04 Residual Add与OP-02 RMSNorm切入，分别由A/B完成；这只是后续入口建议，登记状态当前均为planned。
