# 配置

模型与数值基线放在 [models/qwen3_6_35b_a3b_fp8/](models/qwen3_6_35b_a3b_fp8/)，包括官方配置快照、固定revision、checkpoint元数据和数值契约。说明见 [模型与数值基线](../docs/model-numerical-baseline.md)。

后续kernel运行配置也放在本目录，记录后端、dtype、Tile物理/有效尺寸与布局。具体某次输入、seed和边界场景放在 `cases/`；运行输出放在 `artifacts/`。模型权重保存在仓库外。
