# 协作方式

每个任务以一个OP或明确的kernel为主责单位，另一人交叉评审。A/B是临时角色名，具体GitHub用户名认领后填入Issue。

公共能力的归属、交付顺序和调用约束统一见 [两人分工与公共协作约束](docs/work-allocation.md)。公共定义先以小PR进入主线，依赖它的算子随后接入。

## 认领与分支

- 从 `main` 建立 `feat/op-06-gdn-recurrent` 这类任务分支；已认领的个人开发分支可继续使用，公共改动单独拆出提交。
- Issue写清OP/K编号、主责、范围、输入输出和状态；融合kernel列出全部关联OP。
- 公共shape/dtype/layout、状态更新和结果格式变化，先在PR中说明依赖及受影响调用点。

## 一个实现任务的交付

1. 固定参考来源和版本，写清公式、输入输出、mask、索引与状态，建立独立PyTorch golden。
2. 用Python/PyPTO实现Tile程序，展开逻辑shape、有效区域、分块、数据关系和更新顺序，记录PTO操作对应及补充语义。LPUv2物理容量、Buffer通路和硬件同步随架构映射阶段确定。
3. 按已确认的 [PTO功能验证标准v0](docs/functional-validation-v0.md) 使用相同输入比较golden与Tile程序输出，覆盖尾块、索引和相关状态；保存可复现元数据及实际执行方式。单算子先用随机输入和权重，真实权重在集成阶段接入。
4. 更新对应OP/K的实现及验证字段，说明适用dtype、shape、backend和实际执行的命令。
5. 在 `docs/isa-requirements.md` 记录程序揭示的LPUv2能力需求，关联具体case和kernel。

## 状态

当前按“reference就绪 → Python Tile功能验证通过 → 子图/模型验证”逐级推进，并据此整理LPUv2指令候选集。实际PTO CPU/NPU与LPU功能模型的结果分别登记；现有PTO验证字段用于对应后端，Python阶段在PR和运行记录中写清程序、case及结果。

公共用例、执行、比较与产物入口已建立，先运行 `python -m pytest tests/framework -q`。首次算子实现PR按[验证约定](docs/validation.md)接入独立reference和实际Tile程序，提供数值测试与结果；对应后端的构建和CI随其实现接入。
