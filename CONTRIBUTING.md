# 协作方式

每个任务以一个OP或明确的kernel为主责单位，另一人交叉评审。A/B是临时角色名，具体GitHub用户名认领后填入Issue。

## 认领与分支

- 从 `main` 建立 `feat/op-06-gdn-recurrent` 这类任务分支。
- Issue写清OP/K编号、主责、范围、输入输出和状态；融合kernel列出全部关联OP。
- 公共shape/dtype/layout、状态更新和结果格式变化，先在PR中说明依赖及受影响调用点。

## 一个实现任务的交付

1. 固定参考来源和版本，写清公式、输入输出、mask、索引与状态，建立独立PyTorch golden。
2. 记录Tile物理/有效形状、布局、临时对象、数据搬运、生命周期和同步关系，再实现PTO kernel。
3. 使用相同输入比较golden与实际后端输出，覆盖尾块、有效范围和相关状态；保存可复现元数据。
4. 更新对应OP/K的实现及验证字段，说明适用dtype、shape、backend和实际执行的命令。
5. 在 `docs/isa-requirements.md` 记录程序揭示的LPUv2能力需求，关联具体case和kernel。

## 状态

`planned → golden_ready → pto_cpu_verified → subgraph_verified → model_verified` 表示推进方向。登记表保留golden、PTO实现、PTO验证和模型验证的独立字段，允许逐项推进。模型验证结果写明模型配置及执行场景。

当前只搭建框架。首次实现PR同时接入必要的构建与测试入口，届时再建立实际数值CI。
