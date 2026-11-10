# pymodel/ref

后续建立独立PyTorch golden，按算子、子图、层和模型导出输出、索引与状态。

共同输入是 [Qwen3.6-35B-A3B-FP8基线](../../configs/models/qwen3_6_35b_a3b_fp8/README.md)：[模型结构与来源](../../configs/models/qwen3_6_35b_a3b_fp8/baseline.json)、[数值契约](../../configs/models/qwen3_6_35b_a3b_fp8/numerics.json)。计算时保留官方dtype、权重解释及cast位置，实际执行环境和比较规则随case记录。

当前只完成配置和规则整理，reference实现仍为planned。MTP需先补充支持它的执行reference，详见 [权重与参考范围说明](../../docs/model-numerical-baseline.md#11-模型权重与量化参数)。
