# pymodel/ref

独立PyTorch reference按算子组织，供后续PTO/Tile实现及模型组装共用。

共同输入是 [Qwen3.6-35B-A3B-FP8基线](../../configs/models/qwen3_6_35b_a3b_fp8/README.md)：[模型结构与来源](../../configs/models/qwen3_6_35b_a3b_fp8/baseline.json)、[数值契约](../../configs/models/qwen3_6_35b_a3b_fp8/numerics.json)。计算时保留官方dtype、权重解释及cast位置，实际执行环境和比较规则随case记录。

当前Python功能reference与Tile实现采用 [PTO功能验证标准v0](../../docs/functional-validation-v0.md)，共同配置为 [functional-v0.json](../../configs/models/qwen3_6_35b_a3b_fp8/functional-v0.json)。按已确认的动态量化、误差阈值及TopK规则构造参考结果，并在case中显式声明检查。

A组（jcx）的reference已落在 [common.py](common.py)、[vision.py](vision.py)、[multimodal.py](multimodal.py) 和 [gdn.py](gdn.py)。接口、测试及FP8投影依赖见 [A组reference说明](../../docs/reference-a.md)。运行 `python -m pytest tests/operators -q`。

视觉位置插值元数据由 `vision_position_interpolation(grid_thw)` 独立生成，返回int64 `interp_indices[N,4]`和FP32 `interp_weights[N,4]`，可直接传给 `vision_position_embedding`；该计算计划映射到LPU。

MTP需先补充支持它的执行reference，详见 [权重与参考范围说明](../../docs/model-numerical-baseline.md#11-模型权重与量化参数)。完整模型及FP8端到端reference随相应任务接入。
