# Case定义

小型JSON配置记录case编号、OP/K、数值profile、seed、参数及逐输出比较规则，通过 `pymodel.common.case.load_case` 读取。[OP-04示例](examples/op04-residual-bf16.json)与[GDN recurrent核心](operators/gdn-recurrent-bf16.json)已用于PyTorch reference检查；PTO路径随kernel接入。

[视觉位置插值元数据case](operators/vision-position-interpolation.json)验证OP-11前置阶段的`interp_indices`和`interp_weights`两个命名输出。

输入tensor由各算子用例显式构造，随机数使用 `case.generator()`。有效区域、尾块、非零状态及路由等场景随对应算子加入。使用方法见[验证框架](../docs/validation.md)。
