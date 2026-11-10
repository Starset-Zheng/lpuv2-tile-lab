# Case定义

小型JSON配置记录case编号、OP/K、数值profile、seed、参数及逐输出比较规则，通过 `pymodel.common.case.load_case` 读取。示例见 [op04-residual-bf16.json](examples/op04-residual-bf16.json)，对应算子尚待实现。

输入tensor由各算子用例显式构造，随机数使用 `case.generator()`。有效区域、尾块、非零状态及路由等场景随对应算子加入。使用方法见[验证框架](../docs/validation.md)。
