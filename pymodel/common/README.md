# pymodel/common

[case.py](case.py)提供 `Case`、`OutputCheck` 和 `load_case`：固定case编号、seed、数值profile、参数及命名输出的比较规则。`case.generator()`返回独立CPU随机数生成器。

Tensor/Tile的shape、dtype、layout、stride、valid region及状态更新语义随实际算子接口完善。使用方法见[验证框架](../../docs/validation.md)。
