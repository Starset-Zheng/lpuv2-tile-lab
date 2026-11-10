# pymodel/utils

[compare.py](compare.py)提供 `compare_outputs(actual, expected, checks)`。所有输出按名字比较：浮点逐元素应用显式容差，整数/索引精确检查，并报告shape、dtype、最大误差、最坏坐标和不匹配比例。

使用方法与非有限值策略见[验证框架](../../docs/validation.md)。
