# 验证框架自身的检查

运行 `python -m pytest tests/framework -q`。

这里用小型 fixture 检查逐输出比较、BF16 / FP32 / 索引精度、状态隔离、输入别名、异常及复现文件。fixture 的 backend 和 numerical_profile 均为 `framework_fixture`，不登记为模型算子或 PTO kernel 的验证结果。
