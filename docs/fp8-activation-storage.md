# FP8 激活存储、量化融合与复用

状态：LPUv2 候选方案，待算子实现与验证。模型 reference 遵循 [模型与数值基线](model-numerical-baseline.md)，本文讨论该基线之上的数据流优化。

## 按消费者选择激活表示

可以为不同算子的输入选择 BF16 或 FP8 表示。选择依据是数据流中消费者的完整输入契约，包括是否量化、分组、scale 和舍入位置。FP8 权重 Linear 的输出仍可为 BF16，随后进入 Norm、门控或残差计算；因此，激活精度需要标在具体的数据流边上。

保存量化激活时，逻辑对象至少包含：

```text
QuantizedActivation = {
    data: FP8 payload,
    scale: scale payload,
    format, quantization_group_axis_and_size, scale_convention,
    logical_shape, layout, valid_region,
    source_rounding_boundary
}
```

消费者根据约定使用 scale 恢复实数值，例如乘法约定为 `x_approx = float(data) * scale`。具体 scale dtype、分组及乘数/倒数约定与执行 kernel 一起固定；裸 FP8 payload 无法独立表示原激活。

固定 Transformers FP8 Linear 接口接收 BF16 / FP16 激活。直接消费预量化的 FP8 + scale，需要后端提供对应入口，并明确输出 dtype；实现时一并定义这个契约。

## 能省掉哪些工作

| 方式 | 数据流 | 作用 |
| --- | --- | --- |
| 基线 | 生产者输出 BF16 → 消费者动态量化 → FP8 GEMM | 保留现有参考计算路径 |
| 量化融合到生产者 | 生产者完成 BF16 舍入并计算 scale、量化 → 输出 FP8 + scale → 消费者直接 GEMM | 可减少 BF16 写回与重读，以及独立量化步骤的调度；动态 scale 计算与量化仍然发生 |
| 多个消费者复用量化输入 | 同一个 BF16 输入只量化一次 → 多个兼容量化契约的 GEMM 共享 FP8 + scale | 减少重复的 scale 计算与量化；分组或舍入契约不兼容时分别处理 |

若生产者只持有局部 Tile，而量化分组跨多个 Tile，则需要先完成该组的统计、保留相关数据或增加一遍读取。能否节省搬运量，要结合 Tile 划分、统计归约和数据生命周期判断。按当前输入取动态 scale 时，这部分工作需要在某处执行；换为校准后的静态 scale 属于另一套量化方案，需要另行验证。

## Qwen3.6 中的适用位置

| 位置 | 输入的消费者 | 候选实现 |
| --- | --- | --- |
| Full Attention 的 Norm 输出 | Q/gate、K、V 均为 FP8 权重投影 | 核对输入分组、格式和 scale 约定后，复用同一份量化输入 |
| GDN 的 Norm 输出 | qkv/z 为 FP8 权重投影；a/b 为 BF16 权重投影 | BF16 值供 a/b 使用，FP8 + scale 供兼容的 qkv/z 使用；可以融合生产这两种表示 |
| MoE 的 Norm 输出 | Router、shared 标量门控使用 BF16；专家投影使用 FP8 | 路由保留 BF16 输入；专家之间可复用兼容的量化表示。Dispatch / Gather 同时维护 payload 与 scale 的对应关系 |
| 专家 `SiLU(gate) * up` 输出 | 下游为 FP8 Down 投影 | 适合把 Down 的输入量化融合到乘法输出阶段，同时保留原有 BF16 舍入顺序 |
| Vision、Residual、Norm、KV / GDN State | 各自遵循基线的 BF16 / FP32 契约 | 按实际后续计算保留相应表示；状态量化作为独立方案评估 |

混合分支中的 BF16 输入可在相关消费者使用完毕后释放，无需长期保存两份。把 FP8 数据重新反量化为 BF16 只能得到量化后的近似值，无法恢复原 BF16 输入；例如这样替换 Router 输入可能改变专家索引。

## 保留数值边界

将量化融合到生产者时，保留原有回转与舍入顺序：

```text
# 原路径
y_bf16 = BF16(producer_result)
(q, s) = dynamic_quantize(y_bf16)

# 融合路径：BF16 值可留在寄存器中，不必单独写回
(q, s) = dynamic_quantize(BF16(producer_result))
```

直接量化 `producer_result` 的高精度值会绕过一次 BF16 舍入，可能得到不同的 scale 或 FP8 值。跨 Norm、非线性、残差移动量化位置也会改变误差传播。后续将每种融合 / 复用方式登记为 kernel 候选，先比较量化 payload 与 scale，再比较算子输出、路由索引及 KV / GDN 状态；浮点容差随实际后端用例校准。

这些尝试将为 programming model 提供具体需求：量化 Tile 与 scale 的关联、不同消费者的表示需求、多输出、量化分组与 Tile 有效区域的关系，以及数据复用期间的生命周期。候选方案通过验证后，再确定对应的 LPUv2 指令或组合方式。
