# Qwen3.6-35B-A3B-FP8 模型与数值基线

确认日期：2026-11-10；基线ID：`qwen3_6_35b_a3b_fp8-v1`。本基线供两人共同实现reference、PTO kernel及逐级验证。当前固定模型配置、checkpoint版本和来源代码，并保存可核对的数值规则。

## 文件位置与事实来源

模型专属配置集中在 [configs/models/qwen3_6_35b_a3b_fp8/](../configs/models/qwen3_6_35b_a3b_fp8/)。[baseline.json](../configs/models/qwen3_6_35b_a3b_fp8/baseline.json)记录版本与结构，[numerics.json](../configs/models/qwen3_6_35b_a3b_fp8/numerics.json)记录计算规则。`data/`继续管理OP/K/接口登记及依赖入口；`cases/`后续描述某次具体输入与场景；`artifacts/`保存运行结果。

| 对象 | 固定版本 / 来源 |
| --- | --- |
| 模型与权重 | [Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8/tree/c653bba46dfa06e9658dcc0b0dc87fe4685f14a2)；revision `c653bba46dfa06e9658dcc0b0dc87fe4685f14a2` |
| 官方模型配置 | [config.json](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8/resolve/c653bba46dfa06e9658dcc0b0dc87fe4685f14a2/config.json)，本地保存原始字节及SHA-256 |
| 官方生成默认值 | [generation_config.json](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8/resolve/c653bba46dfa06e9658dcc0b0dc87fe4685f14a2/generation_config.json)，本地保存原始字节 |
| 模型源码 | [Transformers `f90e93c4f839`](https://github.com/huggingface/transformers/blob/f90e93c4f83974f634860e246cdfd2c68666d605/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py) |
| FP8加载/运算集成 | [finegrained_fp8](https://github.com/huggingface/transformers/blob/f90e93c4f83974f634860e246cdfd2c68666d605/src/transformers/integrations/finegrained_fp8.py)、[quantizer](https://github.com/huggingface/transformers/blob/f90e93c4f83974f634860e246cdfd2c68666d605/src/transformers/quantizers/quantizer_finegrained_fp8.py) |
| 权重索引与header | [固定权重索引](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8/resolve/c653bba46dfa06e9658dcc0b0dc87fe4685f14a2/model.safetensors.index.json)；全部42个header已和索引逐key核对 |

checkpoint记录的`transformers_version`为`4.57.1`，所固定源码自身版本为`5.22.0.dev0`。源码commit是本基线的参考依据；配置中的版本字符串保留为checkpoint元数据。后续运行还需记录实际Torch、依赖kernel、device和后端版本。

## 模型结构

官方配置的实现类名为 `Qwen3_5MoeForConditionalGeneration`；这里按该实现族读取已固定的 Qwen3.6 checkpoint，模型身份仍由仓库名和revision确定。

| 模块 | 固定内容 |
| --- | --- |
| 文本主干 | hidden2048，40层；[GDN,GDN,GDN,Full Attention]×10；每层MoE；前置Norm与两次残差 |
| Full Attention | Q16/KV2，head_dim256，GQA8；Q/gate投影8192；每head内部512拆256+256；Partial MRoPE旋转64维 |
| GDN | Q/K16 heads、V32 heads，均128维；qkv8192、z4096、a/b各32；Conv核4且无bias；FP32 State[B,32,128,128]；chunk默认64 |
| MoE | 256个routed专家，Top-8；1个shared专家；中间维度512；shared gate每Token一个标量 |
| Vision | 27 blocks，hidden1152，16 heads/head_dim72；Patch[3,2,16,16]；MLP1152→4304→1152 |
| Vision Merger | 1152维LayerNorm→相邻2×2 Patch拼接4608→Linear4608→GELU→Linear2048 |
| 输出 | 词表248320；Embedding和LM Head权重独立 |
| MTP | 配置为1层；checkpoint包含1560个MTP张量，fc权重[2048,4096]；执行reference范围见后文 |

## 存储dtype与运算dtype

已读取42个safetensors header，共64,196个张量；未加载张量payload。header及索引SHA保存在[checkpoint_metadata.json](../configs/models/qwen3_6_35b_a3b_fp8/checkpoint_metadata.json)。

| 对象 | 核对结果 / 执行规则 |
| --- | --- |
| checkpoint FP8张量 | 31,745个，safetensors dtype为F8_E4M3；对应Torch float8_e4m3fn |
| checkpoint BF16张量 | 32,451个，其中31,745个是weight_scale_inv；其余706个为非scale张量 |
| Embedding / LM Head / Vision | checkpoint为BF16；各层Norm、Router、GDN a/b、Conv等按相应张量元数据读取 |
| weight_scale_inv存储 | 全部31,745个为BF16，形状与128×128分块逐一匹配 |
| scale运行时 | 固定FP8Linear在默认scale_fmt=float下声明FP32参数；加载后记录实际dtype并保留checkpoint scale值 |
| GDN State与核心递推 | FP32；A_log/dt_bias存为BF16并不改变源码显式float()后的运算规则 |

基础BF16、权重FP8、scale存储dtype与归约/状态FP32分别记录。Reference保留官方源码里的cast位置；额外的全FP32排错路径须使用独立profile名称。

默认KV cache保持K/V投影输出的dtype，在目标基础profile中为BF16；另行启用KV量化时记录独立case配置。Token ID与Torch TopK索引以int64表示，后端使用uint32等存储形式时显式检查范围并转换，比较其逻辑索引值。

## 算子数值规则

| 计算 | 基线规则 |
| --- | --- |
| 普通RMSNorm | eps=1e-6；输入转FP32，计算平方均值和rsqrt，乘1+weight.float()，最后转回输入dtype；用于Decoder、Q/K与Final Norm |
| GDN输出gated Norm | 在128维做FP32平方均值归一化→先转回输入dtype→乘直接weight→乘SiLU(z.float())→转回输入dtype；没有+1 |
| Vision LayerNorm | eps=1e-6，最后1152维、带weight和bias，方差correction=0；具体累加实现随Torch/device记录 |
| Residual | 保留对应残差源，以源码张量dtype执行加法；两次残差分别使用各自输入 |
| Full Attention | score缩放1/sqrt(256)；因果mask；eager Softmax在FP32计算再转query dtype；Q/K先Norm再RoPE，K此后进入cache |
| Text MRoPE | theta=10,000,000；旋转64维，其余192维保留；THW分组[11,11,10]；频率/cos/sin在FP32计算后转输入dtype；rotate_half采用half-split配对 |
| GDN Q/K | repeat_interleave到32 heads；FP32 L2Norm使用sum(x²)+1e-6；Q额外乘1/sqrt(128)，只缩放一次 |
| GDN门控与状态 | beta=sigmoid(b)，核心再转FP32；g=-exp(A_log.float())×softplus(a.float()+dt_bias)；按衰减、预测、修正、更新、读出顺序维护FP32 State |
| GDN chunk | 默认64；Q/K/V/beta/g尾块补0；块末State依次传递；与recurrent比较输出和最终State，浮点结果不要求逐bit一致 |
| MoE Router | FP32 Softmax→Top-8→选中权重重归一化→转router_logits dtype；同值TopK顺序未由模型指定 |
| Routed / Shared FFN | SiLU(gate)×up后down；routed输出乘对应路由权重再合并；shared使用独立权重和sigmoid标量gate |
| Vision Attention | 每有效片段内非因果MHA，缩放1/sqrt(72)；axial RoPE使用完整72维，显式FP32计算后转回Q/K原dtype |
| Vision位置 | 学习表[2304,1152]、二维bilinear插值、align_corners=True；按Patch顺序组织并转hidden dtype后相加 |
| Vision激活 | block MLP为GELU(tanh)；Merger为GELU exact，均含两次线性投影的bias |
| Embedding / Scatter / Head | 保持Token、视觉行和placeholder对应；Head为无bias线性投影，模型forward不无条件把logits转FP32 |

逐项来源链接和完整cast顺序保存在[numerics.json](../configs/models/qwen3_6_35b_a3b_fp8/numerics.json)。Vision配置未显式给出的axial RoPE theta由固定配置类与[RoPE默认值](https://github.com/huggingface/transformers/blob/f90e93c4f83974f634860e246cdfd2c68666d605/src/transformers/modeling_rope_utils.py#L753-L799)解析为10,000。

## FP8与量化边界

权重按[out_features,in_features]的128×128 block组织。对于weight[N,K]，scale形状为[ceil(N/128),ceil(K/128)]；全部31,745对权重/scale的shape已核对。概念恢复关系为：`W_real[n,k] = float(W_fp8[n,k]) * float(weight_scale_inv[n//128,k//128])`。保留checkpoint提供的scale值，不能根据`_inv`名字再取倒数，也不能用重新计算的absmax scale替代原值。

官方配置保留648条不量化清单，包含别名；固定loader会正规化module名称。完整清单在[config.json](../configs/models/qwen3_6_35b_a3b_fp8/config.json)中。激活为动态量化；固定集成将运算交给DeepGEMM或finegrained-fp8等kernel。实际选择的kernel revision、device、动态scale/舍入/clip和GEMM累加行为随执行case记录。

CPU或不具备目标能力的设备上，官方loader可能自动反量化到BF16。此路径可标记为反量化排错profile；FP8验收必须记录实际量化执行路径，不能把该回退结果登记为原生FP8结果。

## MTP参考范围

checkpoint确实含MTP权重：fc、前置两路Norm、输出Norm为BF16，MTP Attention/专家投影中包含FP8。固定Transformers源码在[PreTrainedModel](https://github.com/huggingface/transformers/blob/f90e93c4f83974f634860e246cdfd2c68666d605/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L993-L1027)里忽略`^mtp.*`的unexpected keys，没有提供MTP前向实现。因此当前主干/视觉源码不能直接用作MTP golden。

MTP任务开始前需另锁定支持MTP的reference，确认两路h/e位置、Norm权重解释、共享参数及draft状态规则；本基线先固定其配置和checkpoint元数据。OP-15/16及相关kernel仍为planned。

## 运行与比较记录

共同采用eval/inference语义，保留源码的dtype/cast。Attention数学reference采用固定eager路径；GDN采用固定Torch fallback recurrent/chunk；开启fused、FlashAttention或FP8专用kernel时记录对应版本和选择方式。官方[generation_config.json](../configs/models/qwen3_6_35b_a3b_fp8/generation_config.json)保持原样；greedy回归等覆盖作为case参数显式记录。

每次比较记录输出名、shape/dtype、最大绝对/相对误差与最坏位置；路由索引、KV、State和Conv历史分别比较。整数及唯一确定的索引精确检查；同值TopK按固定后端语义处理。浮点atol/rtol随首个实际OP/后端/精度用例校准，当前为null，尚无统一的FP8容差承诺。

当前交付为配置与规则整理：来源hash、全部分片header和结构一致性已核对；尚未实现golden、运行完整权重模型或执行PTO/LPU数值验收。
