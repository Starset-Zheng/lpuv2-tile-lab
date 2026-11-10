# LPUv2 Tile Lab

面向 Qwen3.6-35B-A3B 的算子协作工程：建立 PyTorch golden，使用 PTO 编写 Tile kernel，逐算子与模型验证，并据此完善 LPUv2 Tile ISA 与 programming model。

**当前阶段：仓库框架。** 已登记16条模型OP、37个kernel候选、A/B分工及固定版本PTO指令索引。所有算子实现和数值验证状态均为 `planned`，源码目录以说明文件标明职责。

## 从这里开始

1. 阅读 [目录分层](docs/architecture.md)，确定自己修改的层。
2. 在 [两人分工](docs/work-allocation.md) 中认领OP及kernel。
3. 按 [开发流程](CONTRIBUTING.md) 先确认计算与接口，再提交golden和Tile实现。
4. 查阅 [PTO参考入口](docs/pto-isa/README.md) 和 [机器可读指令目录](data/pto_isa_catalog.json)。

```bash
git clone --recurse-submodules https://github.com/cxjiang-bk/lpuv2-tile-lab.git
cd lpuv2-tile-lab
# 已经普通clone时：
git submodule update --init --recursive
```

PTO固定在 `90a3796597a5fabc2ce5066e73f785c2e85822f7`，来源为公开 [hw-native-sys/pto-isa](https://github.com/hw-native-sys/pto-isa)。当前PTO目录覆盖139个core参考条目（136个core公开API名）及12个通信接口。语义变体、兼容名和命名空间重名在目录中分别记录。

## 模型与数值基线

模型固定为 **Qwen/Qwen3.6-35B-A3B-FP8**。共同基线放在 [configs/models/qwen3_6_35b_a3b_fp8/](configs/models/qwen3_6_35b_a3b_fp8/)，包含官方配置快照、模型/源码revision、checkpoint dtype元数据和数值规则。两人实现reference前阅读 [模型与数值基线](docs/model-numerical-baseline.md)。当前尚未实现golden或完成模型数值验证。

## 目录

```text
pymodel/             Python侧职责分层，参考pymodel的组织方式
  common/            shape、dtype、layout、状态等公共契约
  isa/               PTO调用适配；后续接入候选LPU ISA
  semantic/          后续候选LPU指令的独立语义定义
  ops/               算子入口与kernel选择
  ref/               独立PyTorch golden
  models/qwen36/     子图、Decoder layer和模型集成
  task_gen/          可复现case及实验产物组织
  utils/             输出、索引、状态比较等公共工具
kernels/pto/         原生PTO Tile kernel
tests/               指令、kernel、算子、子图和模型验证
cases/               测试场景说明与小型case定义
configs/             模型基线和后续运行配置
data/                OP、kernel、接口及PTO指令登记
docs/                架构、协作、验证和ISA需求记录
third_party/pto-isa/  固定版本上游子模块
artifacts/           本地运行产物，默认不入库
```

## 后续闭环

PyTorch golden → 实际PTO CPU程序 → 同输入比较输出/索引/状态 → 子图与模型接入 → LPUv2能力需求。执行环境、精度和目标后端随具体kernel记录；当前尚未接入编译、算子运行或CI测试入口。
