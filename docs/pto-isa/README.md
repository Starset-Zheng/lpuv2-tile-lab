# PTO Tile ISA参考入口

上游：<https://github.com/hw-native-sys/pto-isa>；固定提交：`90a3796597a5fabc2ce5066e73f785c2e85822f7`。

- [上游手册](https://github.com/hw-native-sys/pto-isa/blob/90a3796597a5fabc2ce5066e73f785c2e85822f7/docs/PTO-Virtual-ISA-Manual.md)
- [逐指令参考](https://github.com/hw-native-sys/pto-isa/blob/90a3796597a5fabc2ce5066e73f785c2e85822f7/docs/isa/README.md)
- [CPU模拟说明](https://github.com/hw-native-sys/pto-isa/blob/90a3796597a5fabc2ce5066e73f785c2e85822f7/docs/coding/cpu_sim.md)
- [本仓库机器可读目录](../../data/pto_isa_catalog.json)

目录按namespace+参考条目统计：139个core参考条目对应136个core公开API名；另有12个通信接口。TMATMUL_MX_HIF4、TQUANT_DN、TQUANT_HIF4为单独说明的重载语义变体；TGATHER/TSCATTER区分本地与pto::comm。

目录中的实现路径表示固定版本的源码证据。当前仓库没有登记已执行的算子验证。完整中文总结：[PTO Tile ISA 指令集参考](https://tju-opentpu.feishu.cn/docx/GFfBdiCNLokRTsxmUbPcVHYankk)（飞书访问权限按文档设置）。
