# Phase 14：多机 5D 并行实验

2026-10-28 建立（用户："开一个新的 logbook phase14 文件夹，叫multinode_5D_experiments"）。

放多机（跨节点）的 5D 并行实验：FSDP2 / HSDP、TP/SP、EP、PP、CP 组合的运行脚本、结果和说明。

在这之前，5D 的数值结果都是单机跑的，相关记录在：
- `Raising_PRs/RFC_MINIMAX_M3/`：MiniMax M3 在 8 x RTX 5060 Ti 和 4 x H100 NVL 上的矩阵；
- `Raising_PRs/PR_K3_PARALLELISM/`：Kimi K3 的并行 PR 和 H100 复测。

本 phase 的文件：
- `multinode.md`：用户放进来的纸面规划（K3 缩小版在 2 × 8 卡 H100 / H200 上的 5 种组合，§9 是 A100 80GB）；
- `kit/`：按规划准备的脚本，对应 titan 实验分支 `phase14_multinode_5d`（`ac36e017b`），加了 A100 80GB 的配置和 MiniMax M3 缩小版，说明见 `kit/README.md`。

约定（沿用前几个 phase）：
- 每次实验一个 kit 子目录（脚本、环境、结果），说明文件用中文、带日期；
- 数值表遵守 CLAUDE.md 的数值表和验收规则：同一数据流配对，共用一份预热过的编译 cache，每张表带 noise floor 行。
