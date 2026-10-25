# 给 GPU 会话：MiniMax M3 的 4D 并行矩阵（2026-10-25，CPU 会话写）

背景见 `REVIEW_M3_PRS_2026-10-25.md`。用户的顺序：5D 整个分支先开 draft 占位，这个矩阵跑完再开 FSDP2 基线。下面是要叠加到你手上工作里的部分。

--- PROMPT BEGIN ---

在 8 x A4000 上把 MiniMax M3 的 4D 并行矩阵补全，CP 不跑。

- **树：** `minimax_m3_5d` = `798f8817f`。不改模型代码。
- **kit：** `Raising_PRs/RFC_MINIMAX_M3/kit_m3_5d_2026-10-24/`，协议照 10-24：
  - 单卡写 seed，每格只加载模型；
  - 所有格子共用一份 cache，每格先预热 10 步，再测 20 步；
  - 报第 1 / 3 / 10 步的 loss 和 grad norm，第 1 步给全精度。
- **kit 要先改一处：** `m3_text18_probe.py` 的 `_degrees` 加 `dpr`，对应 `data_parallel_replicate_degree`；PP 的 micro-batch 数要除以 `dpr * dp`，不是只除以 `dp`。
- **约束：** MoE 要求 EP >= TP，所以开 TP 的格子 EP 关不掉。EP 开和关的对比只在 FSDP、HSDP、PP 的组合上做。
- **要补的格子：** 按数据流分组，每组只和自己的参照比；参照在每条数据流上跑两次，作 noise floor。
  - dp1（参照：单卡）：tp2 x ep2 x pp2，tp2 x ep2 x pp4，tp4 x ep4 x pp2。
  - dp2（参照：fsdp2）：
    - fsdp2 x pp2，ep2 x fsdp2 x pp2，fsdp2 x pp4，ep2 x fsdp2 x pp4；
    - fsdp2 x tp2 x ep2，fsdp2 x tp2 x ep4，fsdp2 x tp4 x ep4，fsdp2 x tp4 x ep8。
  - dp4（参照：fsdp4，新加）：
    - HSDP 2 x 2，ep2 x HSDP 2 x 2，ep4 x fsdp4；
    - fsdp4 x pp2，ep4 x fsdp4 x pp2，HSDP 2 x 2 x pp2；
    - fsdp4 x tp2 x ep2，HSDP 2 x 2 x tp2 x ep2。
  - dp8（参照：fsdp8，新加；10-24 的 `ep8 x fsdp8` 就和它配对）：HSDP 2 x 4，HSDP 4 x 2，ep4 x HSDP 2 x 4。
  - 布局跑不了（titan 拒绝，或者放不下）就跳过，把报错原文记下来。
- **判断：**
  - 同一条数据流里，只差 FSDP、HSDP、PP 的格子，第 1 步 loss 应该逐位相同；
  - 带 TP 或 EP 的格子，差异要能归到 10-24 已经定位的原因（row-parallel 的规约顺序、选块和路由的翻转、`grouped_mm` 的行数）；
  - 归不进去的差距，先在同一份 cache 上重跑、对第 1 步梯度定位，定位之前不写进正文。
- **CPU 全量：** `minimax_m3`、`minimax_m3_5d` 和 main（它们的 base `3ed5aa976`）各跑一遍 `tests/unit_tests/cpu`，报只在分支上失败的个数，应该是 0。
- **写回：**
  - 结果追加进 `M3_5D_2026-10-24.md`；
  - 5D draft 正文（`PR_BODY_M3_5D_2026-10-24.md`）的 Test plan 最后一条换成每条数据流一张表（第 1 / 3 / 10 步，带 noise floor 行），表下每个差距一条单行 bullet；
  - FSDP2 基线正文（`PR_BODIES_M3_FSDP_2026-10-24.md`）加一张 dp4 的表：fsdp4、HSDP 2 x 2 和 noise floor。HSDP 跑通了，基线正文、README 里的 HSDP 才保留，否则删掉。
- **不做：** 不开 PR，不推 PR 分支，不加新的 CI 格子。

--- PROMPT END ---
