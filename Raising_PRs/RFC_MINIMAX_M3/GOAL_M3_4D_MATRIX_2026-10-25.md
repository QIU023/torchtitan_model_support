# MiniMax M3 4D 并行矩阵（2026-10-25 晚的 goal）

## 来由

- 用户把 CPU 会话的 `GPU_PROMPT_M3_4D_MATRIX_2026-10-25.md` 接在 `GOAL_PR_READY_2026-10-25.md` 后面设成了 goal，然后说"已经完成的旧goal帮我去掉，重新给"。
- 去掉的部分（都已完成，记录在 `CHECKS_2026-10-25.md`）：
  - 旧 goal 的六条：两个分支已 force-push，`minimax_m3` = `9a5ab9ebe`，`minimax_m3_5d` = `89fc094d1`；
  - prompt 里的 CPU 全量：三棵树都跑过，只在分支上失败的是 0；
  - prompt 里 probe 加 `dpr` 那一处。
- 和 prompt 原文的出入：
  - 树写成现在 fork 上的 `89fc094d1`。代码和 `798f8817f` 相同，只差 README、docstring 和一行注释的文字。
  - 机器换成本机 8 x RTX 5060 Ti，kit 用已经加了 `dpr` 的 `kit_m3_5060_2026-10-25/`。
  - 10-24 在 A4000 上跑过的格子也在本机重跑。每条数据流的表必须来自同一台机器、同一份 cache，A4000 的数不能和 5060 的数放在一起比。
  - 写回按用户 10-25 的决定："基线正文的 HSDP 表写 Pending (H100)"，以及 memory 规则"正文里的 loss / grad 表要 H100"：
    - 基线正文不动；
    - 5D 正文只写覆盖情况，表留在 logbook。
    - 用户要在 5D draft 里直接贴表的话，改写回的第二条。

## goal 文字

```
完成

MiniMax M3 的 4D 并行矩阵在本机（8 x RTX 5060 Ti）补全，写回 logbook 和 5D draft 正文，CP 不跑。

- 树：minimax_m3_5d = 89fc094d1（fork 上的 head，代码和 798f8817f 相同，只差 README 和 docstring 文字）。不改模型代码，不建新提交。
- 环境：venv_1025；kit 用 Raising_PRs/RFC_MINIMAX_M3/kit_m3_5060_2026-10-25/。
  - probe 已加 dpr，PP 的 micro-batch 数按 dpr x dp 算；
  - shim/sitecustomize.py 让 sm_120 不选 FA4（本机没有 flash_attn）。
- 协议照 10-24 的 kit_m3_5d_2026-10-24/run_m3_text18.sh，写一份 5060 版 runner：
  - c4_test 纯文本，seq 512，每步 4096 token，AdamW，seed 42，deterministic；
  - 单卡写一个 seed checkpoint，所有格子只加载它；
  - 所有格子共用一份 cache，放在运行目录下面；
  - 每格先串行预热 10 步，正式的 20 步在预热后 cache 的副本上跑，可以并行；
  - 报第 1 / 3 / 10 步的 loss 和 grad norm，第 1 步给全精度；非 PP 格子加第 1 步梯度 dump。
- 约束：MoE 要求 EP >= TP，开 TP 的格子 EP 关不掉；EP 开和关的对比只在 FSDP、HSDP、PP 的组合上做。
- 格子：按数据流分组，每组只和自己的参照比，参照跑两次作 noise floor。10-24 在 A4000 上跑过的格子也在本机重跑，和新格子用同一份 cache。
  - dp1（参照：单卡）：pp2，pp2 x vpp2，pp4，pp8，tp2 x ep2，tp4 x ep4，tp2 x ep2 x pp2，tp2 x ep2 x pp4，tp4 x ep4 x pp2。
  - dp2（参照：fsdp2）：ep2 x fsdp2，fsdp2 x pp2，ep2 x fsdp2 x pp2，fsdp2 x pp4，ep2 x fsdp2 x pp4，fsdp2 x tp2 x ep2，fsdp2 x tp2 x ep4，fsdp2 x tp4 x ep4，fsdp2 x tp4 x ep8，ep2 x fsdp2 x tp2 x pp2，ep4 x fsdp2 x tp2 x pp2。
  - dp4（参照：fsdp4）：HSDP 2 x 2，ep2 x HSDP 2 x 2，ep4 x fsdp4，fsdp4 x pp2，ep4 x fsdp4 x pp2，HSDP 2 x 2 x pp2，fsdp4 x tp2 x ep2，HSDP 2 x 2 x tp2 x ep2。
  - dp8（参照：fsdp8）：ep8 x fsdp8，HSDP 2 x 4，HSDP 4 x 2，ep4 x HSDP 2 x 4。
  - titan 拒绝或放不下的布局跳过，把报错原文记下来。
- 判断：
  - 同一条数据流里只差 FSDP、HSDP、PP 的格子，第 1 步 loss 应该逐位相同；
  - 带 TP 或 EP 的格子，差异要归到 10-24 已定位的原因（row-parallel 的规约顺序、选块和路由的翻转、grouped_mm 的行数）；
  - 归不进去的差距，先在同一份 cache 上重跑，再用第 1 步梯度定位（规约顺序的差用 kit 里 reduce_emulate.py 那套）；定位之前不写进任何正文。
- 写回：
  - 每条数据流一张表（第 1 / 3 / 10 步，带 noise floor 行），表下每个差距一条，再加跳过的布局和报错原文，追加进 M3_5D_2026-10-24.md；
  - 5D draft 正文（PR_BODY_M3_5D_2026-10-24.md）的 Test plan 最后一条改成覆盖情况：跑了哪些布局，只差 FSDP/HSDP/PP 的格子第 1 步是否逐位，TP/EP 的差是否都已定位，标 8 x RTX 5060 Ti；loss 表照 10-25 的决定写 Pending (H100)，正文不贴 5060 的数；
  - 基线正文不动：HSDP 已跑通，结果保持 Pending (H100)；哪个 HSDP 组合跑不通，只记报错，正文不声称它；
  - logbook 提交前先 fetch（CPU 会话也在改这几个文件），提交后推送。
- 不做：不开 PR，不推 minimax_m3 / minimax_m3_5d，不加 CI 格子，正文不带 Claude 署名。这台 5060 只由执行这个 goal 的会话使用。
```

## 进度（10-25 晚，5060 会话）

- **矩阵：完成。**
  - 36 个布局（10-24 的 12 格在本机重跑，另加新的 24 格）全部跑满 20 步，titan 没有拒绝任何一个；四个参照各跑两次，20 步全同。
  - 每条数据流的表、判断和定位追加在 `M3_5D_2026-10-24.md` 的"4D 矩阵"一节。
- **判断：**
  - 只差 FSDP、HSDP、PP 的格子，第 1 步 loss 全部逐位；只开 EP 不开 TP 的格子也全部逐位，而且 EP 只改变 routed experts 的梯度。
  - TP 的差，用强制单卡的选块和路由在本机复现了 10-24 的归因：tp2 x ep2 从 2.79e-2 降到 5.39e-3，tp4 x ep4 从 3.93e-2 降到 5.73e-3。
  - 一处没有定位：tp2 x ep2 x pp2 / pp4 和 tp2 x ep2 之间差 1 个 fp32 单位。这是两个 TP 格子之间的差，不进正文；用 static shapes 复查时撞到了 flex 自己的 `recompile_limit=16`。
- **写回：**
  - 5D 正文 Test plan 最后一条换成覆盖情况，表写 Pending (H100)；
  - 基线正文没动；
  - 可以直接贴的英文表在 `M3_4D_TABLES_EN_2026-10-25.md`。
- CPU 全量白天已经完成，今晚没有改代码，不重跑。

**用户 10-25 晚再次确认（goal 原文和之前的决定冲突时问的）：**
- 基线正文的 dp4 结果保持 "Pending (H100)."；
- 5D draft 的 Test plan 最后一条保持覆盖情况加 "Loss and grad norm tables: Pending (H100)."。
- 正文都不贴 5060 的表；四张表和 dp4 表只留在 `M3_4D_TABLES_EN_2026-10-25.md` 和 `M3_5D_2026-10-24.md`。
- 这个 goal 就此完成。
