# 给 GPU 会话：M3 两个分支 rebase 到最新 main（2026-10-26，CPU 会话写）

来由：`REVIEW_M3_PRS_2026-10-25.md` 第二次审核第 1 条。main 10-27 合了 #5583（Tied RoPE cache），两个分支在 `torchtitan/models/common/decoder.py` 和 main 冲突。

--- PROMPT BEGIN ---

把 MiniMax M3 的两个分支 rebase 到 upstream main 的最新提交，重跑检查。

- **冲突：**
  - 只在 `torchtitan/models/common/decoder.py` 的 import 块：#5583 加了 `from torchtitan.models.common.rope import RoPE`，P1b（`a48507a84`）加了 `from torchtitan.models.common.norm import OffsetRMSNorm`。两行都留。
  - 其余文件 `git merge-tree` 显示可以自动合并。rerere 关着，只解冲突段。
- **核对：**
  - 每个提交的增删行和 rebase 前逐行相同（patch-id 不同时，比增删行）；
  - #5583 让参数相同的 RoPE 共用一份 cache。M3 的 indexer 用注意力自己的 RoPE，会被共享，按 #5583 的说法数值不变。
- **重跑：**
  - CPU 全量：两个分支和新 main 各一遍，报只在分支上失败的个数，应该是 0。main 新加了 `tests/unit_tests/cpu/models/common/test_rope_cache_tying.py`。
  - `tests/unit_tests/gpu/models/minimax_m3`；
  - 集成格子：基线 `minimax_m3_mm_fsdp`，4D `minimax_m3_mm_fsdp+tp+sp+ep`。
  - 对拍脚本（bf16 默认）跑一次，和基线正文 Results 第一张表的第一行比；fsdp4 跑 3 步，和 HSDP 表里 fsdp4 的第 1 步比。不同就先定位，不改正文。
- **推送：**
  - `minimax_m3` 还没有 PR：先把旧 head 推成 fork 上的 backup 分支，再 force-with-lease。
  - `minimax_m3_5d` 是 #5619 的 PR 分支：按规则先推 review 分支，用户点头后再同步。新开 review 分支要先问用户，推之前停下来问。
- **正文：**
  - 两份正文 Test plan 的 CPU 计数换成新 main 上的结果；
  - 状态区记下新的 head 和 main 的提交号。

--- PROMPT END ---
