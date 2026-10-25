# MiniMax M3 开 PR 前的目标（2026-10-25，5060 会话接手）

## 来由

用户 10-25："现在minimax m3的部分有最新的进展，但是中间换机器了，现在接手重新读取 然后记录这个目标，帮我给出goal的文字"。下面是用户给的对照和待办，原样记录。依据是 CPU 会话的 `REVIEW_M3_PRS_2026-10-25.md`。

> **和 #4025 合入标准的对照**
>
> * 复用 titan 组件：达到。 MSA 是 `GQAttention` / `FlexInnerAttention` 的子类。router、`RoutedExperts`、`ClampedSwiGLU`、`VisionTransformerBlock`、`AuxLoss`、DSV3 adapter 的 seam、common 的 dummy 视觉输入都是复用。代码里没有 hook、全局状态或换 `__class__`。
> * HF 对拍：超过。 固定 revision 拿 config，默认 bf16，另加 fp32；强制用 HF 的选块和路由后全位置 KL 4.1e-5，残差也定位了。
> * 单测：超过。 #4025 合入时只有 3 个；M3 有 CPU 16 个、adapter 2 个、GPU 2 个。GPU 测试是单卡的，main 现在的 CI 会自动跑，不用像 #4025 那样改 workflow。
> * 集成格子：达到。 #4025 自己就加了 `kimi_k3_mm_fsdp`，M3 加 `minimax_m3_mm_fsdp` 有先例。
> * 注释密度：相当。 #4025 合入的 diff 是 6.2%，M3 是 7.1%（都含 license 头）。
> * 维护者在 #4025 上提过的其他要求都满足： 层号从 0 起，debug recipe 用默认并行，支持 flex，支持样本打包，indexer 在第一个 PR 就有 KL 梯度，纯文本 batch 也跑 dummy 视觉前向。
> * 并行验证：缺。 #4025 给了单卡、FSDP、HSDP 各 200 步的 loss；M3 基线正文里没有，只有 4 卡 FSDP 格子跑了 10 步。HSDP 在 logbook 里一次都没跑过，但正文、README、5D 正文都写了支持。
>
> **开 PR 前要做**
>
> 1. 补 HSDP，或者去掉。 按数值表规则，在同一条数据流上配对：fsdp4 对 HSDP 2 x 2（都是 dp4），加一行 noise floor，报第 1 / 3 / 10 步。跑不了就把三处 HSDP 都删掉。这一条要 GPU。
> 2. CPU 全量。 正文只跑了指定的几个文件，CI 会跑整个 CPU 套件。prep 提交动了 Qwen3.5 / 3.6 / 3.8，5D 动了 Kimi K2.7 / K3 的 sharding，两个分支都要各跑一遍全量，和 main 比只在分支上失败的应该是 0。
> 3. 只开两个 PR 的话，正文要改。 现在的写法是先开三个 prep PR，M3 正文写的是 "Stacked on #<P1> #<P2> #<P3>"。如果 prep 提交留在基线 PR 里：
>    * 要照 #4025 加一节 "Changes outside `models/minimax_m3/`"，每个 prep 提交一条；
>    * 5D 正文的第一行也要跟着改。
> 4. flex 的 inductor bug。 正文和代码注释都断言 PyTorch 的 flex backward autotune 绑错了捕获张量的顺序，但 issue 还没发，而 flex 的作者 drisspg 就在 titan 评审。建议先用 logbook 里的最小复现发 issue 再链接过去，或者把这句改成只讲约束。发 issue 是对外动作，你定。
> 5. "only the index projections" 不准。 README 和 docstring 都这么写，但 KL 也训两个 index norm。两个分支都还没开 PR，amend 基线提交、再把 5D 的三个提交 rebase 上去即可，force-push 前先推备份。
> 6. 补一节 "Current scope"，照 #4025 写。
>    * 写明支持和不支持的并行，以及视频和 compile 的情况；
>    * 写明 MSA 的选块和 KL 目前是 PyTorch 按 128 个 query 一块循环算的，带 host 同步，融合 kernel 是后续工作。
>    * 维护者在 #4025 上要过 "training ready version using kernels"，先写明比被问到好。
>
> 5D 正文是按规则写的：数据流标明了，有 noise floor，PP / EP / TP 的差都定位了。作为 `[DO NOT MERGE]` 的占位可以开，只要跟着改 HSDP 和第一行那几处。

## 接手时核对的现状（10-25，5060 本机，用 HTTPS 读 fork）

- **分支**（和 10-24 推送时相同，之后没人动过）：
  - `minimax_m3` = `de7806dae`，基于 upstream `3ed5aa976`。往上依次是：
    - P1 `9172a00f1`（OffsetRMSNorm 进 common）；
    - P2 `64be06102`（`RoPE` 支持部分旋转）；
    - P3 `69dda3713`（DSV3 adapter 的 seam）；
    - P1b `a48507a84`（norm 类型）；
    - 基线 `de7806dae`。
  - `minimax_m3_5d` = `798f8817f`：P4、P5（sharding config 进 common），然后 TP/SP、EP、PP。
  - 两个 worktree 原来在 A4000 机器上；本机只有 fetch 下来的 `refs/remotes/fork_https/minimax_m3*`，开工前要建 worktree。
- **HSDP 写在三处：**
  - README 第 46 行 `| FSDP2 / HSDP | Supported |`；
  - 基线正文 Design 里的 "FSDP2 and HSDP."；
  - 5D 正文的 "composes FSDP2/HSDP"。
- **"index projections"：** README 第 14 行、`attention.py` 第 515 行（`MiniMaxSparseAttention` 的 docstring）。
- **正文占位：** 基线正文是 "Stacked on #<P1> ... #<P2> ... #<P3>"，5D 正文第一行和 Summary 里是 `#<FSDP2>`。
- **flex bug 的材料：** `flex_bwd_capture_order/`（最小复现、spy、补丁）和 `PARITY_HF_2026-10-24.md`。
- **本机条件：**
  - 8 张 RTX 5060 Ti，HSDP 2 x 2 对 fsdp4 能在这里跑；
  - 要先确认 main `3ed5aa976` 要求的 torch 版本，A4000 用的是 2.16.0.dev20261023，本机的 venv_1006 是 10-05 / 10-06 的 nightly；
  - memory 的规则是"上游 PR 的数值要 H100 测，5060 只做 smoke"。M3 已有的数都是在 A4000 上测的，HSDP 那张表用哪台机器，由用户定。
- **推送：** 10-25 这台机器的 SSH key 被 GitHub 拒绝，用户在重新加；在这之前 logbook 只在本地提交，fork 只能读。

## 每一条怎样算完成

1. **HSDP：**
   - 同一条数据流上配对 fsdp4 和 HSDP 2 x 2（都是 dp4，同一个 seed checkpoint，同一份预热过的 cache），再加一行 noise floor（同一格跑两次），报第 1 / 3 / 10 步的 loss 和 grad norm。
   - 第 1 步 loss 逐位相同，后面的差要么在 noise floor 以内，要么定位到具体原因。
   - 结果写进基线正文，README 和 5D 正文的说法保持一致。
   - 跑不了就把三处 HSDP 都删掉。
2. **CPU 全量：**
   - 两个分支各跑一遍 `tests/unit_tests/cpu` 全量，再在它们的 base 上跑一遍 main；
   - 只在分支上失败的是 0；
   - 通过数写进两份正文的 Test plan。
3. **两个 PR 的正文：**
   - 如果 prep 提交留在基线 PR 里：
     - 加 "Changes outside `models/minimax_m3/`"，每个 prep 提交一条；
     - P2 那条写上 Qwen3.5 前后每步 loss 相同的检查；
     - 5D 正文第一行和 Summary 里的 `#<FSDP2>` 和 prep 的说法跟着改；
     - P4、P5 要么单独开，要么在 5D 正文里说明。
   - 正文里不再有 `#<...>` 占位。
4. **flex bug：** 两条路，二选一，对外动作由用户定：
   - 用户同意后，用最小复现发 PyTorch issue，正文和代码注释链接过去；
   - 或者把正文和注释改成只讲约束（"the mask captures one tensor"）。
5. **"only the index projections"：**
   - README 和 docstring 都改成 KL 训 indexer 的投影和它们的两个 norm；
   - 做法是 amend `de7806dae`，再把 5D 的三个提交 rebase 上去；
   - force-push 之前，先把旧 head 推到 backup 分支，备份和 force-push 写在同一条 `&&` 链里。
6. **"Current scope"：**
   - 照 #4025 写：支持和不支持的并行、视频、compile；
   - 写明 MSA 的选块和 KL 目前是 PyTorch 按 128 个 query 一块循环算的，带 host 同步（每个 batch 建 metadata 时一次 `.tolist()`，KL 每层一次 `nonzero().tolist()`，每个 tile 一次 `torch.tensor(...)`），融合 kernel 是后续工作。

**约束：**
- titan 的提交和正文不带任何 Claude 署名；
- 新的修改单独成提交，第 5 条的 amend 除外；
- 发 issue、开 PR、force-push 已经公开的分支，都等用户点头；
- 正文里的数按 logbook 的数值表规则来（同一条数据流、noise floor、第 1 步逐位）。

## goal 文字

```
MiniMax M3 两个 PR（FSDP2 基线 minimax_m3 = de7806dae，5D 占位 minimax_m3_5d = 798f8817f）达到 #4025 的合入标准，可以开 PR。按 logbook Raising_PRs/RFC_MINIMAX_M3/GOAL_PR_READY_2026-10-25.md 完成六条：
(1) HSDP：在同一条数据流上配对 fsdp4 和 HSDP 2x2（都是 dp4），加 noise floor，报第 1/3/10 步的 loss 和 grad norm，第 1 步 loss 逐位相同，写进基线正文；跑不了就删掉 README、基线正文、5D 正文三处 HSDP。
(2) 两个分支各跑一遍 CPU 全量，和 main 比，只在分支上失败的是 0，通过数写进 Test plan。
(3) prep 提交留在基线时，正文加 "Changes outside models/minimax_m3/"（每个 prep 提交一条），5D 正文第一行跟着改，正文里不再有 #<...> 占位。
(4) flex backward autotune 的断言：用户同意后发 PyTorch issue 并链接过去，否则改成只讲约束。
(5) README 和 docstring 里的 "only the index projections" 改成包含 index norm；amend 基线提交，5D 的三个提交 rebase 上去，force-push 前先推备份。
(6) 正文加 "Current scope"：支持和不支持的并行、视频、compile，以及 MSA 选块和 KL 用 PyTorch 分块循环、带 host 同步，融合 kernel 是后续工作。
titan 的提交和正文不带 Claude 署名；发 issue、开 PR、force-push 已经公开的分支，都先问用户。
```

## 进度（10-25 5060 会话，细节见 `CHECKS_2026-10-25.md`）

1. **HSDP：数据和定位都完成，正文已写。**
   - 第 1、2 步 loss 和 grad norm 逐位；fsdp4 跑两次，10 步全同。
   - 第 3 步起的差定位到 FSDP 规约的求和顺序：两格的规约输入逐位相同，两边的第 1 步梯度都能按各自的求和树逐位重建。
   - README 和 5D 正文保留 HSDP；5D 的 Summary 改成不暗示 HSDP 和 TP / EP / PP 组合测过。
   - 表是 5060 测的，正文里写不写由用户定。
2. **CPU 全量：完成。**
   - 基线 1930 passed、5D 1932 passed，main 1906 passed；三棵树都是 18 skipped、3 xfailed，唯一的 1 failed 三棵树都有。
   - 只在分支上失败的是 0，计数已写进两份 Test plan。
3. **正文：完成。**
   - 基线加了 "Changes outside `models/minimax_m3/`"，四个 prep 提交各一条，外加测试登记一条。
   - 5D 第一行改写；两份粘贴段都没有 `#<...>`。
4. **flex：正文和代码注释都只讲约束。**
   - issue 草稿 `flex_bwd_capture_order/ISSUE_DRAFT_2026-10-25.md`，在本机 nightly 上复现、验证、查重过，等用户定。
5. **amend 和 rebase：本地完成。**
   - 基线 `9a5ab9ebe`、5D `89fc094d1`，本地备份分支各一个。
   - force-push 等用户点头。本机 SSH 还被拒，点头之后也要先有能写的远端。
6. **Current scope：完成。**
   - 包括并行、视频、compile 和 CUDA graph、MSA 的分块循环和三处 host 同步、只有 debug recipe。

**用户 10-25 的决定：**
- force-push：同意。等这台机器的 SSH key 在 GitHub 生效，按 `CHECKS_2026-10-25.md` 末尾那条 `&&` 链执行（先推两个 backup，再 force-with-lease）。
- PyTorch issue：不发；第 4 条走"只讲约束"那一支，已经完成。
- HSDP 表：正文改成 "Pending (H100)."；5060 的表和定位留在 `CHECKS_2026-10-25.md`。
- `lm_head` 的附带发现：只记在 logbook。

- 另一个会话（CPU）记的用户顺序（`GPU_PROMPT_M3_4D_MATRIX_2026-10-25.md`）：
  - 先开 5D 的 draft 占位，等完整并行矩阵跑完再开 FSDP2 基线；
  - 5D 正文已由 CPU 会话改成短的占位版，本会话只把 Test plan 第一条换成 CPU 全量。

**force-push 已完成（10-25）：** `minimax_m3` = `9a5ab9ebe`，`minimax_m3_5d` = `89fc094d1`，旧 head 在 fork 的两个 `backup/..._pre_20261025` 分支。
