# MiniMax M3 FSDP2 基线 PR 的正文（2026-10-24 写，10-25 改为只开两个 PR）

## 状态（不粘贴）

- **10-26 H100（`root@87.116.91.146 -p 11468`，4 x H100 NVL 95 GB，driver 595.91 / CUDA 13.2，torch 2.16.0.dev20261024+cu130）：正文里每个数字现在都来自这台机器。**
  - 用户："本机重新改写2个body，FSDP2基线和4D body，不需要出现H200之外的数据"。H200 释放后改成："FSDP2里面的表重新来一遍"，以及"4D全部数值只用5060，FSDP2基线现在去跑"。
  - kit 在 `kit_m3_5060_2026-10-25/h100/`（README 里写了运行顺序），结果在 `results_h100/runs/`；正文链接钉在 logbook 提交 `b8492e78`。
  - 每一项的来源（全部是 H100）：
    - dp4 表：`hsdp_s100/table_4312.md`，和 H200 那份逐字节相同。三个格子的 loss 文件各 100 行全同，第 1 步梯度的两份对比也逐字节相同。所以表里的数一个没变，只改了硬件和 kit 链接。
    - 规约重建：`hsdp_s100/rebuild/reduce_emulate.txt`。规约输入 1272/1272 逐位相同；按 HSDP 的树和按 ring (0,1,2,3) 重算，各自 159/159 逐位复现第 1 步梯度；其余 5 种 ring 顺序都对不上。正文去掉了 "measured on 8 x RTX 5060 Ti"。
    - 对拍表：`parity/*.txt`，四行全部换成 H100 的数。bf16 不强制选择那行的选块一致率是 93.8%（A4000 是 94.4%），差别来自近似平局在不同硬件上翻向不同的一边。
    - "fp32 下每个模块都和 HF 一致到舍入误差"：`parity_diag/` 的 7 个探针在 H100 上重跑，结论和 10-24 在 A4000 上的一致。强制选择后，attn0 7.5e-7、ffn0 1.8e-6、attn1 6.4e-7、ffn1 4.9e-3；用 titan 自己的参数按 fp32 算 routed experts，对 HF 是 1.7e-7。
    - CPU 全量：基线单独跑 1931 passed / 18 skipped / 3 xfailed，main 是 1907 passed，只在分支上失败的是 0。第一遍和 main 同时跑，8 个 `test_parallelism_context` 测试抢写死的 12356 端口失败了，已单独重跑。5060 上两边都失败的那个 NaN 测试，在这台机器上两边都通过，所以正文删掉了那条说明。
    - GPU 单测 2 passed；集成格子 `minimax_m3_mm_fsdp` 跑满 10 步，rc=0，loss 从 8.106 降到 4.021。
    - pre-commit：Pyrefly 在基线和 main 上都是 0 errors。hook 失败只是因为 `--remove-unused-ignores` 删掉了 `video.py` 里 `import av` 的 ignore（这台的 venv 装了 PyAV），main 上一模一样（`precommit/pyrefly_probe_*.diff`）。5060 上的 67 条是那台 venv 的环境造成的，两台的 pyrefly 都是 0.45.1。
    - P2：dense 用 1 卡，MoE 用 4 卡（tp 2 x pp 2 x ep 2，`h100/p2cfg/`）；main、基线、main 再跑一次，`--assert-equal` 全部通过。正文 Changes outside 里 MoE 的配置改成了这个。
- **10-26 H200（已被上面的 H100 取代，正文不再引用；`root@217.18.55.94`，4 x H200，driver 595 / CUDA 13.2，torch 2.16.0.dev20261024+cu130，和 5060 同一个 build）：**
  - 用户："如果work的话最后body写h200"；"我们就100步，c4，按照我的4312来"。
  - 协议：
    - `c4_test`（2000 篇、369 万 token，100 步只读 11%，不会重复读）；
    - 100 步，表照 #4312 的格式：第 1 / 10 / 50 / 100 步，每格原始值，下面写 identical 或百分比；
    - 一个单卡 seed，一份 10 步预热的 cache，正式运行用 cache 副本，参照跑两次。
  - kit 在 `kit_m3_5060_2026-10-25/h200/`，结果在 `results_h200/runs/`。正文里的链接钉在 logbook 提交 `b8dea79`（仓库已改名为 `torchtitan_model_support`）。
  - 基线正文 Results 第二部分从 "Pending (H100)." 换成 H200 的 dp4 表（`results_h200/runs/hsdp_s100/table_4312.md`）：
    - fsdp4 两次 100 步全同；
    - HSDP 2 x 2 前 3 步逐位，从第 4 步的 grad norm、第 5 步的 loss 开始分开；
    - 第 1 步梯度 126/159 逐位，最大 5.22e-8，没有符号翻转。
  - 规约顺序的逐位重建是 5060 上做的，正文那句注明了 "measured on 8 x RTX 5060 Ti"。
- **10-25 晚 4D 矩阵（`GOAL_M3_4D_MATRIX_2026-10-25.md`）：** 正文没动。
  - dp4 这条数据流在 5D 树上用 20 步重跑了 fsdp4（两次）和 HSDP 2 x 2：第 1 步 loss 和 grad norm 都逐位，第 1 步梯度的特征和白天定位的一样（117/159 逐位，最大 5.27e-8）。
  - 其余 HSDP 组合也全部跑通：ep2 x HSDP 2 x 2、HSDP 2 x 2 x pp2、HSDP 2 x 2 x tp2 x ep2、HSDP 2 x 4、HSDP 4 x 2、ep4 x HSDP 2 x 4。所以 README 和正文里的 HSDP 保留。
  - goal 原文要给这份正文加一张 dp4 的表，但用户 10-25 定了正文里写 "Pending (H100)."，所以没加。可以直接贴的 dp4 表（fsdp4、fsdp4 again、HSDP 2 x 2 等）在 `M3_4D_TABLES_EN_2026-10-25.md`。
- **10-25（goal：两个 PR 达到 #4025 的合入标准，见 `GOAL_PR_READY_2026-10-25.md`；5060 会话）：**
  - 只开两个 PR：四个 prep 提交留在基线里。
    - 正文照 #4025 加了 "Changes outside `models/minimax_m3/`"，每个 prep 提交一条，内容取自下面三份旧 prep 正文；
    - P2 那条带上 Qwen3.5 前后每步 loss 相同的检查；
    - 最后一条是 M3 提交本身在共享测试和 README 里的登记。
    - Summary 里的 `#<P1>` / `#<P2>` / `#<P3>` 换成一句"前四个提交是共享代码的准备"。
    - 下面三份 prep 正文不再使用，留作出处。
  - 第 1 条 HSDP：本机 5060 上测完并定位，细节见 `CHECKS_2026-10-25.md`。
    - 第 1、2 步 loss 和 grad norm 都逐位；
    - fsdp4 跑两次，10 步全同；
    - 差异定位到 FSDP 规约的求和顺序，两边的第 1 步梯度都能从截获的规约输入逐位重建。
    - 正文 Results 里只写一行 "Pending (H100)."（用户 10-25 选的）：照 memory 规则，正文的 loss / grad 表要 H100 的数；5060 的表和定位留在 logbook。H100 上要跑的就是 `CHECKS_2026-10-25.md` 里这组（`kit_m3_5060_2026-10-25/run_hsdp.sh`，FA shim 在 H100 上不需要）。
  - 第 4 条：Design 里 "One capture is deliberate: flex backward autotuning binds ..." 那条删了，只留 "Its `mask_mod` captures one `[T, G, S, 2]` tensor of key ranges."。
    - 代码注释已经在 amend 里改成只讲约束；
    - issue 草稿在 `flex_bwd_capture_order/ISSUE_DRAFT_2026-10-25.md`；用户 10-25 定为不发，正文和注释保持只讲约束。
  - 第 5 条：README 和 `MiniMaxSparseAttention` 的 docstring 改成 "the index query and key projections and their norms"。
    - amend 成 `9a5ab9ebe`，5D 三个提交 rebase 上去；
    - 用户 10-25 同意后已 force-push，旧 head 先推成了 fork 上的 backup 分支（见 `CHECKS_2026-10-25.md` 第 5 条）。
  - 第 6 条：新加 "Current scope"。
    - Design 原来的 Parallelism 一条挪进 Current scope。
    - 视频、compile、CUDA graph、MSA 的分块循环和 host 同步，都是对着代码核过的：
      - `build_msa_metadata` 一次 `.tolist()`；
      - `MSAIndexerLoss.forward` 每层一次 `nonzero().tolist()`，每个 tile 一次 `torch.tensor(..., device=)`；
      - recipe 里 `disable_cuda_graphs=True`。
  - 第 2 条：Test plan 的 CPU 一行换成全量 `pytest tests/unit_tests/cpu`，和 main 逐个测试比过，只在分支上失败的是 0。
    - 原来分开列的 M3 CPU、adapter、通用四个文件三行并进了这一行。
  - Test plan 里每条的来源：
    - GPU 单测（2 passed）：10-25 在 5060 上对 `9a5ab9ebe` 重跑；
    - 集成格子 `minimax_m3_mm_fsdp`：10-24 在 A4000 上对 `de7806dae` 跑过，amend 只改了 README、docstring 和一行注释；
    - 对拍：10-24 A4000；
    - Pyrefly：10-25 和 main 逐条比过，67 条完全相同。
  - 硬件：用户 10-25 定为 Pending (H100)（见上）。M3 已有的对拍表（A4000，对的是 HF，不是 titan main）和 5D 的表（A4000）这次没有动。
  - 自查：
    - 段落一行一段；
    - 无 we/our/us、无破折号、纯 ASCII；
    - Design 嵌套 bullet 且每条一行；
    - 没有 changed-files 块、没有 Relation 段、没有 `#<...>`。
- **10-24（A4000 会话）的记录：**
  - 正文按 #4577 格式重写，排版照 `PR_K3_VIDEO/PR_BODY_K3_VIDEO_2026-10-15.md`。
  - 对着代码核出来、改掉的错：
    - KL loss 训练的是整个 indexer（`wq`、`wk` 和它们的两个 norm）。核对：单卡第 1 步第 1 层 indexer `q_norm` / `k_norm` 的梯度范数都是 1.8e-3。
    - PP 在基线里不是 `NotImplementedError`：`supports_pipeline_parallel = False` 让 `BaseModel.pipeline` 抛 `RuntimeError`；只有 TP、CP、EP 在 `MiniMaxM3Model.__init__` 里抛 `NotImplementedError`。
    - GPU 第二个测试是 `test_training_injects_the_kl_gradient_into_the_indexer_only`。
  - 复核：
    - 对拍重跑（bf16 默认）：333 个 token，像素 4.768e-7，选块 94.4%，路由 98.3%，末 token KL 8.4318e-6，全位置 KL 2.9207e-3，和表一致；其余三行来自 `PARITY_HF_2026-10-24.md`。
    - diff 审读（`git diff a48507a8 de7806da`，逐行）：新增注释都是一行的约束说明，没有 logbook 路径，没有实验数字。
  - 4025 上 shuhuayu 说过 "titan's gemm only supports bf16, so we can simply use bf16, no fp32"。表里保留了 fp32 两行，用来说明 bf16 的残差来自精度、不是语义；想贴合 4025 的口径，可以只留 bf16 两行。
- 分支：`minimax_m3`（worktree `/workspace/wt_m3`，基于 upstream main `3ed5aa976`）。
  - fork 上现在是 `9a5ab9ebe`（10-25 force-push）；10-24 的 `de7806dae` 在 fork 的 `backup/minimax_m3_pre_20261025`。
  - 提交顺序：
    - `9172a00f1` P1（OffsetRMSNorm 移到 common）；
    - `64be06102` P2（partial RoPE 进 RoPE）；
    - `69dda3713` P3（DSV3 adapter seam）；
    - `a48507a84` P1b（norm 类型放宽）；
    - `9a5ab9ebe` M3。

---

## M3 PR

标题：`[minimax_m3] add MiniMax M3 with MiniMax Sparse Attention on FSDP2`

--- PASTE BEGIN ---
## Summary
Add MiniMax M3 (428B total, 23B activated), a natively multimodal MoE whose decoder runs MiniMax Sparse Attention (MSA, https://arxiv.org/abs/2606.13392) in 57 of its 60 layers, as an eager model on FSDP2 and HSDP.
- Add MSA to `torchtitan/models/minimax_m3/attention.py`: `MSAIndexer`, block selection, a flex `BlockMask` built from the selection, `MSAInnerAttention(FlexInnerAttention)` and `MiniMaxSparseAttention(GQAttention)`.
- Add `MSAIndexerLoss(AuxLoss)` next to it, the KL loss that trains the indexer, injected on the attention output like the other auxiliary losses.
- Add the CLIP-style vision tower with 3D RoPE in `torchtitan/models/minimax_m3/vision_encoder.py`, built on the common `VisionTransformerBlock`.
- Add `MiniMaxM3StateDictAdapter` on `DeepSeekV3StateDictAdapter`'s per-expert conversion, and `scripts/checkpoint_conversion/numerical_tests_minimax_m3.py`, a parity test against HF `minimax_m3_vl`.
- Add CPU and GPU tests, and a 4-GPU multimodal FSDP cell to the models suite.
- The first four commits prepare shared code; they are listed under "Changes outside `models/minimax_m3/`".

## Design
- Block selection:
  - `MSAIndexer` has one index query head per KV group and one shared index key head, each with offset RMSNorm and the decoder's partial RoPE.
  - The indexer reads the detached layer input.
  - Key blocks are document-local: a block starts at each document start and every `index_block_size` tokens, so packed rows need no special case.
  - Each query keeps the top-k blocks per KV group by the max index score over the block's causal keys.
  - A query's own block is always kept.
- Sparse attention:
  - The selection becomes a flex `BlockMask` over 128-token tiles.
  - Its `mask_mod` captures one `[T, G, S, 2]` tensor of key ranges.
- Indexer loss:
  - The teacher is the main attention averaged over the query heads of a KV group, rebuilt from the flex LSE under `no_grad`.
  - The student is the softmax of the index scores over the same keys.
  - `MSAIndexerLoss` injects the KL gradient on the attention output.
  - The indexer input is detached, so the loss trains only the indexer (`wq`, `wk` and their norms).
  - DeepSeek V4's `SparseIndexerLoss` distills a head-summed target over shared compressed slots; MSA's target is per KV group over the selected blocks' tokens, so the loss is a separate class.
- Indexer warmup: `build_model_config(..., sparse_attention=False)` attends every causal key while the indexer keeps training.
- Checkpoints:
  - The adapter supplies M3's key map and `CosSinRoPE` through the DeepSeek V3 seams.
  - It reshapes the `Conv3d` patch embedding to the encoder's `Linear`.

## Changes outside `models/minimax_m3/`
- `models/common/norm.py`: `OffsetRMSNorm` moves here unchanged from `qwen3_5/model.py`, and its GPU test to `tests/unit_tests/gpu/models/common`.
- `models/common/rope.py`: `RoPE.forward` rotates the first `dim` channels of a wider query and key and passes the rest through, replacing Qwen3.5's `_partial_rope` and its `rotary_dim` field (a TODO there asked for this move).
  - Qwen3.5 debug recipes, deterministic, 10 steps on one warmed compile cache: identical loss on every step before and after, dense on 1 GPU and MoE at tp 2 x pp 2 x ep 2 on 4 GPUs.
- `models/deepseek_v3/state_dict_adapter.py`: the HF key map comes from an overridable `_build_from_hf_map(model_config)` and the RoPE check from `hf_rope_config_cls`; DeepSeek V3, V4 and Kimi K2.7 keep both.
- `models/common/decoder.py`, `attention/attention.py` and `config_utils.py`: `Decoder.Config.norm`, `GQAttention.Config.qk_norm` and `make_gqa_config(qk_norm=...)` accept `OffsetRMSNorm.Config`; only the field types change, and the eleven `bad-argument-type` ignores in Qwen3.5 and Qwen3.8 go away.
- Tests and README: MiniMax M3 joins the shared CPU tests' model lists (`test_model_defaults_to_varlen` skips it, since MSA runs on flex), the adapter tests and the README's model table.

## Test plan
- `pytest tests/unit_tests/cpu` (`1931 passed, 18 skipped, 3 xfailed`), including the 16 MiniMax M3 tests and the 2 MiniMax M3 adapter tests.
- `pytest tests/unit_tests/gpu/models/minimax_m3 -q` (`2 passed`)
- `python -m tests.integration_tests.run_tests <out> --test_suite models --test_name minimax_m3_mm_fsdp --ngpu 4` (`passed`, 10 steps)
- `python -m scripts.checkpoint_conversion.numerical_tests_minimax_m3`, with `--force-hf-blocks --force-hf-routing` and with `--dtype float32 --attn_backend flex` (first table below)
- Pre-commit on the changed files: every hook passes except Pyrefly, which reports 0 errors and, as on main, only removes the unused `missing-import` ignore on `import av` in `components/data/multimodal/utils/video.py`.

## Results
The parity test reduces the released HF config to `debugmodel`, loads the TorchTitan weights into HF, and runs one 333-token image and text prompt on one H100 NVL; "HF imposed" runs TorchTitan on HF's block and expert selections.

| dtype | selections | block selections match | expert choices match | last-token KL | all-position KL |
|---|---|---|---|---|---|
| bf16 | free | 93.8% | 97.9% | 1.50e-5 | 3.61e-3 |
| bf16 | HF imposed | 100% | 100% | 8.26e-6 | 4.07e-5 |
| fp32 | free | 98.9% | 99.6% | -9.8e-7 | 4.68e-4 |
| fp32 | HF imposed | 100% | 100% | -1.9e-7 | 2.13e-6 |

- Pixels match to 4.8e-7.
- In fp32 every module matches HF to rounding except the routed experts, whose grouped GEMMs run in bf16 by design.
- The fp32 rows use `--attn_backend flex`, since varlen computes in bf16.

4 x H100 NVL, debug model reading `c4_test` as text-only 512-token rows, 4096 tokens per step (two micro-batches per rank), AdamW, seed 42, deterministic, 100 steps; every cell loads one single-GPU seed checkpoint and runs on a copy of one compile cache that 10 steps of each layout warmed. fsdp4 and HSDP 2 x 2 (dp_replicate 2 x dp_shard 2) read the same samples, and fsdp4 runs twice as the noise floor. Kit, not part of this PR: [run_hsdp_h100.sh](https://github.com/QIU023/torchtitan_model_support/blob/b8492e785771/Raising_PRs/RFC_MINIMAX_M3/kit_m3_5060_2026-10-25/h100/run_hsdp_h100.sh).

HSDP matches FSDP bitwise through step 3 and parts from the grad norm at step 4, because its reduction adds the ranks' gradients in a different order.

| cell | loss, step 1 | step 10 | step 50 | step 100 | grad norm, step 1 | step 10 | step 50 | step 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fsdp4 | `8.032684` | `3.497031` | `3.007276` | `2.682869` | `1.7898` | `1.0201` | `0.6219` | `0.4218` |
| fsdp4, again | `8.032684`<br>identical | `3.497031`<br>identical | `3.007276`<br>identical | `2.682869`<br>identical | `1.7898`<br>identical | `1.0201`<br>identical | `0.6219`<br>identical | `0.4218`<br>identical |
| HSDP 2 x 2 | `8.032684`<br>identical | `3.494834`<br>-0.06% | `3.005924`<br>-0.04% | `2.682940`<br>+0.00% | `1.7898`<br>identical | `1.0071`<br>-1.27% | `0.6005`<br>-3.43% | `0.4254`<br>+0.86% |

- The step-1 gradients differ in 33 of 159 parameters, by at most 5.2e-8 in relative L2 norm, with no sign flips.
- fsdp4 sums the four ranks in one reduce-scatter, while HSDP sums pairs over dp_shard and then the two replicas; re-adding the gradients captured at FSDP's reduction in each order reproduces both layouts' step-1 gradients bitwise ([reduce_emulate.txt](https://github.com/QIU023/torchtitan_model_support/blob/b8492e785771/Raising_PRs/RFC_MINIMAX_M3/kit_m3_5060_2026-10-25/results_h100/runs/hsdp_s100/rebuild/reduce_emulate.txt)).

## Current scope
- Parallelism: FSDP2 and HSDP; TP, CP and EP raise `NotImplementedError`, and `supports_pipeline_parallel = False` refuses PP.
- Inputs: text and images, packed or not; the vision tower splits videos into segments, but no test covers video yet.
- Compile: the local compile regions (loss, fused activation, offset RMSNorm, partial RoPE, fp32 split) and flex attention; CUDA graphs stay off, since MSA syncs with the host inside the forward.
- MSA's block selection and KL loss are PyTorch loops over 128-query tiles:
  - one `.tolist()` per batch builds the tile metadata;
  - each sparse layer's KL takes one `nonzero().tolist()` and one host-to-device index copy per query tile;
  - fused selection and KL kernels are follow-up work.
- Only debug recipes are registered, and the adapter does not load the released NVFP4 and MXFP8 M3.1-Flash weights.
--- PASTE END ---

---

以下三份是 10-24 准备单独开 prep PR 时写的正文。10-25 决定只开两个 PR，prep 提交留在基线里，这三份不再粘贴，只作为上面 "Changes outside" 的出处。

## P1 + P1b PR（不再使用）

标题：`[models] move OffsetRMSNorm to models/common and accept it where RMSNorm configs are taken`

--- PASTE BEGIN ---
## Summary
Move `OffsetRMSNorm` (the float32 `(1 + weight) * norm(x)` RMSNorm) from `qwen3_5/model.py` to `models/common/norm.py`, and let `Decoder.Config.norm`, `GQAttention.Config.qk_norm` and `make_gqa_config(qk_norm=...)` take it.
- The class body is unchanged; Qwen3.5's model and flavors import it from `models/common`, and its GPU test moves to `tests/unit_tests/gpu/models/common`.
- The three config fields were typed `RMSNorm.Config`, so Qwen3.5 carried ten `bad-argument-type` ignores and Qwen3.8 one; they are removed.
- The field types are the only change; nothing changes at runtime.
- MiniMax M3 uses the same norm for its decoder, QK and indexer norms.

## Test plan
- `pytest tests/unit_tests/cpu/models/qwen3_5 tests/unit_tests/cpu/models/qwen3_8 tests/unit_tests/cpu/models/qwen3_6 tests/unit_tests/cpu/models/test_build_model_config.py -q` (`passed`)
- `pytest tests/unit_tests/gpu/models/common/test_offset_rmsnorm.py -q` (`passed`)
- Scoped pre-commit checks, including formatting, lint, and Pyrefly (`passed`)
--- PASTE END ---

## P2 PR（不再使用）

标题：`[rope] rotate the first dim channels of wider inputs`

--- PASTE BEGIN ---
## Summary
`RoPE.forward` applies partial RoPE itself: when the query and key are wider than the module's `dim`, their first `dim` channels rotate and the rest pass through, inside the existing `partial_rope` compile region.
- Qwen3.5 did this in `Qwen35Attention._partial_rope` with a `rotary_dim` field that always equalled its rope's `dim`, under a TODO to move it into the rope; the attention now calls its rope directly, and `rotary_dim` leaves `Qwen35Attention.Config` and the Qwen3.5 layer builders.
- Any `GQAttention` whose rope `dim` is smaller than its `head_dim` gets partial RoPE the same way (MiniMax M3 rotates 64 of 128 channels).

## Test plan
- `pytest tests/unit_tests/cpu/models/common/test_rope.py -q` (`passed`, adds `TestPartialRoPE`)
- Qwen3.5 debug recipes, CUDA graphs off, deterministic, 10 steps, one warmed inductor cache: the loss is identical on every step before and after, for the dense model on 1 GPU and the MoE model at dp_shard 2 x tp 2 x pp 2 on 8 GPUs; main run twice on the MoE model is identical too.
- Scoped pre-commit checks, including formatting, lint, and Pyrefly (`passed`)
--- PASTE END ---

## P3 PR（不再使用）

标题：`[deepseek_v3] let state dict adapter subclasses supply the key map and RoPE type`

--- PASTE BEGIN ---
## Summary
`DeepSeekV3StateDictAdapter` builds its HF key map in an overridable `_build_from_hf_map(model_config)` and validates the RoPE type named by `hf_rope_config_cls`, so another decoder can reuse its per-expert conversion.
- Its per-expert conversion (split grouped experts into HF expert weights, stack them back on load, online and offline) is the only one in the tree that takes the HF expert key from a class attribute, but `__init__` read the MLA key map from `model_config.layers[0].attention` and `from_hf` checked `ComplexRoPE`.
- DeepSeek V3, V4 and Kimi K2.7 keep the same map and the same check; MiniMax M3 overrides both.

## Test plan
- `pytest tests/unit_tests/cpu/models/test_state_dict_adapter.py tests/unit_tests/cpu/models/test_mla_state_dict_adapter.py -q` (`passed`)
- `pytest tests/unit_tests/cpu/models/deepseek_v3 tests/unit_tests/cpu/models/deepseek_v4 -q` (`passed`)
- Scoped pre-commit checks, including formatting, lint, and Pyrefly (`passed`)
--- PASTE END ---
