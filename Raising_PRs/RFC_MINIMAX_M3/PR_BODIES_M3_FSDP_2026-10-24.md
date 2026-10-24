# MiniMax M3 FSDP2 基线 PR 及三个 prep PR 的正文（2026-10-24）

## 状态（不粘贴）

- **10-24 晚 goal（用户："把两个PR的body写出来 FSDP2和整个5D的 ... 按照logbook的模版"）：** M3 正文按 #4577 格式重写（`PR_WRITING_RULES.md`、logbook CLAUDE.md 的 "#4577 is the reference for code and body"），排版照最近的 `PR_K3_VIDEO/PR_BODY_K3_VIDEO_2026-10-15.md`：Summary 一句话加每个组件一条、Design 嵌套 bullet、Test plan 列命令和通过数、数值放 Results。三个 prep PR 只改了两处：P1 里一条写了两句的 bullet 拆成两条；P3 的 "Kimi K2.5" 改成 "Kimi K2.7"，和树里 `kimi_k2_7` 文件夹、README 标题一致（5D 正文同样改了）。
  - 和上一版相比改了什么（都是对着代码和测试核出来的错）：
    - KL loss 训练的是整个 indexer（`wq`、`wk` 和它们的两个 norm），上一版写成只训 `wq`、`wk`。核对：单卡第 1 步梯度里第 1 层 indexer `q_norm` / `k_norm` 的梯度范数都是 1.8e-3，不是 0。
    - PP 在基线里不是 `NotImplementedError`：`supports_pipeline_parallel = False` 让 `BaseModel.pipeline` 抛 `RuntimeError`；只有 TP、CP、EP 在 `MiniMaxM3Model.__init__` 里抛 `NotImplementedError`。
    - GPU 第二个测试是 `test_training_injects_the_kl_gradient_into_the_indexer_only`：训练和 eval 的输出、q/k/v 梯度逐位相同，只有 index q/k 多出 KL 梯度。上一版写成 "the training forward against the eval forward"。CPU 测试的描述也按 10 个测试名改了（上一版把 GPU 测的 "KL 梯度只进 indexer" 算到了 CPU 上）。
    - 对拍表移到 `## Results`，表下的说明拆成三条单行 bullet；Design 里用分号连着两件事的条目都拆开了。
  - 复核（10-24 晚，`de7806da`）：
    - 测试重跑：M3 CPU 16 passed、adapter `-k MiniMaxM3` 2 passed、通用 4 个文件 91 passed、M3 GPU 2 passed；集成格子 `minimax_m3_mm_fsdp` 之前在同一提交上跑过（passed，10 步）。
    - 对拍重跑（bf16 默认）：333 个 token，像素 4.768e-7，选块 94.4%，路由 98.3%，末 token KL 8.4318e-6，全位置 KL 2.9207e-3，和表一致；其余三行来自 `PARITY_HF_2026-10-24.md`。
    - 正文里点名的类和参数都在树里：`SparseIndexerLoss(AuxLoss)`（`deepseek_v4/compressor.py`）、`CosSinRoPE`、`index_block_size`、`VisionTransformerBlock`、`sparse_attention=False`、脚本的 `--dtype {bfloat16,float32}`、`--attn_backend {varlen,flex}`、`--force-hf-blocks`、`--force-hf-routing`。
    - diff 审读（`git diff a48507a8 de7806da`，逐行）：新增注释都是一行的约束说明；多行 docstring 都是一句摘要加一段代码看不出的事实（-1 表示空槽、返回值、dense 模式、视频分段）；没有 logbook 路径，没有实验数字。
  - 粘贴前要填：Summary 最后一条的 `#<P1>`、`#<P2>`、`#<P3>`，换成三个 prep PR 的编号（先开 prep PR）。
  - 两点留给用户定（都不影响正文）：
    - `MiniMaxSparseAttention` 的 docstring 写 "its KL loss trains only the index projections"，严格说两个 index norm 也在训，改成 "trains only the indexer" 更准。要改就得 amend `de7806da`，再把 5D 分支的三个提交 rebase 上去。
    - 4025 上 shuhuayu 说过 "titan's gemm only supports bf16, so we can simply use bf16, no fp32"。表里保留了 fp32 两行，用来说明 bf16 的残差来自精度、不是语义；想贴合 4025 的口径，可以只留 bf16 两行和 Results 的前两条 bullet。
  - 自查：段落一行一段、无 we/our/us、无破折号、纯 ASCII、Design 嵌套 bullet 且每条一行、没有 changed-files 块、没有 Relation 段。
- 分支：`minimax_m3`（worktree `/workspace/wt_m3`，基于 upstream main `3ed5aa97`），10-24 已推到 fork（`origin/minimax_m3`），没有开 PR。
- 提交顺序：`9172a00f` P1（OffsetRMSNorm 移到 common）、`64be0610` P2（partial RoPE 进 RoPE）、`69dda371` P3（DSV3 adapter seam）、`a48507a8` P1b（norm 类型放宽）、`de7806da` M3。
- 发 PR 时按用户之前的决定拆开：P1 + P1b 一个 PR，P2 一个，P3 一个，M3 叠在三者之上。到时再从这几个 commit cherry-pick 出各自的分支（推送前给用户看 diff）。
- 待办：三个 prep PR 的 Test plan 现在只写了 `passed`，通过数要在各自 cherry-pick 出来的分支上单独跑出来再填（在叠了 M3 的分支上跑，计数会混进 M3 的条目）。

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
- Stacked on #<P1> (OffsetRMSNorm in `models/common`), #<P2> (partial RoPE in `RoPE`) and #<P3> (DeepSeek V3 adapter seams).

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
  - One capture is deliberate: flex backward autotuning binds two or more captured tensors in a different order than its kernel reads them.
- Indexer loss:
  - The teacher is the main attention averaged over the query heads of a KV group, rebuilt from the flex LSE under `no_grad`.
  - The student is the softmax of the index scores over the same keys.
  - `MSAIndexerLoss` injects the KL gradient on the attention output.
  - The indexer input is detached, so the loss trains only the indexer (`wq`, `wk` and their norms).
  - DeepSeek V4's `SparseIndexerLoss` distills a head-summed target over compressed slots that all heads share.
  - MSA's target is per KV group over the tokens of the selected blocks, so the loss is a separate class.
- Indexer warmup: `build_model_config(..., sparse_attention=False)` attends every causal key while the indexer keeps training.
- Checkpoints:
  - The adapter supplies M3's key map and `CosSinRoPE` through the DeepSeek V3 seams.
  - It reshapes the `Conv3d` patch embedding to the encoder's `Linear`.
- Parallelism:
  - FSDP2 and HSDP.
  - TP, CP and EP raise `NotImplementedError`, and `supports_pipeline_parallel = False` refuses PP.

## Test plan
- `pytest tests/unit_tests/cpu/models/minimax_m3 -q` (`16 passed`): block selection, the flex mask and the KL against dense references, dense mode, the flavors and their layer layout, and the M3 flavor's 427,040,140,160 checkpoint elements (the released count).
- `pytest tests/unit_tests/cpu/models/test_state_dict_adapter.py -k MiniMaxM3 -q` (`2 passed`)
- `pytest tests/unit_tests/cpu/models/test_build_model_config.py tests/unit_tests/cpu/models/common/test_transformer_block.py tests/unit_tests/cpu/config/test_freeze_moe_routing.py tests/unit_tests/cpu/models/common/test_optional_imports.py -q` (`91 passed`)
- `pytest tests/unit_tests/gpu/models/minimax_m3 -q` (`2 passed`): compiled sparse flex attention against masked SDPA, and a training step whose output and q/k/v gradients are bitwise equal to an eval step's while the index queries and keys get the KL gradient.
- `python -m tests.integration_tests.run_tests <out> --test_suite models --test_name minimax_m3_mm_fsdp --ngpu 4` (`passed`, 10 steps)
- `python -m scripts.checkpoint_conversion.numerical_tests_minimax_m3`, with `--force-hf-blocks --force-hf-routing` and with `--dtype float32 --attn_backend flex` (results below)
- Scoped pre-commit checks, including formatting, lint, and Pyrefly on the changed files (`passed`; Pyrefly reports only errors that main already has in the touched test files)

## Results
The parity test reduces the released HF config to `debugmodel`, loads the TorchTitan weights into HF through HF's checkpoint conversion, and runs one 333-token image and text prompt; "HF imposed" runs TorchTitan on HF's block and expert selections.

| dtype | selections | block selections match | expert choices match | last-token KL | all-position KL |
|---|---|---|---|---|---|
| bf16 | free | 94.4% | 98.3% | 8.43e-6 | 2.92e-3 |
| bf16 | HF imposed | 100% | 100% | 8.50e-6 | 4.10e-5 |
| fp32 | free | 98.8% | 99.6% | -7.2e-7 | 4.69e-4 |
| fp32 | HF imposed | 100% | 100% | -2.0e-7 | 2.12e-6 |

- Pixels match to 4.8e-7.
- In fp32 every module matches HF to rounding except the routed experts, whose grouped GEMMs run in bf16 by design.
- The fp32 rows use `--attn_backend flex`, since varlen computes in bf16.
--- PASTE END ---

---

## P1 + P1b PR

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

---

## P2 PR

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

---

## P3 PR

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
