---
name: flex-bwd-autotune-capture-order-bug
description: torch nightly flex backward autotune binds 2+ captured mask_mod tensors out of order -> OOB reads / illegal memory access; keep mask_mods to one captured tensor
metadata:
  node_type: memory
  type: project
  originSessionId: e7080317-3a4f-4454-b734-9868986c7b7e
  modified: 2026-10-24T10:05:20.079Z
---

In torch 2.16.0.dev20261023 the flex_attention *backward* lowering autotunes on `bwd_input_nodes + score/mask_mod_other_buffers + mutated_grads` (HOP order) while the generated kernel binds captures in first-use order; the forward lowering already uses `list(choices[0].input_nodes)`. A mask_mod capturing two or more tensors whose orders differ makes the backward benchmark read the wrong buffer (out of bounds). It surfaced as `CUDA error: an illegal memory access` only on 2-GPU FSDP (cold cache), never under `CUDA_LAUNCH_BLOCKING=1`; real kernel calls bind correctly.

**Why:** found 2026-10-24 while bringing up MiniMax M3 MSA (see [[minimax-m3-integration]]); evidence, minimal repro, spy and the one-line fix are in `torchtitan_model_support/Raising_PRs/RFC_MINIMAX_M3/PARITY_HF_2026-10-24.md` and `flex_bwd_capture_order/`. Not yet reported upstream (user decides).

**How to apply:** any titan mask_mod (M3 or later models) should capture a single tensor; if an illegal memory access appears in flex backward autotuning, check capture order with the spy before suspecting the model. Tools that localized it: compute-sanitizer memcheck with `PYTORCH_NO_CUDA_MEMORY_CACHING=1`, then shape spy on `TritonTemplateCaller.benchmark`.
