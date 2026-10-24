---
name: flex-automatic-dynamic-pp-numerics
description: VLM PP-vs-single-GPU step-1 grads not bitwise because flex attention recompiles with dynamic heads where the vision tower compiled it first; set automatic_dynamic_shapes=False to compare
metadata:
  node_type: memory
  type: project
  originSessionId: e7080317-3a4f-4454-b734-9868986c7b7e
  modified: 2026-10-24T15:50:59.357Z
---

Found 2026-10-24 on MiniMax M3 (torch 2.16 nightly, sm86). On a rank that holds both the vision tower and the decoder, `FlexInnerAttention.compiled_flex_attn` is compiled first for the tower's head count; the decoder's call then recompiles and dynamo's automatic dynamic shapes mark the KV head dim dynamic. A pipeline stage without the tower compiles it once, static. Forward stays bitwise, but the backward dQ differs in a few rows per micro-batch, so PP step-1 gradients (norms, experts, embedding) drift from the single GPU while DeepSeek V3 (no tower, varlen) shows only dense-projection differences.

**Why:** a PP gap that looked like "titan PP accumulation" was half this compile artifact; the earlier write-up claimed the wrong split and had to be corrected.

**How to apply:** when a VLM's PP cell is not bitwise against the single GPU, first rerun both with `torch._dynamo.config.automatic_dynamic_shapes = False` (kit `step1_probe.py` env `M3_STATIC_SHAPES=1`) and check `TORCH_LOGS=recompiles`; what remains is titan's cross-micro-batch accumulation of the bf16 dense projection grads (also moved by `fsdp_defer_gradient_reduction`). For TP gaps, impose the single GPU's discrete decisions (MSA selections `M3_FORCE_SELECT`, MoE routing `M3_FORCE_ROUTE`) to separate flips from reduction order. Likely affects K3/Qwen3.5 too (untested). Related: [[minimax-m3-integration]].
