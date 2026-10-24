---
name: upstream-model-pr-reuse-titan-modules
description: "New torchtitan model PRs are assembled from titan modules (HF only as numerics oracle); lessons come only from maintainers' reviews"
metadata:
  node_type: memory
  type: feedback
  originSessionId: e7080317-3a4f-4454-b734-9868986c7b7e
  modified: 2026-10-24T07:36:35.777Z
---

When writing a new model's eager implementation for torchtitan, assemble it from titan's modules (common attention, fused feed forward, `_grouped_mm` MoE, common router, common `VisionTransformerBlock` with configs, existing sparse-attention paths) and use the HF implementation only as the numerical-parity reference in a reduced-config test. When mining past PRs for lessons, use only the maintainers' review comments (tianyu-l, shuhuayu, drisspg); the user's own review comments on others' PRs carry no weight.

**Why:** pytorch/torchtitan#4025 (Kimi K3 eager + FSDP2, filed by someone else 3 days after the K3 release) got "seems an AI-slopped implementation" and many "reuse X" requests because it ported modeling_kimi_k3.py; tianyu-l: verifying numerics on a slow HF-style port means redoing it after swapping in titan modules. The user said (2026-10-24) only maintainer reviews matter.

**How to apply:** before writing model code, map every component to the titan class that owns it; the checklist distilled from #4025 lives in the logbook at `Raising_PRs/RFC_MINIMAX_M3/LESSONS_4025_MAINTAINER_REVIEW.md`. Related: [[upstream-rfc-short-local-maintainers-own-tracking]], [[minimax-m3-integration]].
