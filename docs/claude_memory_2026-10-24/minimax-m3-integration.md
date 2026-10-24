---
name: minimax-m3-integration
description: "MiniMax M3 into torchtitan, started 2026-10-24; FSDP2 baseline branch minimax_m3 and 5D branch minimax_m3_5d pushed to the fork 2026-10-24, no PR opened; both PR bodies written; CP deliberately not implemented"
metadata:
  node_type: memory
  type: project
  originSessionId: e7080317-3a4f-4454-b734-9868986c7b7e
  modified: 2026-10-24T22:15:10.661Z
---

Started 2026-10-24: bring MiniMax M3 (and M3.1-Flash, same backbone, released 2026-10-17) into torchtitan. The "technical report" the M3 README links is the MSA paper (arXiv 2606.13392). No torchtitan issue or PR existed for MiniMax/MSA on 2026-10-24; Megatron-Bridge supports M3 but runs full attention with the indexer frozen.

**Why:** the K3 lesson: the user kept 5D work local and someone else's FSDP2 baseline (#4025) landed first, 3 days after release; the first PR is what claims the model.

**How to apply:** prioritise the eager + FSDP2 PR, then parallelism. Status 2026-10-24 evening: both branches pushed to the fork (`origin/minimax_m3` = `de7806da`, `origin/minimax_m3_5d` = `798f8817`, tracking set), no PR opened; logbook committed and pushed (`36834c0e` K3 A4000 kit, `9a79058e` the M3 dir):
- Branch `minimax_m3` (worktree `/workspace/wt_m3`): prep P1 OffsetRMSNorm to common, P2 partial RoPE in `RoPE`, P3 DSV3 adapter seams, P1b norm types, then the M3 FSDP2 baseline `de7806da`. Bodies: logbook `Raising_PRs/RFC_MINIMAX_M3/PR_BODIES_M3_FSDP_2026-10-24.md` (M3 body ready except `#<P1..P3>` numbers; prep PRs still to be cut by cherry-pick, their pass counts unfilled).
- Branch `minimax_m3_5d` (worktree `/workspace/wt_m3_5d`): P4/P5 sharding configs to common, then TP/SP + EP + PP `798f8817`. CP not implemented on purpose (data-dependent per-layer mask; user rule: skip dimensions needing special handling like K3's VPP AttnRes cache). User said the whole 5D will likely not be filed as is, only as a draft placeholder: body `PR_BODY_M3_5D_2026-10-24.md`, title `[DO NOT MERGE]` (PR29 precedent).
- Matrix: 11 non-CP text cells + pp2 x vpp2 pass; every step-1 gap located (notes `M3_5D_2026-10-24.md`), see [[flex-automatic-dynamic-pp-numerics]].
- Open for the user: optional one-word docstring fix ("index projections" -> "indexer"), whether to drop the fp32 parity rows (4025 maintainer said bf16 only), P4/P5 as separate prep PRs, PP CI cell.
Related: [[upstream-model-pr-reuse-titan-modules]], [[upstream-rfc-short-local-maintainers-own-tracking]], [[flex-bwd-autotune-capture-order-bug]], [[commit-attribution-titan-vs-logbook]], [[sparse-attention-cp-gap]] (why CP was skipped, and what doing it would take).
