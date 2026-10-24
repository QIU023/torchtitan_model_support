---
name: sparse-attention-cp-gap
description: "titan has no CP for indexer-based sparse attention (DSV4, M3, GLM-5); first blocker is attn_gym lightning_indexer lacking query-chunk mode; DeepSeek/Megatron use halo + all-gather compressed KV"
metadata:
  node_type: memory
  type: project
  originSessionId: e7080317-3a4f-4454-b734-9868986c7b7e
  modified: 2026-10-24T18:11:10.897Z
---

Researched 2026-10-24 on titan main `3ed5aa97`, attn_gym 0.0.17. Notes: logbook `Raising_PRs/RFC_MINIMAX_M3/SPARSE_ATTN_CP_RESEARCH_2026-10-24.md`.

- DSV4 refuses CP (`deepseek_v4/model.py:258-261`, since #3634). Our M3 refuses it too. GLM-5 PR #5578 (open, external) "supports" CP with a dense `[1,T,T]` mask and a frozen indexer, so it does not scale.
- The gap is a whole family's CP backend, not titan's CP framework (`CPInnerAttention`/`prepare_cp_metadata`/`ContextParallelTransform` can host it). Linear-attention CP was filled the same way: attn_gym routing first, then titan `cp_kda`/`cp_gdn`.
- Hard blocker: attn_gym `lightning_indexer` "processes whole documents, not query chunks with cached history" and has no q causal offsets.
- Soft blocker: `gather_attn` requires the SWA `local_kv` to have the same length as the query, so a halo cannot be passed.
- Consensus design: DeepSeek V4 report §3.4.3 and Megatron #7469 (merged 2026-10-09) use a left-boundary halo, an all-gather of the compressed KV and indexer K with select-and-pad, and contiguous layout. Ulysses is a bad fit (q/o per token is about 400x the compressed-pool volume).
- M3 needs no kernel change (torch indexer + flex), only in-forward local-Q/global-KV masks.

**Why:** the user suspected CP for sparse attention is a core titan gap, not an M3 quirk; this records what the evidence supports.

**How to apply:** if M3 or DSV4 CP comes up, start from the notes file. Before writing code, DM the maintainers: drisspg (attn_gym, DSV4 gather_attn), acisseJZhong (titan CP), fegin (#5071). Re-search titan, attn_gym and Megatron first, since this area moves weekly. Related: [[minimax-m3-integration]], [[upstream-rfc-short-local-maintainers-own-tracking]].
