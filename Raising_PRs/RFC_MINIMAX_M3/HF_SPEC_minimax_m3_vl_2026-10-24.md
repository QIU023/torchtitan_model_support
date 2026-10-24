# HF transformers 5.20.2 `minimax_m3_vl` 实现规格（2026-10-24）

出处和用法：

- 由子 agent 只读整理（读 `transformers/models/minimax_m3_vl/` 全部文件，加上在 CPU 上用随机小模型做的实验，只下载了 checkpoint 的 index 等小 JSON）。正文保留英文原文，因为几乎全是代码、参数名和行号。
- 我亲自复核过的两点：indexer 打分没有 `1/sqrt(d)`、只旋转前 64 维（`modeling_minimax_m3_vl.py:548-575`）；视觉塔把一次调用里所有 patch 拼成一条序列、`attention_mask=None`（`:1185-1196`）。后一点和 SGLang 不同：SGLang 每个图片或视频一个 varlen 段，视频按 4 个时间组切段（`sglang/srt/models/minimax_vl_common.py` 的 `_compute_cu_seq_len`、`_apply_max_frames_limit`），titan 按 SGLang 做。
- 用途：写 titan 实现时的数值参考（**不是代码模板**，见 `LESSONS_4025_MAINTAINER_REVIEW.md` 第 0 节）。和论文不一致的地方以发布的权重和 HF 为准，KL loss 的写法以论文为准（HF 没有 KL loss）。
- 和论文的差异：
  - index 分支有 q/k 的 Gemma RMSNorm 和部分 RoPE，打分不除以 `sqrt(d_idx)`。选块不受缩放影响，但 KL 里 `P_idx` 的 softmax 受影响，训练时缩放要做成明确的配置。
  - 本地块占 16 个名额中的一个。

路径缩写：`MOD` = `.../transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py`，`CFG` = `configuration_minimax_m3_vl.py`，`PROC` / `IMG` / `VID` = processing / image_processing / video_processing，`TF` = `.../site-packages/transformers`。子 agent 的脚本和下载在 scratchpad `m3/agent_hf/`。参数总数按 HF 代码加 config 推出来是 427,040,140,160，与 Hub 的统计一致。

## 0. Config to effective values (`CFG`:120-152)

- hidden 6144, 60 layers, 64 q heads, 4 KV heads, head_dim 128, vocab 200064, `rms_norm_eps` 1e-6, 128 experts top-4, MoE intermediate 3072, shared 3072, dense 12288, `routed_scaling_factor` 2.0, `swiglu_alpha` 1.702, `swiglu_limit` 7.0.
- `rope_parameters = {"rope_theta": 5000000, "partial_rotary_factor": 0.5, "rope_type": "default"}` (Flash: theta 1e7), moved from top-level keys by `convert_rope_params_to_dict` (`TF/modeling_rope_utils.py`:752-781).
- Indexer: `index_n_heads` 4, `index_head_dim` 128, `index_block_size` 128, `index_topk_blocks` 16, `index_local_blocks` 1, mapped from the legacy `sparse_attention_config` keys (`CFG`:128-136).
- `layer_types[i]` = `"minimax_m3_sparse"` if `sparse_attention_freq[i]` else `"full_attention"` (`CFG`:141-146): M3 layers 0-2 full, 3-59 sparse; Flash all 60 sparse. `mlp_layer_types[i]` = `"sparse"` (MoE) if `moe_layer_freq[i]` else `"dense"` (`CFG`:149-152): both models 0-2 dense, 3-59 MoE.
- `hidden_act` `"swigluoai"` is overwritten to `"silu"` (`CFG`:126) and never used by the math.
- Keys never referenced by HF: text `use_gemma_norm`, `attention_output_gate`, `use_qk_norm`, `qk_norm_type`, `scoring_func`, `use_routing_bias`, `n_shared_experts`, `num_mtp_modules`, `num_nextn_predict_layers`, `router_jitter_noise`; `rotary_dim` (rotary width comes only from `partial_rotary_factor`); sparse `use_sparse_attention`, `sparse_disable_index_value`, `sparse_score_type`, `sparse_init_block`; top level `image_grid_pinpoints`, `image_seq_length`, `process_image_mode`, `vision_feature_layer`, `vision_feature_select_strategy`, `projector_hidden_act`, `multimodal_projector_bias`; vision `img_token_compression_config`, `vision_segment_max_frames`, `image_size`, `projection_dim`, `rope_mode`. HF hardcodes behaviour matching these values.

## 1. Decoder layer (`MOD`:627-666)

- Pre-norm, no sandwich norms, no residual scaling:
  ```python
  residual = h; h = self.input_layernorm(h); h, _ = self.self_attn(h, ...); h = residual + h
  residual = h; h = self.post_attention_layernorm(h); h = self.mlp(h); h = residual + h
  ```
- RMSNorm (`MOD`:141-157), Gemma style, zero-initialised weights (`MOD`:147, 700-701), computed in fp32:
  ```python
  def _norm(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
  def forward(self, x):
      output = self._norm(x.float())
      output = output * (1.0 + self.weight.float())
      return output.type_as(x)
  ```
- `self_attn` builds an indexer only when `layer_types[i] == "minimax_m3_sparse"` (`MOD`:417-419); `mlp` is the MoE block when `mlp_layer_types[i] == "sparse"`, else the dense MLP (`MOD`:634-638).
- Text model (`MOD`:704-781): embedding without scaling; RoPE cos/sin once per forward; `position_ids` default `arange(S) + past_seen` [1, S]; final norm (`MOD`:776).

## 2. Attention (`MOD`:397-478)

- `q_proj` [8192, 6144], `k_proj` [512, 6144], `v_proj` [512, 6144], `o_proj` [6144, 8192], no biases; heads contiguous (head h = rows [128h, 128h+128)).
- GQA: query head i uses KV head i // 16 (`repeat_kv`, `MOD`:317-326). Scale `head_dim**-0.5`, no soft-cap, dropout 0.
- QK norm: one Gemma RMSNorm weight [128] for q and one for k, shared across heads, over all 128 dims, **before** RoPE (`MOD`:432-437):
  ```python
  query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
  key_states   = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
  value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
  query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
  ```
- Partial RoPE (`MOD`:296-314, 372-394): first 64 dims of each head rotate, dims 64-127 pass through; half-split (pairs (d, d+32)), frequency `theta^(-2d/64)`, no scaling; cos/sin cast to the model dtype before use:
  ```python
  dim = int(head_dim * partial_rotary_factor)                       # 64
  inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
  freqs = position_ids[..., None].float() * inv_freq; emb = torch.cat((freqs, freqs), -1)
  q_rot, q_pass = q[..., :64], q[..., 64:]
  q_embed = cat([q_rot*cos + rotate_half(q_rot)*sin, q_pass]); rotate_half(x) = cat((-x[..., 32:], x[..., :32]))
  ```
- `attention_output_gate`: not implemented (no parameters, no checkpoint weights).
- Backends: eager (fp32 softmax), sdpa, and the `MiniMaxAI/msa` hub kernel; flash and flex unsupported (`MOD`:676-678). For sparse layers under eager/sdpa the mask is replaced by `build_block_mask` (`MOD`:456-464).

## 3. Indexer (`MiniMaxM3VLIndexer`, `MOD`:481-624)

- Parameters (`MOD`:536-539): `q_proj` [512, 6144] (4 index heads x 128), `k_proj` [128, 6144] (one shared key head), `q_norm` / `k_norm` [128] Gemma RMSNorm; no value path, no biases, no per-head weights.
- Forward (`MOD`:548-586):
  ```python
  idx_q = self.q_proj(hidden_states).view(batch, q_len, -1, self.head_dim)
  idx_q = self.q_norm(idx_q).transpose(1, 2)                      # [B, 4, Sq, 128]
  idx_k = self.k_proj(hidden_states).view(batch, q_len, 1, self.head_dim)
  idx_k = self.k_norm(idx_k).transpose(1, 2)                      # [B, 1, Sq, 128]
  cos, sin = position_embeddings                                  # main attention's 64-wide cos/sin
  idx_q, idx_k = apply_rotary_pos_emb(idx_q, idx_k, cos[..., :self.head_dim], sin[..., :self.head_dim])
  scores = torch.matmul(idx_q.float(), idx_k.float().transpose(-1, -2))     # fp32, [B,4,Sq,Sk], no 1/sqrt(d)
  token_future = k_positions[None, None, None, :] > position_ids[:, None, :, None]
  scores = scores.masked_fill(token_future, float("-inf"))
  if pad: scores = F.pad(scores, (0, pad), value=float("-inf"))
  block_scores = scores.view(batch, self.num_heads, q_len, num_key_blocks, self.block_size).amax(dim=-1)
  q_block = position_ids // self.block_size
  if self.local_blocks > 0:
      local_idx = (q_block[..., None] - torch.arange(self.local_blocks).view(1, 1, -1)).clamp(min=0)
      block_scores.scatter_(-1, local_idx.unsqueeze(1).expand(-1, self.num_heads, -1, -1), float("inf"))
  topk_scores, topk_indices = block_scores.topk(min(self.topk_blocks, num_key_blocks), dim=-1)
  return topk_indices.masked_fill(topk_scores == float("-inf"), -1)         # int64 [B,4,Sq,topk]
  ```
- Each index head selects independently for its KV group; no union across heads. The forced own block (`position_ids // 128`, +inf) is **inside** the 16-block budget (own block plus up to 15 past blocks); block-aligned, not a sliding window. Init/sink blocks are not implemented. Ties follow `torch.topk`.
- `build_block_mask` (`MOD`:600-624): (b, query head h, query i, key j) is attended iff `j // 128` is in `block_indices[b, h // 16, i]` (ignoring -1) and j is causal for i and not padded; the result is a dense additive mask [B, 64, Sq, Sk] with `finfo.min`. Verified by the agent against an independent implementation for eager and sdpa.
- Consequence: when `q_block + 1 <= 16` every past block is selected, so the layer is exactly dense causal; **sparsity starts at query position 2048**. Parity tests need S > 2048.
- `sparse_disable_index_value` is never read; HF implements only the variant without an index value path, and the checkpoint has no index value weights.
- Gradients: no `no_grad`/`detach`, but the only output is integer indices, so the indexer parameters get **no gradient** (verified `.grad is None`). No KL or any indexer loss anywhere in HF.

## 4. Caches, decode, packing

- Sparse layers use `MiniMaxM3VLSparseCacheLayer` (`MOD`:53-89) with cached post-norm post-RoPE index keys; incremental decode equals the full forward (verified). Only right padding is equivalent (left padding TODO at `MOD`:512-524).
- `MiniMaxAI/msa` kernel path (`TF/integrations/msa_attention.py`): SM100, head_dim 128, block 128, top-k in {4, 8, 16, 32}, `causal=True`, ignores `attention_mask`, a `custom_op` without autograd (forward only).
- Packed documents: dense layers work only with `use_cache=False` (same-document causal mask from `find_packed_sequence_indices`); **sparse layers do not support packing** (the indexer compares absolute key slots with reset positions and forces the wrong document's block; doc-2 logits mismatch by 0.28 max abs in the tiny model).

## 5. MoE (`MOD`:181-263)

- Router (`MOD`:231-240):
  ```python
  router_logits = F.linear(hidden_states.to(self.weight.dtype), self.weight)   # [N,128], no bias
  routing_weights = F.sigmoid(router_logits.float())
  scores_for_choice = routing_weights + self.e_score_correction_bias           # bias for selection only
  _, top_k_index = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)
  top_k_weights = routing_weights.gather(1, top_k_index)
  top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)                     # no eps
  ```
  No group-limited routing, capacity or jitter; `e_score_correction_bias` is a persistent buffer with no update rule in HF.
- Experts: `gate_up_proj` [128, 6144, 6144] (rows 0:3072 gate = ckpt w1, 3072:6144 up = ckpt w3), `down_proj` [128, 6144, 3072], no bias. swigluoai (`MOD`:213-219), same in the dense MLP and the shared expert:
  ```python
  gate, up = gate_up.chunk(2, dim=-1)
  gate = gate.clamp(max=7.0)                 # upper clamp only
  up = up.clamp(min=-7.0, max=7.0)
  glu = gate * torch.sigmoid(gate * 1.702)
  return (up + 1.0) * glu
  ```
- Shared expert: one dense MLP of 3072, no gate, added unscaled. Combine: `out = 2.0 * sum_k w_k * Expert_k(x) + Shared(x)`.
- Numerics: default experts implementation `"grouped_mm"` multiplies bf16 expert outputs by fp32 weights and sums top-k in fp32 (`TF/integrations/moe.py`:492-505); `"eager"` accumulates in bf16; `* 2.0` and `+ shared` in bf16.
- Aux loss (`MOD`:784-850): Mixtral style on the **softmax** of raw router logits, pooled over MoE layers, only with `output_router_logits=True`, weight 0.001.

## 6. Dense MLP (layers 0-2, `MOD`:163-178)

- `gate_up_proj` [24576, 6144] = cat(gate_proj, up_proj), `down_proj` [6144, 12288], no bias, the same clamped swigluoai.

## 7. Embeddings, head, MTP

- `embed_tokens` [200064, 6144], no scaling; final Gemma RMSNorm; `lm_head` [200064, 6144] untied, no bias, no soft-cap. Loss is `ForCausalLMLoss` (fp32 logits, shift by one, ignore -100).
- MTP not implemented; `mtp.*` ignored on load (`MOD`:687). The M3 index has no mtp keys, and `metadata.total_size` exceeds the listed tensors by 14,984,738,304 bytes, about one DeepSeek-style MTP block: the MTP weights look stripped from the release (inference).

## 8. Vision tower (`MOD`:953-1196)

- Patch embedding: `Conv3d(3, 1280, kernel=stride=(2, 14, 14), bias=False)` over `pixel_values` [N, 1176] viewed as (3, 2, 14, 14); images duplicate each frame along T=2; no class token, no absolute position embedding.
- Encoder: `pre_layrnorm` LayerNorm(1280, eps 1e-5); 32 CLIP pre-norm layers `x += attn(LN1(x)); x += fc2(GELU(fc1(LN2(x))))`, fc1 1280 to 5120 with bias, exact GELU; **no post-layernorm**; features are the last layer's output.
- Attention: q/k/v/out with bias, 16 heads x 80, scale 80^-0.5, bidirectional, `attention_mask=None` over all patches of all images and videos in the call (no `cu_seqlens`): images attend to each other (agent measured 27% of image A's query mass on image B at random init). Whether the original model isolates images is not determinable from HF.
- 3D RoPE (`MOD`:996-1046; positions `TF/vision_utils.py`:81-127): positions (t, h, w) restart per image, spatial order block-major over 2x2 merge windows:
  ```python
  spatial_dim = 2 * ((80 // 3) // 2)     # 26
  inv_freq = 1.0 / (10000 ** (torch.arange(0, 26, 2, dtype=torch.float) / 26))   # 13 freqs
  freq = (position_ids[..., None].float() * inv_freq).flatten(1); cos = torch.cat([freq, freq], -1)  # 78 dims
  q_rot = q[..., :78]; q = cat([q_rot*cos + rotate_half(q_rot)*sin, q[..., 78:]])   # pairs (i, i+39)
  ```
- Projector (`MOD`:1199-1218): per patch `linear_2(gelu(linear_1(x)))` 1280 to 6144 to 6144 with bias; then 4 consecutive patches (a 2x2 window) reshape to 24576 and `merge_linear_2(gelu(merge_linear_1(h)))` to 6144. Temporal merge is in the Conv3d, spatial 2x2 merge in the projector.

## 9. Multimodal merge and processor

- `masked_scatter` of image features at token 200025 and video features at 200026 (`MOD`:1307-1320, 1401-1417); plain 1D positions and a plain causal mask in the LLM.
- Tokens: image 200025, video 200026, start of image 200029, end of image 200030 (videos are also bracketed with the image start/end tokens; the video start/end tokens are unused by HF).
- Images: Qwen2-VL `smart_resize`, factor 28, `min_pixels` 3136, `max_pixels` 451584 = 672^2; bicubic, CLIP mean/std; text expansion start + image x (H/14 * W/14 / 4) + end; at most 576 tokens per image.
- Videos: frames must already be multiples of 28; odd frame counts pad the last frame; per temporal group an optional timestamp token, then start + video x (H/14 * W/14 / 4) + end.
- Unused by HF processing: `image_grid_pinpoints` (no anyres tiling), `process_image_mode`, `vision_segment_max_frames`. The repo also ships remote-code processors that the agent did not read.

## 10. Parameter inventory (HF name, shape, checkpoint key)

- fp32 in the checkpoint (inferred from the exact dtype counts): the 57 MoE router weights and `e_score_correction_bias`, and the patch embedding; everything else bf16.
- Embedding: `model.language_model.embed_tokens.weight` [200064, 6144] from `language_model.model.embed_tokens.weight`.
- Layer prefix: HF `model.language_model.layers.N.` from checkpoint `language_model.model.layers.N.`:
  - `input_layernorm.weight`, `post_attention_layernorm.weight` [6144]; `self_attn.{q,k,v,o}_proj.weight` as in section 2; `self_attn.{q,k}_norm.weight` [128].
  - Dense layers 0-2: `mlp.gate_up_proj.weight` [24576, 6144] = cat of `mlp.gate_proj.weight` and `mlp.up_proj.weight` [12288, 6144]; `mlp.down_proj.weight` [6144, 12288].
  - Sparse layers: `self_attn.indexer.{q_proj,k_proj,q_norm,k_norm}.weight` from `self_attn.index_{q_proj,k_proj,q_norm,k_norm}.weight`.
  - MoE layers 3-59: `mlp.gate.weight` [128, 6144] from `block_sparse_moe.gate.weight`; `mlp.gate.e_score_correction_bias` [128] from `block_sparse_moe.e_score_correction_bias`; `mlp.experts.gate_up_proj` [128, 6144, 6144] stacked from `block_sparse_moe.experts.{e}.w1.weight` and `.w3.weight` [3072, 6144] (w1 first); `mlp.experts.down_proj` [128, 6144, 3072] from `experts.{e}.w2.weight`; `mlp.shared_experts.gate_up_proj.weight` from `block_sparse_moe.shared_experts.{gate_proj,up_proj}.weight`; `mlp.shared_experts.down_proj.weight` from `block_sparse_moe.shared_experts.down_proj.weight`.
- Final norm `language_model.model.norm.weight`; head `language_model.lm_head.weight`.
- Vision: `vision_tower.vision_model.embeddings.patch_embedding.weight` [1280, 3, 2, 14, 14]; `vision_tower.vision_model.pre_layrnorm.*`; `vision_tower.vision_model.encoder.layers.N.{layer_norm1,layer_norm2,self_attn.{q,k,v,out}_proj,mlp.fc1,mlp.fc2}.*` (N = 0..31); no post-norm.
- Projector: `multi_modal_projector.linear_{1,2}.*` and `patch_merge_mlp.linear_{1,2}.*` (HF names `merge_linear_{1,2}`).
- Renames and merges: `TF/conversion_mapping.py`:656-721. Every checkpoint key is consumed; no `mtp.*` keys. About 25.96B active parameters per token for the text model including embedding and head.

## 11. MiniMax-M3.1-Flash

- All 60 layers sparse (layers 0-2 also get an indexer; the Flash index has `index_*` weights for all 60), theta 1e7, no MTP keys, vision `image_size` 672 (unused), the same 427.04B logical parameters.
- `quant_method "mxfp8"` routes to `FineGrainedFP8HfQuantizer`; experts become `FP8Experts` with the same swigluoai. `moe_quant_algo` NVFP4, `moe_quant_format`, `group_size` are referenced nowhere in HF.
- Checkpoint layout: routed experts per expert `w{1,2,3}.{weight_packed, weight_scale, weight_global_scale, input_global_scale}` (compressed-tensors NVFP4, packed U8); attention, indexer, dense MLP and shared experts `.weight` plus `.weight_scale_inv` (MXFP8, block [1, 32]); unquantized router, bias, norms, embedding, head, vision, projector. Loading Flash's NVFP4 experts through HF 5.20.2 appears unsupported (not tested).

## Parity-test notes

- S > 2048 to exercise the indexer; `use_cache=False`, one document per row, no padding (or right padding only).
- Set the experts implementation explicitly (`"eager"` or `"grouped_mm"`): they differ in bf16 vs fp32 accumulation.
- HF has no `_keep_in_fp32_modules`, so a bf16 load rounds the fp32-stored router weight, bias and patch embedding to bf16, which can change expert selection.
- cos/sin are cast to the model dtype before use; the sparse eager/sdpa path materialises [B, 64, S, S] masks; compare one image per vision call.
