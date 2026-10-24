"""Probe: released config fields, reduced HF model keys/shapes vs titan to_hf."""
import json, sys
from dataclasses import replace
import torch
from transformers import AutoConfig
from transformers.models.minimax_m3_vl import MiniMaxM3SparseForConditionalGeneration

path = sys.argv[1]
cfg = AutoConfig.from_pretrained(path)
print(type(cfg).__name__, "model_type", cfg.model_type)
t, v = cfg.text_config, cfg.vision_config
for k in ("vocab_size", "hidden_size", "intermediate_size", "dense_intermediate_size", "shared_intermediate_size",
          "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "rope_parameters", "rotary_dim",
          "num_local_experts", "num_experts_per_tok", "routed_scaling_factor", "index_n_heads", "index_head_dim",
          "index_block_size", "index_topk_blocks", "index_local_blocks", "rms_norm_eps", "tie_word_embeddings"):
    print(" text", k, getattr(t, k, "<missing>"))
print(" text layer_types[:5]", t.layer_types[:5], "mlp_layer_types[:5]", t.mlp_layer_types[:5])
for k in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "patch_size",
          "temporal_patch_size", "spatial_merge_size", "layer_norm_eps", "rope_parameters", "hidden_act", "num_channels"):
    print(" vision", k, getattr(v, k, "<missing>"))
print(" top", "image_token_index", cfg.image_token_index, "video_token_index", cfg.video_token_index,
      "projector_hidden_size", cfg.projector_hidden_size, "merged_hidden_size", getattr(cfg, "merged_hidden_size", None))
