"""Parse the released configs with HF's config classes (no weights) and print derived fields."""
import json
import sys

from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLConfig

for path in sys.argv[1:]:
    with open(path) as f:
        d = json.load(f)
    d.pop("auto_map", None)
    cfg = MiniMaxM3VLConfig(**{k: v for k, v in d.items() if k != "model_type"})
    t = cfg.text_config
    v = cfg.vision_config
    print("=====", path.split("/")[-1])
    print("text rope_parameters:", t.rope_parameters)
    print("hidden_act:", t.hidden_act, "rms_norm_eps:", t.rms_norm_eps, "rotary_dim:", t.rotary_dim)
    print("index:", t.index_n_heads, t.index_head_dim, t.index_block_size, t.index_topk_blocks, t.index_local_blocks)
    lt = t.layer_types
    print("layer_types sparse idx:", [i for i, x in enumerate(lt) if x == "minimax_m3_sparse"][:5], "... count", lt.count("minimax_m3_sparse"), "full", lt.count("full_attention"))
    mt = t.mlp_layer_types
    print("mlp dense idx:", [i for i, x in enumerate(mt) if x == "dense"], "sparse count", mt.count("sparse"))
    print("experts:", t.num_local_experts, "topk:", t.num_experts_per_tok, "routed_scaling:", t.routed_scaling_factor,
          "aux_coef:", t.router_aux_loss_coef, "output_router_logits:", t.output_router_logits)
    for name in ["use_gemma_norm", "attention_output_gate", "use_qk_norm", "qk_norm_type", "scoring_func",
                 "use_routing_bias", "n_shared_experts", "num_mtp_modules", "num_nextn_predict_layers",
                 "sparse_attention_config", "moe_layer_freq", "max_position_embeddings", "use_cache",
                 "tie_word_embeddings", "bos_token_id", "eos_token_id", "pad_token_id", "attention_dropout"]:
        val = getattr(t, name, "<absent>")
        if isinstance(val, (dict, list)):
            val = f"<{type(val).__name__} present>"
        print(f"  text.{name} = {val}")
    print("vision rope_parameters:", v.rope_parameters, "spatial_merge:", v.spatial_merge_size,
          "temporal_patch:", v.temporal_patch_size, "hidden_act:", v.hidden_act, "image_size:", v.image_size)
    for name in ["img_token_compression_config", "vision_segment_max_frames", "projection_dim", "rope_mode"]:
        print(f"  vision.{name} = {getattr(v, name, '<absent>')}")
    for name in ["image_token_id", "video_token_id", "projector_hidden_size", "merged_hidden_size", "tie_word_embeddings",
                 "image_seq_length", "vision_feature_layer", "vision_feature_select_strategy", "projector_hidden_act",
                 "multimodal_projector_bias", "process_image_mode", "quantization_config"]:
        val = getattr(cfg, name, "<absent>")
        if isinstance(val, dict):
            val = {k: (vv if not isinstance(vv, list) else f"<list len {len(vv)}>") for k, vv in val.items()}
        print(f"  top.{name} = {val}")
