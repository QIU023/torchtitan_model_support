"""Print HF state_dict names/shapes for a tiny MiniMax-M3 VL model (text layer0 dense/full, layer1 MoE/sparse)."""
import torch

from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLConfig
from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3SparseForConditionalGeneration

cfg = MiniMaxM3VLConfig(
    text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=16, dense_intermediate_size=24,
                     shared_intermediate_size=20, num_hidden_layers=2, num_attention_heads=8,
                     num_key_value_heads=4, head_dim=8, num_local_experts=3, num_experts_per_tok=1,
                     rope_theta=5e6, partial_rotary_factor=0.5, moe_layer_freq=[0, 1],
                     sparse_attention_config={"sparse_attention_freq": [0, 1], "sparse_num_index_heads": 4,
                                              "sparse_index_dim": 8}),
    vision_config=dict(hidden_size=16, intermediate_size=40, num_hidden_layers=1, num_attention_heads=2,
                       rope_theta=10000.0),
    projector_hidden_size=12, image_token_index=60, video_token_index=61,
)
m = MiniMaxM3SparseForConditionalGeneration(cfg)
for k, v in m.state_dict().items():
    print(f"{k:70s} {tuple(v.shape)} {v.dtype}")
print("non-persistent buffers:", [n for n, _ in m.named_buffers() if n not in m.state_dict()])
