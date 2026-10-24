"""CPU-only checks of the HF MiniMax-M3 vision path with a tiny random config (fp32)."""
import torch

from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLConfig
from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLModel
from transformers.vision_utils import get_vision_position_ids

torch.manual_seed(0)
cfg = MiniMaxM3VLConfig(
    text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=16, dense_intermediate_size=16,
                     shared_intermediate_size=16, num_hidden_layers=1, num_attention_heads=4,
                     num_key_value_heads=4, head_dim=8, num_local_experts=2, num_experts_per_tok=1,
                     rope_theta=5e6, partial_rotary_factor=0.5, moe_layer_freq=[0]),
    vision_config=dict(hidden_size=80 * 2, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                       patch_size=14, temporal_patch_size=2, spatial_merge_size=2, rope_theta=10000.0),
    projector_hidden_size=48, image_token_index=60, video_token_index=61, attn_implementation="sdpa",
)
m = MiniMaxM3VLModel(cfg).float().eval()
vt = m.vision_tower
print("vision head_dim:", 160 // 2, "inv_freq len:", vt.rotary_emb.inv_freq.numel(), vt.rotary_emb.inv_freq.tolist()[:3])
grid = torch.tensor([[1, 4, 6], [2, 2, 2]])
pos = get_vision_position_ids(grid, 2, include_temporal=True)
print("position_ids shape:", tuple(pos.shape), "first 8 rows (t,h,w):", pos[:8].tolist())
print("video part (t,h,w):", pos[24:].tolist())
cos, sin = vt.rotary_emb(torch.zeros(1), pos)
print("cos shape:", tuple(cos.shape))
# check layout: cos[:, i] for i in [0,13) depends on t, [13,26) on h, [26,39) on w, then repeated
inv = vt.rotary_emb.inv_freq
exp = torch.cat([pos[:, 0:1] * inv, pos[:, 1:2] * inv, pos[:, 2:3] * inv], -1).cos()
print("cos == cat([t*inv, h*inv, w*inv]) x2:", torch.allclose(cos, torch.cat([exp, exp], -1), atol=1e-6))

pix_a = torch.randn(1 * 4 * 6, 3 * 2 * 14 * 14)
pix_b = torch.randn(2 * 2 * 2, 3 * 2 * 14 * 14)
with torch.no_grad():
    fa = m.get_image_features(pix_a, grid[:1]).pooler_output[0]
    fab = m.get_image_features(torch.cat([pix_a, pix_b]), grid).pooler_output
    print("image A features independent of co-batched image B:", torch.allclose(fa, fab[0], atol=1e-5),
          "max diff", (fa - fab[0]).abs().max().item(), "| split sizes:", [t.shape[0] for t in fab])
