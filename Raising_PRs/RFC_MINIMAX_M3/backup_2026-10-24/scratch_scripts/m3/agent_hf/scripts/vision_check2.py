"""Probe whether HF MiniMax-M3 vision attention mixes tokens across co-batched images (CPU, fp32)."""
import torch

from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLVisionConfig
from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLVisionModel

for impl in ("eager", "sdpa"):
    torch.manual_seed(0)
    vcfg = MiniMaxM3VLVisionConfig(hidden_size=160, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                                   rope_theta=10000.0, attn_implementation=impl, initializer_range=0.5)
    vt = MiniMaxM3VLVisionModel(vcfg).float().eval()
    grid = torch.tensor([[1, 4, 6], [2, 2, 2]])
    pa = torch.randn(24, 1176)
    pb = torch.randn(8, 1176)
    captured = []
    attn0 = vt.layers[0].self_attn
    orig_fwd = attn0.forward

    def spy(*args, **kwargs):
        out = orig_fwd(*args, **kwargs)
        captured.append((kwargs.get("attention_mask"), out[1]))
        return out

    attn0.forward = spy
    with torch.no_grad():
        ha = vt(pa, grid[:1]).last_hidden_state[0]
        hab = vt(torch.cat([pa, pb]), grid).last_hidden_state[0]
    print(f"[{impl}] max |A alone - A in batch|:", (ha - hab[:24]).abs().max().item(),
          "| mean |h|:", ha.abs().mean().item())
    mask, w = captured[-1]
    print(f"[{impl}] attention_mask seen by layer0 attn:", None if mask is None else tuple(mask.shape))
    if w is not None:
        print(f"[{impl}] attn weight mass from image-A queries onto image-B keys:", w[0, :, :24, 24:].sum(-1).mean().item())
