"""Compare compiled flex grads (static vs after a dynamic recompile) with an SDPA reference."""
import sys
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import AuxRequest, create_block_mask, create_mask
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.minimax_m3.attention import build_msa_metadata, msa_block_mask, select_msa_blocks

dev, dt = "cuda", torch.float32
torch.manual_seed(0)
flex = FlexInnerAttention.compiled_flex_attn
warm = sys.argv[1] == "dynamic"

def call(q, k, v, mask, gqa):
    out, aux = flex(q, k, v, score_mod=None, block_mask=mask, scale=None, enable_gqa=gqa,
                    return_aux=AuxRequest(lse=True), kernel_options={})
    return out

if warm:  # force automatic-dynamic recompiles like the vision tower does
    for n in (1296, 4):
        q, k, v = (torch.randn(1, 4, n, 16, device=dev, dtype=dt, requires_grad=True) for _ in range(3))
        call(q, k, v, create_block_mask(lambda b, h, qi, ki: qi >= 0, 1, 1, n, n, device=dev), False).sum().backward()

T, H, G, D, I = 2048, 8, 2, 32, 32
positions = torch.cat([torch.arange(T - 700), torch.arange(700)]).to(dev)
meta = build_msa_metadata(positions, block_size=16, tile_size=128, dense_mask=None)
sel = select_msa_blocks(torch.randn(T, G, I, device=dev), torch.randn(T, I, device=dev), meta, topk_blocks=4)
mask, _ = msa_block_mask(sel, meta, num_heads=H)
q_THK = torch.randn(T, H, D, device=dev, dtype=dt, requires_grad=True)
k_THK, v_THK = (torch.randn(T, G, D, device=dev, dtype=dt, requires_grad=True) for _ in range(2))
go = torch.randn(1, H, T, D, device=dev, dtype=dt)

out = call(q_THK.transpose(0, 1)[None], k_THK.transpose(0, 1)[None], v_THK.transpose(0, 1)[None], mask, True)
got = torch.autograd.grad(out, (q_THK, k_THK, v_THK), go)

dense = create_mask(mask.mask_mod, 1, H, T, T, device=dev)
ref_out = F.scaled_dot_product_attention(q_THK.transpose(0, 1)[None], k_THK.transpose(0, 1)[None].repeat_interleave(H // G, 1),
                                         v_THK.transpose(0, 1)[None].repeat_interleave(H // G, 1), attn_mask=dense)
ref = torch.autograd.grad(ref_out, (q_THK, k_THK, v_THK), go)
print(sys.argv[1], "out max|diff|", (out - ref_out).abs().max().item())
for name, a, b in zip("qkv", got, ref):
    print(f"  d{name} max|diff| {(a - b).abs().max().item():.3e}  (ref max {b.abs().max().item():.3e})")
