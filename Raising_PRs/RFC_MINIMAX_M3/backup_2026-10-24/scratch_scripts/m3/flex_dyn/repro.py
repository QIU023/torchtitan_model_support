"""Replay rank 0's compiled_flex_attn call sequence from the 2-GPU run on a cold cache."""
import sys
import torch
from torch.nn.attention.flex_attention import AuxRequest, create_block_mask
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.minimax_m3.attention import build_msa_metadata, msa_block_mask, select_msa_blocks

dev, dt = "cuda", torch.bfloat16
torch.manual_seed(0)
flex = FlexInnerAttention.compiled_flex_attn
seq = sys.argv[1].split(",")  # e.g. "vis1296,vis4,msa"

def run(q, k, v, mask, gqa):
    out, aux = flex(q, k, v, score_mod=None, block_mask=mask, scale=None, enable_gqa=gqa,
                    return_aux=AuxRequest(lse=True), kernel_options={})
    (out.float().sum() + aux.lse.float().sum()).backward()
    torch.cuda.synchronize()

for item in seq:
    if item.startswith("vis"):
        n = int(item[3:])
        q, k, v = (torch.randn(1, 4, n, 16, device=dev, dtype=dt, requires_grad=True) for _ in range(3))
        mask = create_block_mask(lambda b, h, qi, ki: qi >= 0, 1, 1, n, n, device=dev)
        run(q, k, v, mask, False)
    else:
        T, H, G, D, I = 2048, 8, 2, 32, 32
        positions = torch.cat([torch.arange(T - 700), torch.arange(700)]).to(dev)
        meta = build_msa_metadata(positions, block_size=16, tile_size=128, dense_mask=None)
        iq = torch.randn(T, G, I, device=dev); ik = torch.randn(T, I, device=dev)
        sel = select_msa_blocks(iq, ik, meta, topk_blocks=4)
        mask, _ = msa_block_mask(sel, meta, num_heads=H)
        q = torch.randn(1, H, T, D, device=dev, dtype=dt, requires_grad=True)
        k, v = (torch.randn(1, G, T, D, device=dev, dtype=dt, requires_grad=True) for _ in range(2))
        run(q, k, v, mask, True)
    print("ok", item, flush=True)
