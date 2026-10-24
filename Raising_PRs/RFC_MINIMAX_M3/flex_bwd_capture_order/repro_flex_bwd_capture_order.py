"""flex_attention backward autotuning binds two captured mask_mod tensors in the wrong order.

Run on a cold inductor cache, e.g. under
  PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool memcheck python this.py
"""
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

T, H, D, B = 2048, 4, 64, 128
dev = "cuda"
block_of_key = torch.arange(T, device=dev) // B                      # [T], read at kv
picked = (torch.arange(T, device=dev) // B)[:, None].repeat(1, 4)    # [T, 4], read at q


def mask_mod(b, h, q_idx, kv_idx):
    key_block = block_of_key[kv_idx]          # traced (lifted) first
    hit = picked[q_idx, 0] == key_block       # but the kernel loads `picked` first
    for s in range(1, 4):
        hit = hit | (picked[q_idx, s] == key_block)
    return hit & (kv_idx <= q_idx)


block_mask = create_block_mask(mask_mod, 1, 1, T, T, device=dev)
flex = torch.compile(flex_attention, options={"max_autotune": True})
q, k, v = (torch.randn(1, H, T, D, device=dev, dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
flex(q, k, v, block_mask=block_mask).sum().backward()
torch.cuda.synchronize()
print("done")
