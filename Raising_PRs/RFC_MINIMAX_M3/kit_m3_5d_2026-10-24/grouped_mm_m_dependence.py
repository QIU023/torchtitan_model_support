"""Is a row of torch._grouped_mm's output independent of how many rows share its group? (bf16, this GPU)"""
import torch
torch.manual_seed(0)
dev = "cuda"
D, F = 256, 256
w = torch.randn(2, F, D, device=dev, dtype=torch.bfloat16)          # two experts [E, F, D]
rows = torch.randn(8, D, device=dev, dtype=torch.bfloat16)
for m in (8, 16, 64, 128, 256, 512, 1024):
    extra = torch.randn(m - 8, D, device=dev, dtype=torch.bfloat16)
    x = torch.cat([rows, extra])                                        # expert 0 gets m rows, expert 1 gets 8
    x = torch.cat([x, rows])
    offs = torch.tensor([m, m + 8], device=dev, dtype=torch.int32)
    out = torch._grouped_mm(x, w.transpose(-2, -1), offs=offs)
    ref = rows @ w[0].T
    same = torch.equal(out[:8], ref)
    print(f"M={m:5d}: first 8 rows of expert 0 bitwise equal to a plain mm: {same}; max|diff| {(out[:8].float() - ref.float()).abs().max().item():.3e}")
