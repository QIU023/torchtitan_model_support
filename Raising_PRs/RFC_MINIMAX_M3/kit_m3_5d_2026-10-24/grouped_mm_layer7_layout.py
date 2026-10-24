"""The same 10 rows of one expert through torch._grouped_mm in layer 7's non-EP and EP group layouts (bf16, this GPU)."""
import torch
torch.manual_seed(0)
dev = "cuda"
def run(w, rows, sizes, target):
    parts = []
    for e, n in enumerate(sizes):
        parts.append(torch.cat([rows, torch.randn(n - 10, rows.shape[1], device=dev, dtype=rows.dtype)]) if e == target
                     else torch.randn(n, rows.shape[1], device=dev, dtype=rows.dtype))
    offs = torch.tensor(sizes, device=dev).cumsum(0).to(torch.int32)
    out = torch._grouped_mm(torch.cat(parts), w.transpose(-2, -1), offs=offs)
    start = 0 if target == 0 else int(offs[target - 1])
    return out[start:start + 10]
for name, (n_out, n_in) in {"w13": (256, 256), "w2": (256, 128)}.items():
    w8 = torch.randn(8, n_out, n_in, device=dev, dtype=torch.bfloat16)
    rows = torch.randn(10, n_in, device=dev, dtype=torch.bfloat16)
    local = run(w8, rows, [112, 133, 36, 183, 174, 10, 264, 112], 5)          # non-EP, rank 0
    ep = run(w8[4:], rows, [330, 28, 564, 217], 1)                              # EP, rank 1 holds experts 4..7
    print(f"{name}: non-EP layout vs EP layout for expert 5's rows: bitwise={torch.equal(local, ep)} "
          f"max|diff|={(local.float() - ep.float()).abs().max().item():.3e} differing={int((local != ep).sum())}/{local.numel()}")
