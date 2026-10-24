"""Same 10 rows through torch._grouped_mm in a group of exactly 10 rows vs a larger group, for w13- and w2-like shapes."""
import torch
torch.manual_seed(0)
dev = "cuda"
for name, (out_features, in_features) in {"w13 [2F, D]": (256, 256), "w2 [D, F]": (256, 128)}.items():
    w = torch.randn(8, out_features, in_features, device=dev, dtype=torch.bfloat16)
    rows = torch.randn(10, in_features, device=dev, dtype=torch.bfloat16)
    def run(sizes, target, extra_rows):
        parts, offs, total = [], [], 0
        for e, n in enumerate(sizes):
            block = torch.randn(n, in_features, device=dev, dtype=torch.bfloat16)
            if e == target:
                block = torch.cat([rows, torch.randn(extra_rows, in_features, device=dev, dtype=torch.bfloat16)])
                n = block.shape[0]
            parts.append(block); total += n; offs.append(total)
        x = torch.cat(parts)
        out = torch._grouped_mm(x, w.transpose(-2, -1), offs=torch.tensor(offs, device=dev, dtype=torch.int32))
        start = offs[target - 1] if target else 0
        return out[start:start + 10]
    sizes = [112, 133, 36, 183, 174, 10, 264, 112]
    alone = run(sizes, 5, 0)
    for extra in (1, 6, 14, 22, 54):
        bigger = run([s if i != 5 else 10 + extra for i, s in enumerate(sizes)], 5, extra)
        print(f"{name}: group of 10 vs group of {10 + extra}: bitwise={torch.equal(alone, bigger)} max|diff|={(alone.float() - bigger.float()).abs().max().item():.3e}")
    print(f"{name}: group of 10 vs plain mm: bitwise={torch.equal(alone, rows @ w[5].T)}")
