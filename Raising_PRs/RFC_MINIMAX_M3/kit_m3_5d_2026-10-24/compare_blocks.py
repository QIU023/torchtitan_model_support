"""Compare rank 0's first-forward block outputs of two step-1 probes, layer by layer.
A candidate whose output is a sequence shard (SP) is compared with the matching leading rows of the reference.
usage: compare_blocks.py <reference blocks.pt> <candidate blocks.pt>"""
import sys

import torch

ref, cand = torch.load(sys.argv[1]), torch.load(sys.argv[2])
for name in sorted(set(ref) & set(cand), key=lambda n: int(n.split(".")[1])):
    a, b = ref[name], cand[name]
    if b.shape != a.shape:
        a = a[: b.shape[0]]
    d = (a - b).abs()
    print(f"{name:10s} rows={b.shape[0]:5d} bitwise={torch.equal(a, b)!s:5s} max|diff|={d.max().item():.3e} "
          f"rel={d.max().item() / a.abs().max().item():.2e} differing={int((d > 0).sum())}/{d.numel()}")
