"""Compare two step-1 gradient dumps: per parameter group, the max relative difference and the sign flips.
usage: compare_grads.py <reference.pt> <candidate.pt>"""
import re
import sys
from collections import defaultdict

import torch

ref, cand = torch.load(sys.argv[1]), torch.load(sys.argv[2])
assert ref.keys() == cand.keys(), sorted(set(ref) ^ set(cand))[:5]
groups = defaultdict(lambda: [0, 0.0, 0, 0, 0])  # params, max rel, flips, elements, bitwise-equal params
for name in ref:
    a, b = ref[name], cand[name]
    key = re.sub(r"\.\d+\.", ".N.", name, count=1)
    key = re.sub(r"\.(weight|bias|w13_EFD|w2_EDF|\w+_E)$", "", key)
    g = groups[key]
    g[0] += 1
    g[1] = max(g[1], ((a - b).norm() / a.norm().clamp_min(1e-30)).item())
    g[2] += int(((a * b) < 0).sum().item())
    g[3] += a.numel()
    g[4] += int(torch.equal(a, b))
total = len(ref)
equal = sum(g[4] for g in groups.values())
worst = max(g[1] for g in groups.values())
print(f"{equal}/{total} parameters bitwise equal; worst group relative L2 difference {worst:.2e}")
for key, (n, rel, flips, numel, eq) in sorted(groups.items(), key=lambda kv: -kv[1][1])[:12]:
    print(f"  {key:60s} params={n:3d} rel_l2={rel:.2e} sign_flips={flips}/{numel} bitwise={eq}/{n}")
