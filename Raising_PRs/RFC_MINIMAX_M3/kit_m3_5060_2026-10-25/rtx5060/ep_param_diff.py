"""Which step-1 gradients differ between each non-PP EP layout and the same layout without EP.
usage: ep_param_diff.py <matrix run dir> <ref>:<cell> [...]"""
import re
import sys

import torch

out = sys.argv[1]
for pair in sys.argv[2:]:
    ref, cell = pair.split(":")
    a = torch.load(f"{out}/measured/{ref}/grads.pt")
    b = torch.load(f"{out}/measured/{cell}/grads.pt")
    differ = [k for k in a if not torch.equal(a[k], b[k])]
    groups = sorted({re.sub(r"^layers\.\d+\.", "layers.N.", k) for k in differ})
    print(f"{cell} vs {ref}: {len(differ)}/{len(a)} differ -> {groups}")
