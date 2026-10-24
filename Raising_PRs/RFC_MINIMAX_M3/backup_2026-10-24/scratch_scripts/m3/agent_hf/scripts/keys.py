import json
import re
import sys
from collections import Counter, defaultdict

path = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else "patterns"
with open(path) as f:
    idx = json.load(f)
wm = idx["weight_map"]
print("total keys:", len(wm), "total_size:", idx.get("metadata"))

if mode == "patterns":
    pat = Counter()
    layers_for = defaultdict(set)
    for k in wm:
        g = re.sub(r"\.experts\.\d+\.", ".experts.E.", k)
        m = re.search(r"\.layers\.(\d+)\.", g)
        if m:
            layers_for[re.sub(r"\.layers\.\d+\.", ".layers.N.", g)].add(int(m.group(1)))
            g = re.sub(r"\.layers\.\d+\.", ".layers.N.", g)
        pat[g] += 1
    for g, c in sorted(pat.items()):
        ls = sorted(layers_for.get(g, []))
        lsdesc = ""
        if ls:
            lsdesc = f" layers[{len(ls)}]: {ls[0]}..{ls[-1]}" if ls == list(range(ls[0], ls[-1] + 1)) else f" layers{ls}"
        print(f"{c:7d}  {g}{lsdesc}")
elif mode == "grep":
    rx = re.compile(sys.argv[3])
    for k in sorted(wm):
        if rx.search(k):
            print(k, wm[k])
