"""For each element where the ring rebuild misses fsdp4's gradient, try every sequential order of the four ranks in
each micro-batch and print which (order mb0, order mb1) pairs reproduce the run's value.
usage: reduce_orders.py <probe dir>"""
import itertools
import runpy
import sys

import torch

ns = runpy.run_path(sys.argv[0].replace("reduce_orders.py", "reduce_emulate.py"), run_name="emulate")
names, f_in, f_out = ns["names"], ns["f_in"], ns["f_out"]
orders = [p for p in itertools.permutations(range(4)) if p[0] < p[1]]
for n in names:
    rebuilt = ns["fsdp_tree"](f_in, n, (0, 1, 2, 3))
    for idx in (rebuilt != f_out[n]).nonzero().tolist():
        v = [[f_in[r][n][m].float()[tuple(idx)] for m in range(2)] for r in range(4)]

        def seq(order, m):
            acc = v[order[0]][m] + v[order[1]][m]
            for r in order[2:]:
                acc = acc + v[r][m]
            return acc

        hits = [(a, b) for a in orders for b in orders if torch.equal(seq(a, 0) + seq(b, 1), f_out[n][tuple(idx)])]
        chunk = -(-f_out[n].shape[0] // 4)
        print(f"{n}{idx} owner {idx[0] // chunk}: run {f_out[n][tuple(idx)].item()!r} ring {rebuilt[tuple(idx)].item()!r}; "
              f"{len(hits)} of {len(orders) ** 2} order pairs reproduce, e.g. {hits[:4]}")
        print("   inputs mb0", [float(v[r][0]) for r in range(4)], "mb1", [float(v[r][1]) for r in range(4)])
