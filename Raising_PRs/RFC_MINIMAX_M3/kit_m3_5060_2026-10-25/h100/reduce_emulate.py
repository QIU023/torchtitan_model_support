"""Locate the HSDP vs FSDP step-1 gradient gap from the reduce-input dumps (M3_DUMP_REDUCE_INPUTS).

1. Every rank's per-micro-batch gradients entering FSDP's reduction must be bitwise equal in the two layouts.
2. Rebuild each layout's reduced step-1 gradients from those inputs on CPU, in fp32, with its own summation tree,
   and compare with the gradients the run produced (grads.pt):
   - HSDP 2 x 2 (titan all-reduces replicas on the last micro-batch only):
     [(x0_1 + x1_1) + (x0_0 + x1_0)] + [(x2_1 + x3_1) + (x2_0 + x3_0)]   (xr_m = rank r, micro-batch m)
     except lm_head, whose last loss chunk calls set_requires_gradient_sync(True), which also re-enables the all-reduce,
     so it all-reduces every micro-batch: [(x0_0 + x1_0) + (x2_0 + x3_0)] + [(x0_1 + x1_1) + (x2_1 + x3_1)]
   - FSDP 4: a ring reduce-scatter per micro-batch, then the two micro-batches added; for the dim-0 chunk owned by
     rank k the ring adds ((x_{k+1} + x_{k+2}) + x_{k+3}) + x_k along the ring order, tried for every ring; chunks follow
     titan's placements: stacked Linear weights ([2, F, D]: dense and shared-expert w13) Shard(1), the rest Shard(0).
usage: reduce_emulate.py <probe dir holding fsdp4/ and hsdp2x2/>
"""
import itertools
import sys
from pathlib import Path

import torch

D = Path(sys.argv[1])
RANKS = 4


def load(cell):
    inputs = [torch.load(D / cell / f"reduce_inputs_rank{r}.pt") for r in range(RANKS)]
    return inputs, torch.load(D / cell / "grads.pt")


f_in, f_out = load("fsdp4")
h_in, h_out = load("hsdp2x2")
names = sorted(f_in[0])
assert all(set(f_in[r]) == set(names) == set(h_in[r]) for r in range(RANKS)), "parameter sets differ"
assert set(names) <= set(f_out), sorted(set(names) - set(f_out))[:5]
calls = {len(f_in[r][n]) for r in range(RANKS) for n in names} | {len(h_in[r][n]) for r in range(RANKS) for n in names}
dtypes = sorted({str(t.dtype) for r in range(RANKS) for n in names for t in f_in[r][n]})
print(f"{len(names)} reduced parameters, micro-batches per parameter {sorted(calls)}, input dtypes {dtypes}")

pairs = [(r, n, m) for r in range(RANKS) for n in names for m in range(len(f_in[r][n]))]
unequal = [p for p in pairs if not torch.equal(f_in[p[0]][p[1]][p[2]], h_in[p[0]][p[1]][p[2]])]
print(f"1. reduce inputs bitwise equal across layouts: {len(pairs) - len(unequal)}/{len(pairs)} (rank, parameter, micro-batch)")


def x(inputs, r, n, m):
    return inputs[r][n][m].float()


def hsdp_tree(inputs, n):
    def pair(a, b, m):
        return x(inputs, a, n, m) + x(inputs, b, n, m)

    if n == "lm_head.weight":
        return (pair(0, 1, 0) + pair(2, 3, 0)) + (pair(0, 1, 1) + pair(2, 3, 1))
    return (pair(0, 1, 1) + pair(0, 1, 0)) + (pair(2, 3, 1) + pair(2, 3, 0))


def shard_dim(n, t):
    return 1 if t.ndim == 3 and "routed_experts" not in n else 0


def ring_sum(inputs, n, m, ring):
    """Ring reduce-scatter of one micro-batch: each dim-0 chunk accumulates from the owner's successor round to it."""
    dim = shard_dim(n, inputs[0][n][m])
    full = x(inputs, 0, n, m).movedim(dim, 0)
    out = torch.empty_like(full)
    chunk = -(-full.shape[0] // RANKS)
    pos = {rank: i for i, rank in enumerate(ring)}
    for owner in range(RANKS):
        rows = slice(owner * chunk, min((owner + 1) * chunk, full.shape[0]))
        if rows.start >= full.shape[0]:
            continue
        order = [ring[(pos[owner] + j) % RANKS] for j in range(1, RANKS)] + [owner]
        acc = x(inputs, order[0], n, m).movedim(dim, 0)[rows] + x(inputs, order[1], n, m).movedim(dim, 0)[rows]
        for r in order[2:]:
            acc = acc + x(inputs, r, n, m).movedim(dim, 0)[rows]
        out[rows] = acc
    return out.movedim(0, dim)


def fsdp_tree(inputs, n, ring):
    return ring_sum(inputs, n, 0, ring) + ring_sum(inputs, n, 1, ring)


def score(rebuilt, produced):
    params = sum(torch.equal(rebuilt[n], produced[n]) for n in names)
    elems = sum(int((rebuilt[n] == produced[n]).sum()) for n in names)
    total = sum(produced[n].numel() for n in names)
    return params, elems, total


h_rebuilt = {n: hsdp_tree(h_in, n) for n in names}
p, e, t = score(h_rebuilt, h_out)
print(f"2. HSDP tree rebuilt from the inputs vs HSDP's step-1 gradients: {p}/{len(names)} parameters, {e}/{t} elements bitwise")

best = None
for perm in itertools.permutations(range(1, RANKS)):
    ring = (0,) + perm
    f_rebuilt = {n: fsdp_tree(f_in, n, ring) for n in names}
    p, e, t = score(f_rebuilt, f_out)
    print(f"3. FSDP ring {ring} rebuilt vs fsdp4's step-1 gradients: {p}/{len(names)} parameters, {e}/{t} elements bitwise")
    if best is None or e > best[1]:
        best = (ring, e, f_rebuilt)

ring, _, f_rebuilt = best
p, e, t = score(f_rebuilt, h_rebuilt)
print(f"4. the two rebuilt trees against each other (ring {ring}): {p}/{len(names)} parameters, {e}/{t} elements bitwise")
p, e, t = score(f_out, h_out)
print(f"   the two runs against each other: {p}/{len(names)} parameters, {e}/{t} elements bitwise")
zero = sum(not f_out[n].any() for n in f_out)
print(f"   grads.pt holds {len(f_out)} parameters, {zero} of them all-zero, {len(f_out) - len(names)} not reduced by FSDP")
