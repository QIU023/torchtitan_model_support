"""Rebuild the step-1 loss of a non-PP cell and its PP variant from their per-micro-batch losses (M3_DUMP_MB_LOSSES):
the non-PP engine adds them one by one into a clone of the first, the PP engine sums torch.stack of them.
usage: mb_loss_sums.py <non-PP probe dir> <PP probe dir>"""
import pathlib
import sys

import torch


def read(directory: pathlib.Path) -> tuple[list[list[float]], float]:
    lists = [
        [float(x) for x in path.read_text().split()]
        for path in sorted(directory.glob("mb_losses_rank*.txt"))
    ]
    logged = {
        float(line.split()[1])
        for path in directory.glob("loss_rank*.txt")
        for line in path.read_text().split("\n")
        if line.strip() and line.split()[0] == "1" and float(line.split()[1]) >= 0
    }
    assert len(logged) == 1, logged
    return lists, logged.pop()


def sequential(values: list[float]) -> float:
    t = [torch.tensor(v, dtype=torch.float32, device="cuda") for v in values]
    acc = t[0].clone()
    for x in t[1:]:
        acc.add_(x)
    return float(acc)


def stacked(values: list[float]) -> float:
    t = [torch.tensor(v, dtype=torch.float32, device="cuda") for v in values]
    return float(torch.sum(torch.stack(t)))


non_pp_lists, non_pp_logged = read(pathlib.Path(sys.argv[1]))
pp_lists, pp_logged = read(pathlib.Path(sys.argv[2]))
non_pp = next(v for v in non_pp_lists if v)
pp = next(v for v in pp_lists if v)
print(f"non-PP ranks with micro-batch losses: {sum(bool(v) for v in non_pp_lists)}; PP ranks: {sum(bool(v) for v in pp_lists)}")
print(f"{len(non_pp)} micro-batch losses each; bitwise equal across the two cells: {non_pp == pp}")
for i, (a, b) in enumerate(zip(non_pp, pp)):
    print(f"  mb{i}: {a!r} {'==' if a == b else '!='} {b!r}")
print(f"non-PP logged {non_pp_logged!r}; sequential add of its losses {sequential(non_pp)!r}; stacked sum {stacked(non_pp)!r}")
print(f"PP logged     {pp_logged!r}; sequential add of its losses {sequential(pp)!r}; stacked sum {stacked(pp)!r}")
