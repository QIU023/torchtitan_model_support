"""Summarize run_m3_4d.sh: one table per data stream (steps 1 / 3 / 10, step 1 in full precision), each cell against
its stream's reference, step-1 gradients against the reference for non-PP cells, and the error of every failed cell.
usage: summarize_m4d.py <run dir>"""
import pathlib
import re
import sys
from collections import defaultdict

import torch

out = pathlib.Path(sys.argv[1])
REF = {"dp1": "dp1", "dp2": "fsdp2", "dp4": "fsdp4", "dp8": "fsdp8"}
cells = [line.split("|") for line in (out / "cells.txt").read_text().split()]


def steps(name: str) -> dict[int, tuple[float, float]]:
    """step -> (loss, grad norm), the loss taken from the ranks that log it (the last PP stage under PP)."""
    per_step: dict[int, set] = defaultdict(set)
    for path in (out / "measured" / name).glob("loss_rank*.txt"):
        for line in path.read_text().split("\n"):
            if line.strip():
                step, loss, grad = line.split()
                if float(loss) >= 0:
                    per_step[int(step)].add((float(loss), float(grad)))
    for step, values in per_step.items():
        assert len(values) == 1, (name, step, values)
    return {step: next(iter(values)) for step, values in per_step.items()}


def error(name: str) -> str:
    log = out / "measured" / name / "run.log"
    if not log.exists():
        log = out / "warm" / name / "run.log"
    text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(errors="replace")) if log.exists() else ""
    hits = [line.strip() for line in text.split("\n") if re.search(r"(Error|Exception|refuse|not supported|must)", line)]
    return hits[-1][:300] if hits else "no error line"


def grads(ref: str, name: str) -> str:
    a_path, b_path = out / "measured" / ref / "grads.pt", out / "measured" / name / "grads.pt"
    if not (a_path.exists() and b_path.exists()):
        return ""
    a, b = torch.load(a_path), torch.load(b_path)
    if a.keys() != b.keys():
        return f"different parameter sets ({len(a)} vs {len(b)})"
    groups: dict[str, list] = defaultdict(lambda: [0.0, ""])
    equal = flips = 0
    for key in a:
        x, y = a[key], b[key]
        equal += int(torch.equal(x, y))
        flips += int(((x * y) < 0).sum())
        rel = ((x - y).norm() / x.norm().clamp_min(1e-30)).item()
        group = re.sub(r"\.\d+\.", ".N.", key, count=1)
        if rel > groups[group][0]:
            groups[group] = [rel, key]
    worst_group, (worst, _) = max(groups.items(), key=lambda kv: kv[1][0])
    return f"{equal}/{len(a)} bitwise, worst rel L2 {worst:.2e} ({worst_group}), sign flips {flips}"


def fmt(values: dict, step: int, digits: int) -> str:
    if step not in values:
        return "-"
    loss, grad = values[step]
    return f"{loss:.{digits}f} / {grad:.{digits}f}"


for stream in ("dp1", "dp2", "dp4", "dp8"):
    ref = REF[stream]
    members = [ref, f"{ref}_again"] + [c[0] for c in cells if c[3] == stream and c[0] != ref]
    ref_steps = steps(ref)
    print(f"\n### {stream} (reference {ref})\n")
    print("| cell | step 1 loss / grad norm | step 3 | step 10 | step-1 loss vs ref | loss parts at | grad norm parts at | step-1 gradients vs ref |")
    print("|---|---|---|---|---|---|---|---|")
    for name in members:
        values = steps(name) if (out / "measured" / name).exists() else {}
        if not values or max(values) < 20:
            print(f"| {name} | failed or short ({len(values)} steps): {error(name)} | | | | | | |")
            continue
        same1 = values[1][0] == ref_steps[1][0]
        loss_part = next((s for s in sorted(values) if values[s][0] != ref_steps.get(s, (None,))[0]), None)
        grad_part = next((s for s in sorted(values) if values[s][1] != ref_steps.get(s, (None, None))[1]), None)
        print(
            f"| {name} | {fmt(values, 1, 7)} | {fmt(values, 3, 5)} | {fmt(values, 10, 5)} | "
            f"{'bitwise' if same1 else f'{values[1][0] - ref_steps[1][0]:+.2e}'} | "
            f"{loss_part or 'never (20 steps)'} | {grad_part or 'never (20 steps)'} | {grads(ref, name) if name != ref else ''} |"
        )
print("\nFull-precision step 1 per cell:")
for name, *_ in cells:
    for suffix in ("", "_again"):
        path = out / "measured" / f"{name}{suffix}"
        if path.exists():
            values = steps(f"{name}{suffix}")
            if 1 in values:
                print(f"  {name}{suffix}: loss {values[1][0]!r} grad norm {values[1][1]!r}")
