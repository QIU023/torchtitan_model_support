"""PR 4312-format tables from step1_probe loss files: one table per data stream, loss and grad norm at the given steps,
each cell the raw value with "identical" or the % change against the stream's reference beneath it.
usage: tables_4312.py <run dir> <stream>:<reference>:<cell>,<cell>,... [...]   (cells without results are left out)
       tables_4312.py <run dir> auto   (streams and cells from the run dir's cells.txt, references dp1 / fsdp2 / fsdp4 / fsdp8)"""
import os
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
STEPS = [int(s) for s in os.environ.get("TABLE_STEPS", "1,10,50,100").split(",")]
LABEL = {
    "dp1": "single GPU", "dp1_again": "single GPU, again", "pp2_vpp2": "pp2 x vpp2 (Interleaved1F1B)",
    "fsdp2_again": "fsdp2, again", "fsdp4_again": "fsdp4, again", "fsdp8_again": "fsdp8, again",
    "hsdp2x2": "HSDP 2 x 2",
}


def label(name: str) -> str:
    return LABEL.get(name, name.replace("_", " x "))


def steps(name: str) -> dict[int, tuple[float, float]]:
    found: dict[int, tuple[float, float]] = {}
    for path in (out / "measured" / name).glob("loss_rank*.txt"):
        for line in filter(None, path.read_text().split("\n")):
            step, loss, grad = line.split()
            if float(loss) >= 0:
                found[int(step)] = (float(loss), float(grad))
    return found


def cell(value: float, ref: float, digits: int, is_ref: bool) -> str:
    raw = f"`{value:.{digits}f}`"
    if is_ref:
        return raw
    if value == ref:
        return f"{raw}<br>identical"
    return f"{raw}<br>{(value - ref) / abs(ref) * 100:+.2f}%"


def table(stream: str, ref: str, cells: list[str]) -> None:
    if not (out / "measured" / ref).exists():
        return
    ref_steps = steps(ref)
    rows = [name for name in [ref] + cells if (out / "measured" / name).exists() and max(steps(name), default=0) >= max(STEPS)]
    print(f"\n{stream}, reference {label(ref)}:\n")
    print("| cell | " + " | ".join(f"loss, step {s}" if i == 0 else f"step {s}" for i, s in enumerate(STEPS))
          + " | " + " | ".join(f"grad norm, step {s}" if i == 0 else f"step {s}" for i, s in enumerate(STEPS)) + " |")
    print("| --- |" + " ---: |" * (2 * len(STEPS)))
    for name in rows:
        v = steps(name)
        losses = [cell(v[s][0], ref_steps[s][0], 6, name == ref) for s in STEPS]
        grads = [cell(v[s][1], ref_steps[s][1], 4, name == ref) for s in STEPS]
        print(f"| {label(name)} | " + " | ".join(losses) + " | " + " | ".join(grads) + " |")


if sys.argv[2:] == ["auto"]:
    refs = {"dp1": "dp1", "dp2": "fsdp2", "dp4": "fsdp4", "dp8": "fsdp8"}
    specs = [line.split("|") for line in (out / "cells.txt").read_text().split()]
    for stream, ref in refs.items():
        table(stream, ref, [f"{ref}_again"] + [c[0] for c in specs if c[3] == stream and c[0] != ref])
else:
    for spec in sys.argv[2:]:
        stream, ref, cells = spec.split(":")
        table(stream, ref, cells.split(","))
