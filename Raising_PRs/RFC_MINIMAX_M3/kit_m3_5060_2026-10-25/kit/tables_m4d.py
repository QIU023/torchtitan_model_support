"""Paste-ready English tables of run_m3_4d.sh: one per data stream, steps 1 / 3 / 10 (step 1 to 7 decimals).
usage: tables_m4d.py <run dir>"""
import pathlib
import sys
from collections import defaultdict

out = pathlib.Path(sys.argv[1])
LABEL = {
    "dp1": "single GPU", "dp1_again": "single GPU again", "pp2_vpp2": "pp2 x vpp2 (Interleaved1F1B)",
    "fsdp2_again": "fsdp2 again", "fsdp4_again": "fsdp4 again", "fsdp8_again": "fsdp8 again",
    "hsdp2x2": "HSDP 2 x 2", "ep2_hsdp2x2": "ep2 x HSDP 2 x 2", "hsdp2x2_pp2": "HSDP 2 x 2 x pp2",
    "hsdp2x2_tp2_ep2": "HSDP 2 x 2 x tp2 x ep2", "hsdp2x4": "HSDP 2 x 4", "hsdp4x2": "HSDP 4 x 2",
    "ep4_hsdp2x4": "ep4 x HSDP 2 x 4",
}
REF = {"dp1": "dp1", "dp2": "fsdp2", "dp4": "fsdp4", "dp8": "fsdp8"}
cells = [line.split("|") for line in (out / "cells.txt").read_text().split()]


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


for stream in ("dp1", "dp2", "dp4", "dp8"):
    ref = REF[stream]
    members = [ref, f"{ref}_again"] + [c[0] for c in cells if c[3] == stream and c[0] != ref]
    print(f"\n{stream} ({len(members) - 1} layouts and the reference run twice):\n")
    print("| layout | step 1 loss / grad norm | step 3 loss / grad norm | step 10 loss / grad norm |")
    print("|---|---|---|---|")
    for name in members:
        v = steps(name)
        print(f"| {label(name)} | {v[1][0]:.7f} / {v[1][1]:.7f} | {v[3][0]:.5f} / {v[3][1]:.4f} | {v[10][0]:.5f} / {v[10][1]:.4f} |")
