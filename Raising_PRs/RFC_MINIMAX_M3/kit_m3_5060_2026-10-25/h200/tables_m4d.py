"""Paste-ready English tables of run_m3_4d.sh: one per data stream, steps 1 / 3 / 10 (step 1 to 7 decimals).
usage: tables_m4d.py <run dir>"""
import pathlib
import sys
from collections import defaultdict

out = pathlib.Path(sys.argv[1])
import os

COLS = [int(c) for c in os.environ.get("M4D_COLS", "1,3,10").split(",")]
SKIP = set(filter(None, os.environ.get("M4D_SKIP", "").split(",")))
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
    print(f"\n{stream}:\n")
    print("| layout | " + " | ".join(f"step {c} loss / grad norm" for c in COLS) + " |")
    print("|---|" + "---|" * len(COLS))
    for name in members:
        v = steps(name) if (out / "measured" / name).exists() and name not in SKIP else {}
        if not v:
            continue
        cols = [f"{v[c][0]:.7f} / {v[c][1]:.7f}" if c == 1 else f"{v[c][0]:.5f} / {v[c][1]:.4f}" for c in COLS]
        print(f"| {label(name)} | " + " | ".join(cols) + " |")
