"""Summarize a run_m3_text18.sh output dir: loss and grad norm at steps 1 / 10 / 20 per cell, grouped by dp degree
(cells with the same dp read the same samples), with the step-1 loss gap to the group's reference.
usage: summarize_m3_text18.py <out dir>"""
import pathlib
import re
import sys

PAT = re.compile(r"step:\s*(\d+)\s+loss:\s*([-\d.e]+)\s+grad_norm:\s*([-\d.e]+)")
CELLS = [("dp1", 1), ("pp2", 1), ("pp4", 1), ("pp8", 1), ("tp2_ep2", 1), ("tp4_ep4", 1),
         ("fsdp2", 2), ("ep2_fsdp2", 2), ("ep2_fsdp2_tp2_pp2", 2), ("ep4_fsdp2_tp2_pp2", 2), ("ep8_fsdp8", 8)]
NOT_RUN = ["cp2", "cp4", "ep2_tp2_pp2_cp2", "fsdp2_pp2_cp2", "ep2_fsdp2_pp2_cp2", "ep4_fsdp2_tp2_cp2", "ep2_fsdp2_tp2_cp2"]


def read(path: pathlib.Path):
    log = path / "run.log"
    if not log.exists():
        return {}, 0
    text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(errors="replace"))
    steps = {}
    for step, loss, grad in PAT.findall(text):
        if not loss.startswith("-"):
            steps.setdefault(int(step), (loss, grad))
    return steps, text.count("Loading the checkpoint")


out = pathlib.Path(sys.argv[1])
rows, refs, ok = [], {}, 0
for cell, dp in CELLS + [("dp1_rerun", 1), ("dp1_fresh_cache", 1)]:
    phase = "noise" if cell.startswith("dp1_") else "measured"
    steps, loads = read(out / phase / cell)
    if not steps:
        rows.append(f"| {cell} | dp{dp} | no loss lines | | | |")
        continue
    ref = refs.setdefault(dp, steps[1][0])
    gap = abs(float(steps[1][0]) - float(ref))
    ok += phase == "measured" and max(steps) == 20 and loads > 0
    cols = " | ".join(f"{steps[s][0]} / {steps[s][1]}" if s in steps else "-" for s in (1, 10, 20))
    rows.append(f"| {cell} | dp{dp} | {cols} | {gap:.5f} | {loads} |")
print("| cell | data | step 1 loss / grad norm | step 10 | step 20 | step-1 loss gap to dp ref | seed loads |")
print("|---|---|---|---|---|---|---|")
print("\n".join(rows))
print(f"\n{ok} / {len(CELLS)} measured cells ran 20 steps from the seed; not run (CP not supported): {', '.join(NOT_RUN)}")
