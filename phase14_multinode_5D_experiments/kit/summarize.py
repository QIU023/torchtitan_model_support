"""Summarize the cells of one stage directory: python summarize.py <OUT>/<hw>/<stage> [--skip N].

Reads every <cell>/node<k>.log (launch.sh has torchrun prefix lines with [rank<global rank>]:), takes the TorchTitan
step lines of all ranks and prints one TSV row per cell: exit codes, OOM, steps, loss at the first and last step,
the peak reserved memory over ranks (and which rank), and the median tps / tflops / mfu / step time over steps after
--skip (default 3, the compile and warm-up steps), plus the first step's time (startup and compilation included). A
step's time is the gap between the rank's timestamps of that step line and the one before it ("Training starts" for
step 1). step_s_clean and mfu_clean are the medians over every step but the first and the profiled ones (multiples
of the cell's MN_PROFILE), the numbers to compare cells by. ckpt_save_s and ckpt_load_s are the longest checkpoint
save and load any rank logged (MN_CKPT_STEP).
"""

import argparse
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
PREFIX = re.compile(r"^\[(?:rank)?(\d+)\]:")
STEP = re.compile(
    r"step:\s*(\d+)\s+loss:\s*(-?[\d.]+|nan|inf)\s+grad_norm:\s*(-?[\d.]+|nan|inf)\s+"
    r"memory:\s*([\d.]+)GiB\(([\d.]+)%\)\s+tps:\s*([\d,]+)\s+tflops:\s*([\d,.]+)\s+mfu:\s*([\d.]+%|N/A)"
)
OOM = re.compile(r"OutOfMemoryError|CUDA out of memory")
TS = re.compile(r"\[titan\] (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3})")
STARTS = "Training starts at step"
CKPT_SAVE = re.compile(r"Finished (?:saving|staging) the checkpoint in ([\d.]+) seconds")
CKPT_LOAD = re.compile(r"Finished loading the checkpoint in ([\d.]+) seconds")


def cell_rows(cell: Path) -> tuple[list[dict], bool]:
    """Every step line of a cell's logs with its step time (step_s), and whether any rank ran out of memory."""
    rows, oom, last = [], False, {}
    for log in sorted(cell.glob("node*.log")):
        for raw in log.read_text(errors="replace").splitlines():
            line = ANSI.sub("", raw)
            oom = oom or bool(OOM.search(line))
            prefix = PREFIX.match(line)
            rank = int(prefix.group(1)) if prefix else -1
            ts = TS.search(line)
            t = datetime.strptime(ts.group(1), "%Y-%m-%d %H:%M:%S").timestamp() + int(ts.group(2)) / 1e3 if ts else None
            if t is not None and STARTS in line:
                last[rank] = t
            match = STEP.search(line)
            if not match:
                continue
            step, loss, _, mem, mem_pct, tps, tflops, mfu = match.groups()
            step_s = t - last[rank] if t is not None and rank in last else None
            if t is not None:
                last[rank] = t
            rows.append(
                dict(
                    rank=rank,
                    step=int(step),
                    loss=float(loss),
                    mem=float(mem),
                    mem_pct=float(mem_pct),
                    tps=float(tps.replace(",", "")),
                    tflops=float(tflops.replace(",", "")),
                    mfu=None if mfu == "N/A" else float(mfu.rstrip("%")),
                    step_s=step_s,
                )
            )
    return rows, oom


def profile_freq(cell: Path) -> int:
    env = cell / "env_node0.txt"
    text = env.read_text(errors="replace") if env.exists() else ""
    m = re.search(r"^MN_PROFILE=(\d+)$", text, re.M)
    return int(m.group(1)) if m else 0


def checkpoint_times(cell: Path) -> tuple[float | None, float | None]:
    save, load = [], []
    for log in cell.glob("node*.log"):
        text = log.read_text(errors="replace")
        save += [float(x) for x in CKPT_SAVE.findall(text)]
        load += [float(x) for x in CKPT_LOAD.findall(text)]
    return (max(save) if save else None), (max(load) if load else None)


def network(cell: Path) -> str:
    """NCCL's transport from the run's NCCL INFO files (launch.sh): IB+GDR, IB, Socket, or - when not recorded."""
    text = "".join(p.read_text(errors="replace") for p in cell.glob("nccl/*.log"))
    if not text:
        return "-"
    if "NET/IB : Using" in text:
        return "IB+GDR" if re.search(r"NET/IB/\d+/GDRDMA", text) else "IB"
    return "Socket" if "NET/Socket : Using" in text else "?"


def parse_cell(cell: Path, nproc: int, skip: int) -> dict:
    rows, oom = cell_rows(cell)
    rcs = ",".join(p.read_text().strip() for p in sorted(cell.glob("rc_node*"))) or "-"
    out = dict(cell=cell.name, rc=rcs, oom="OOM" if oom else "", steps=0, net=network(cell))
    if not rows:
        return out
    steps = sorted({r["step"] for r in rows})
    peak = max(rows, key=lambda r: r["mem"])
    late = [r for r in rows if r["step"] > skip] or rows
    losses = {r["step"]: r["loss"] for r in rows if r["loss"] > 0}

    def median(key: str, pool=None):
        values = [r[key] for r in (late if pool is None else pool) if r[key] is not None]
        return statistics.median(values) if values else float("nan")

    freq = profile_freq(cell)
    clean = [r for r in rows if r["step"] >= 2 and not (freq and r["step"] % freq == 0)]
    ckpt_save, ckpt_load = checkpoint_times(cell)

    out.update(
        steps=steps[-1],
        loss_first=losses.get(steps[0], float("nan")),
        loss_last=losses.get(steps[-1], float("nan")),
        peak_mem_gib=peak["mem"],
        peak_mem_pct=peak["mem_pct"],
        peak_rank=peak["rank"],
        tps=median("tps"),
        tflops=median("tflops"),
        mfu=median("mfu"),
        step_s=median("step_s"),
        first_step_s=statistics.median(
            [r["step_s"] for r in rows if r["step"] == steps[0] and r["step_s"] is not None] or [float("nan")]
        ),
        step_s_clean=median("step_s", clean),
        mfu_clean=median("mfu", clean),
        ckpt_save_s=ckpt_save,
        ckpt_load_s=ckpt_load,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage_dir", type=Path)
    parser.add_argument("--skip", type=int, default=3)
    parser.add_argument("--nproc", type=int, default=8)
    args = parser.parse_args()
    columns = [
        "cell", "rc", "oom", "steps", "loss_first", "loss_last",
        "peak_mem_gib", "peak_mem_pct", "peak_rank", "tps", "tflops", "mfu", "step_s", "first_step_s",
        "step_s_clean", "mfu_clean", "ckpt_save_s", "ckpt_load_s", "net",
    ]
    print("\t".join(columns))
    for cell in sorted(p for p in args.stage_dir.iterdir() if p.is_dir()):
        if not any(cell.glob("node*.log")):
            continue
        row = parse_cell(cell, args.nproc, args.skip)
        print("\t".join(
            f"{row[c]:.4f}" if isinstance(row.get(c), float) else "-" if c in row and row[c] is None else str(row.get(c, ""))
            for c in columns
        ))


if __name__ == "__main__":
    sys.exit(main())
