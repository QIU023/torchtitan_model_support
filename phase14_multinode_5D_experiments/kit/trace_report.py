"""Time breakdown of the profiled step of every cell in one stage: python trace_report.py <OUT>/<hw>/<stage> [--nproc 8].

For each rank's trace (<cell>/tt/profiling/traces/iteration_<last>/rank<r>_trace.json.gz) the profiled step on the GPU
is split into compute (a non-NCCL kernel runs; NCCL kernels running at the same time count as overlapped), exposed
communication (only NCCL kernels run), copies (only memcpy / memset run) and idle. Each NCCL kernel is attributed to a
process group through the record_param_comms op that launched it; unattributed_pct is the share of NCCL kernel time
that found none. An NCCL kernel's time includes waiting for its peers, so exposed communication also holds load
imbalance and pipeline bubbles, and eff_busbw (message bytes over kernel time, nccl-tests factors) is a lower bound.
Writes trace_report.tsv (one row per cell, joined with summarize.py's tps / mfu / peak memory and monitor.py's
InfiniBand rates), trace_report_pg.tsv (per cell, process group and collective) and trace_report_rank.tsv (per cell
and rank) into the stage directory, and prints the first.
"""

import argparse
import bisect
import gzip
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
LAUNCH_CATS = {"cuda_runtime", "cuda_driver"}
DTYPE_BYTES = {
    "Float": 4, "BFloat16": 2, "Half": 2, "Double": 8, "Long": 8, "Int": 4, "Short": 2, "Char": 1, "Byte": 1,
    "Bool": 1, "Float8_e4m3fn": 1, "Float8_e5m2": 1, "Float8_e4m3fnuz": 1, "Float8_e5m2fnuz": 1,
}
# Process group description -> dimension; the first match wins, so the longer names come first.
LABELS = (
    ("edp_shard", "EFSDP"), ("dp_replicate", "DPR"), ("loss", "LOSS"), ("pp", "PP"), ("ep", "EP"),
    ("dp_shard", "FSDP"), ("fsdp", "FSDP"), ("dp", "DP"), ("cp", "CP"), ("tp", "TP"), ("world", "WORLD"),
    ("default", "WORLD"),
)
NEAR_US = 5000  # a launch this soon after a record_param_comms op (same thread) belongs to it
CELL_COLUMNS = [
    "cell", "ranks", "step_ms", "tps", "mfu", "peak_mem_gib", "peak_rank", "compute_pct", "exposed_pct",
    "exposed_max_pct", "idle_pct", "idle_max_pct", "copy_pct", "overlap_pct", "unattributed_pct", "top_exposed",
    "ib_tx_GBps", "ib_rx_GBps",
]
PG_COLUMNS = [
    "cell", "label", "group", "collective", "group_size", "kernels", "kernel_ms", "exposed_ms", "exposed_pct",
    "mbytes", "eff_busbw",
]
RANK_COLUMNS = [
    "cell", "rank", "node", "window_ms", "compute_ms", "exposed_ms", "copy_ms", "idle_ms", "comm_ms",
    "unattributed_ms",
]


def label_of(desc: str) -> str:
    d = desc.lower()
    for key, label in LABELS:
        if key in d:
            return label
    return desc


def bus_factor(collective: str, n: int) -> float:
    c = collective.lower()
    if n < 2:
        return 1.0
    if "all_reduce" in c or "allreduce" in c:
        return 2 * (n - 1) / n
    if any(k in c for k in ("gather", "reduce_scatter", "reducescatter", "all_to_all", "alltoall")):
        return (n - 1) / n
    return 1.0


def kernel_collective(name: str) -> str:
    m = re.match(r"nccl(?:Dev)?Kernel_([A-Za-z]+)", name)
    return m.group(1) if m else name.split("(")[0]


def message_bytes(args: dict) -> int:
    elem = DTYPE_BYTES.get(str(args.get("dtype", "")).split("::")[-1], 0)
    try:
        nelems = max(int(args.get("In msg nelems") or 0), int(args.get("Out msg nelems") or 0))
    except (TypeError, ValueError):
        return 0
    return nelems * elem


def load_trace(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def analyze_rank(path_str: str) -> dict:
    path = Path(path_str)
    rank = int(re.search(r"rank(\d+)_trace", path.name).group(1))
    try:
        events = load_trace(path).get("traceEvents", [])
    except (OSError, EOFError, ValueError):
        return dict(rank=rank, unreadable=True)
    comms: dict[tuple, list] = defaultdict(list)
    launches: dict = {}
    gpu = []
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat")
        if cat in GPU_CATS:
            gpu.append(e)
        elif cat in LAUNCH_CATS:
            corr = e.get("args", {}).get("correlation")
            if corr is not None:
                launches[corr] = (e.get("pid"), e.get("tid"), float(e["ts"]))
        elif cat == "cpu_op" and e.get("name") == "record_param_comms":
            ts = float(e["ts"])
            comms[(e.get("pid"), e.get("tid"))].append((ts, ts + float(e.get("dur", 0)), e.get("args", {})))
    starts = {}
    for key, ops in comms.items():
        ops.sort(key=lambda op: op[0])
        starts[key] = [op[0] for op in ops]

    def comm_args(kernel: dict):
        args = kernel.get("args", {})
        if "Process Group Description" in args or "Collective name" in args:
            return args
        launch = launches.get(args.get("correlation"))
        if launch is None:
            return None
        key = (launch[0], launch[1])
        if key not in comms:
            return None
        i = bisect.bisect_right(starts[key], launch[2]) - 1
        if i < 0:
            return None
        ts, end, op_args = comms[key][i]
        return op_args if launch[2] <= end or launch[2] - ts <= NEAR_US else None

    intervals = []  # (start, end, kind 0 compute / 1 comm / 2 copy, group key)
    groups: dict[tuple, list] = defaultdict(lambda: [0, 0.0, 0])  # key -> [kernels, kernel_us, bytes]
    unattributed = 0.0
    for e in gpu:
        ts, dur = float(e["ts"]), float(e.get("dur", 0))
        if dur <= 0:
            continue
        name = e.get("name", "")
        if e.get("cat") == "kernel" and name.startswith("nccl"):
            args = comm_args(e)
            if args:
                desc = str(args.get("Process Group Description") or args.get("Process Group Name") or "?")
                collective = str(args.get("Collective name") or kernel_collective(name))
                try:
                    n = int(args.get("Group size") or 0)
                except (TypeError, ValueError):
                    n = 0
                nbytes = message_bytes(args)
            else:
                desc, collective, n, nbytes = "?", kernel_collective(name), 0, 0
                unattributed += dur
            key = ("?" if desc == "?" else label_of(desc), desc, collective, n)
            g = groups[key]
            g[0] += 1
            g[1] += dur
            g[2] += nbytes
            intervals.append((ts, ts + dur, 1, key))
        else:
            intervals.append((ts, ts + dur, 0 if e.get("cat") == "kernel" else 2, None))

    bounds = []
    for i, (s, t, _, _) in enumerate(intervals):
        bounds.append((s, 1, i))
        bounds.append((t, -1, i))
    bounds.sort(key=lambda b: (b[0], b[1]))
    acc = dict(compute=0.0, exposed=0.0, copy=0.0, idle=0.0, comm=0.0)
    exposed_by: dict[tuple, float] = defaultdict(float)
    active = [0, 0, 0]
    comm_active: dict[tuple, int] = defaultdict(int)
    prev = bounds[0][0] if bounds else 0.0
    for t, delta, i in bounds:
        seg = t - prev
        if seg > 0:
            if active[1]:
                acc["comm"] += seg
            if active[0]:
                acc["compute"] += seg
            elif active[1]:
                acc["exposed"] += seg
                total = sum(comm_active.values())
                for key, count in comm_active.items():
                    exposed_by[key] += seg * count / total
            elif active[2]:
                acc["copy"] += seg
            else:
                acc["idle"] += seg
        prev = t
        kind, key = intervals[i][2], intervals[i][3]
        active[kind] += delta
        if kind == 1:
            comm_active[key] += delta
            if comm_active[key] == 0:
                del comm_active[key]
    window = bounds[-1][0] - bounds[0][0] if bounds else 0.0
    return dict(
        rank=rank,
        window=window,
        acc=acc,
        unattributed=unattributed,
        groups=[(key, g[0], g[1], g[2], exposed_by.get(key, 0.0)) for key, g in groups.items()],
    )


def trace_files(cell: Path) -> list[Path]:
    root = cell / "tt" / "profiling" / "traces"
    iterations = sorted(
        (p for p in root.glob("iteration_*") if p.name.split("_")[-1].isdigit()),
        key=lambda p: int(p.name.split("_")[-1]),
    )
    if not iterations:
        return []
    files = list(iterations[-1].glob("rank*_trace.json.gz")) or list(iterations[-1].glob("rank*_trace.json"))
    return sorted(files, key=lambda p: int(re.search(r"rank(\d+)_", p.name).group(1)))


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def cell_rows(cell: Path, ranks: list[dict], nproc: int, extra: dict):
    def pct(r, key):
        return 100 * r["acc"][key] / r["window"] if r["window"] else 0.0

    def mean(values):
        values = list(values)
        return sum(values) / len(values) if values else None

    n = len(ranks)
    mean_window = mean(r["window"] for r in ranks) or 1.0
    comm_total = sum(r["acc"]["comm"] for r in ranks)
    nccl_total = sum(g[2] for r in ranks for g in r["groups"])
    merged: dict[tuple, list] = defaultdict(lambda: [0, 0.0, 0, 0.0])
    for r in ranks:
        for key, kernels, kernel_us, nbytes, exposed_us in r["groups"]:
            m = merged[key]
            m[0] += kernels
            m[1] += kernel_us
            m[2] += nbytes
            m[3] += exposed_us
    by_label: dict[str, float] = defaultdict(float)
    for key, m in merged.items():
        by_label[key[0]] += m[3] / n
    top = sorted(by_label.items(), key=lambda kv: -kv[1])[:3]
    row = dict(
        cell=cell.name,
        ranks=n,
        step_ms=max(r["window"] for r in ranks) / 1e3,
        compute_pct=mean(pct(r, "compute") for r in ranks),
        exposed_pct=mean(pct(r, "exposed") for r in ranks),
        exposed_max_pct=max(pct(r, "exposed") for r in ranks),
        idle_pct=mean(pct(r, "idle") for r in ranks),
        idle_max_pct=max(pct(r, "idle") for r in ranks),
        copy_pct=mean(pct(r, "copy") for r in ranks),
        overlap_pct=100 * (1 - sum(r["acc"]["exposed"] for r in ranks) / comm_total) if comm_total else None,
        unattributed_pct=100 * sum(r["unattributed"] for r in ranks) / nccl_total if nccl_total else None,
        top_exposed=", ".join(f"{label} {100 * us / mean_window:.1f}%" for label, us in top if us > 0) or "-",
    )
    row.update(extra)
    pg_rows = []
    for key in sorted(merged, key=lambda k: -merged[k][3]):
        label, desc, collective, size = key
        kernels, kernel_us, nbytes, exposed_us = merged[key]
        busbw = nbytes / (kernel_us * 1e-6) * bus_factor(collective, size) / 1e9 if nbytes and kernel_us else None
        pg_rows.append(dict(
            cell=cell.name, label=label, group=desc, collective=collective, group_size=size,
            kernels=round(kernels / n, 1), kernel_ms=kernel_us / n / 1e3, exposed_ms=exposed_us / n / 1e3,
            exposed_pct=100 * exposed_us / n / mean_window, mbytes=nbytes / n / 1e6, eff_busbw=busbw,
        ))
    rank_rows = [
        dict(
            cell=cell.name, rank=r["rank"], node=r["rank"] // nproc, window_ms=r["window"] / 1e3,
            compute_ms=r["acc"]["compute"] / 1e3, exposed_ms=r["acc"]["exposed"] / 1e3,
            copy_ms=r["acc"]["copy"] / 1e3, idle_ms=r["acc"]["idle"] / 1e3, comm_ms=r["acc"]["comm"] / 1e3,
            unattributed_ms=r["unattributed"] / 1e3,
        )
        for r in ranks
    ]
    return row, pg_rows, rank_rows


def monitor_stats(stage_dir: Path) -> dict:
    try:
        from monitor import task_stats

        return task_stats(stage_dir.parent / "monitor")
    except Exception as exc:  # the report still stands without the monitor columns
        print(f"no monitor columns ({exc})", file=sys.stderr)
        return {}


def extras(stage_dir: Path, cell: Path, nproc: int, stats: dict) -> dict:
    out = {}
    try:
        from summarize import parse_cell

        s = parse_cell(cell, nproc, 3)
        out.update({k: s.get(k) for k in ("tps", "mfu", "peak_mem_gib", "peak_rank")})
    except Exception as exc:  # the report still stands without the log columns
        print(f"{cell.name}: no summarize columns ({exc})", file=sys.stderr)
    task = f"{cell.name.split('_')[0]}.{stage_dir.name}.{cell.name}"
    nodes = sorted(node for node, t in stats if t == task)
    for key, col in (("ib_tx_mean", "ib_tx_GBps"), ("ib_rx_mean", "ib_rx_GBps")):
        vals = [stats[(node, task)][key] for node in nodes]
        out[col] = ",".join(fmt(v) for v in vals) if vals else None
    return out


def write_tsv(path: Path, columns: list[str], rows: list[dict]) -> None:
    lines = ["\t".join(columns)] + ["\t".join(fmt(r.get(c)) for c in columns) for r in rows]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage_dir", type=Path)
    parser.add_argument("--nproc", type=int, default=int(os.environ.get("NPROC", "8")))
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--cells", default="", help="comma-separated row names (default: every cell with traces)")
    args = parser.parse_args()
    only = set(filter(None, args.cells.split(",")))
    cells = [
        c for c in sorted(p for p in args.stage_dir.iterdir() if p.is_dir() and not p.name.startswith("_"))
        if (not only or c.name in only) and trace_files(c)
    ]
    if not cells:
        print(f"no traces under {args.stage_dir}/*/tt/profiling/traces")
        return 1
    stats = monitor_stats(args.stage_dir)
    cell_out, pg_out, rank_out = [], [], []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for cell in cells:
            files = trace_files(cell)
            ranks = list(pool.map(analyze_rank, [str(f) for f in files]))
            bad = [r["rank"] for r in ranks if r.get("unreadable")]
            ranks = [r for r in ranks if not r.get("unreadable")]
            if bad:
                print(f"{cell.name}: unreadable (truncated?) traces of ranks {bad}", file=sys.stderr)
            if not ranks:
                continue
            row, pg_rows, rank_rows = cell_rows(cell, ranks, args.nproc, extras(args.stage_dir, cell, args.nproc, stats))
            cell_out.append(row)
            pg_out += pg_rows
            rank_out += rank_rows
            print(f"{cell.name}: {len(files)} ranks", file=sys.stderr)
    write_tsv(args.stage_dir / "trace_report.tsv", CELL_COLUMNS, cell_out)
    write_tsv(args.stage_dir / "trace_report_pg.tsv", PG_COLUMNS, pg_out)
    write_tsv(args.stage_dir / "trace_report_rank.tsv", RANK_COLUMNS, rank_out)
    print((args.stage_dir / "trace_report.tsv").read_text(), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
