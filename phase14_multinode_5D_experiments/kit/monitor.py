"""Per-node GPU and InfiniBand sampler for the phase 14 runs: python monitor.py [--show | --summary] [--out DIR].

Without a flag (bootstrap.sh starts it once per container) it appends, every MN_MON_INTERVAL (2) seconds, this node's
GPU utilization, memory, power, SM clock, temperature and clock-event reasons (nvidia-smi) to
<out>/<hw>/monitor/gpu_node<k>.csv, and the transmit and receive rate of every active InfiniBand or RoCE port
(/sys/class/infiniband counters) to ib_node<k>.csv. Each row carries the node's pipeline state, so the samples of one
task can be picked out. --show prints the latest sample of every node; --summary writes monitor_summary.tsv next to
the samples, one row per node and task (its last attempt), and prints it.
"""

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

GPU_FIELDS = ["index", "utilization.gpu", "memory.used", "memory.total", "power.draw", "clocks.sm", "temperature.gpu"]
REASON_FIELDS = ("clocks_event_reasons.active", "clocks_throttle_reasons.active")
GPU_HEADER = ["ts", "state", "gpu", "util_pct", "mem_used_mib", "mem_total_mib", "power_w", "sm_mhz", "temp_c", "reasons"]
IB_HEADER = ["ts", "state", "hca", "port", "link_layer", "tx_GBps", "rx_GBps"]
# Clock-event bits that slow the GPU down: software power cap, hardware slowdown, software and hardware thermal
# slowdown, hardware power brake (GPU idle and the application clock setting are left out).
THROTTLE_BITS = 0x4 | 0x8 | 0x20 | 0x40 | 0x80
SUMMARY_COLUMNS = [
    "node", "task", "attempt", "seconds", "gpu_util_mean", "mem_max_gib", "power_mean_w", "sm_mhz_min",
    "throttle_pct", "ib_tx_mean", "ib_tx_max", "ib_rx_mean", "ib_rx_max", "eth_tx_mean", "eth_rx_mean",
]


def default_out() -> Path:
    for var in ("OUT", "MN_OUT"):
        if os.environ.get(var):
            return Path(os.environ[var])
    return Path(os.environ.get("MN_BASE", str(Path.home() / "p14"))) / "runs"


def number(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- sampling


def reason_field():
    for field in REASON_FIELDS:
        try:
            probe = subprocess.run(
                ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"], capture_output=True, text=True
            )
        except OSError:
            return None
        if probe.returncode == 0 and "not a valid field" not in (probe.stdout + probe.stderr).lower():
            return field
    return None


def gpu_rows(fields: list[str]) -> list[list[str]]:
    res = subprocess.run(
        ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    rows = []
    for line in res.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == len(fields):
            rows.append(parts + [""] * (len(GPU_FIELDS) + 1 - len(parts)))
    return rows


def ib_ports() -> list[tuple[str, str, str, Path]]:
    ports = []
    for port in sorted(Path("/sys/class/infiniband").glob("*/ports/*")):
        try:
            state = (port / "state").read_text().strip()
            layer = (port / "link_layer").read_text().strip()
        except OSError:
            continue
        if "ACTIVE" in state:
            ports.append((port.parent.parent.name, port.name, layer, port / "counters"))
    return ports


def ib_bytes(counters: Path):
    # port_xmit_data / port_rcv_data count 4-byte words.
    try:
        return int((counters / "port_xmit_data").read_text()) * 4, int((counters / "port_rcv_data").read_text()) * 4
    except (OSError, ValueError):
        return None


def appender(path: Path, header: list[str]):
    new = not path.exists() or path.stat().st_size == 0
    f = open(path, "a", newline="")
    writer = csv.writer(f)
    if new:
        writer.writerow(header)
        f.flush()
    return f, writer


def read_state(hb: Path) -> str:
    try:
        return json.loads(hb.read_text())["state"]
    except (OSError, ValueError, KeyError):
        return "-"


def sample(out: Path, hw: str, rank: int, interval: float) -> None:
    d = out / hw / "monitor"
    d.mkdir(parents=True, exist_ok=True)
    reason = reason_field()
    fields = GPU_FIELDS + ([reason] if reason else [])
    gpu_file, gpu_writer = appender(d / f"gpu_node{rank}.csv", GPU_HEADER)
    ib_file, ib_writer = appender(d / f"ib_node{rank}.csv", IB_HEADER)
    hb = out / hw / "pipeline" / "hb" / f"node{rank}"
    ports = ib_ports()
    last = {(hca, port): ib_bytes(c) for hca, port, _, c in ports}
    last_t = time.time()
    print(f"monitor: node {rank}, every {interval} s, {len(ports)} active IB/RoCE ports, reasons field {reason}", flush=True)
    tick = 0
    while True:
        time.sleep(interval)
        now = time.time()
        state = read_state(hb)
        ts = f"{now:.1f}"
        try:
            for row in gpu_rows(fields):
                gpu_writer.writerow([ts, state] + row)
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"monitor: nvidia-smi failed: {exc}", flush=True)
        dt = now - last_t
        for hca, port, layer, counters in ports:
            cur, prev = ib_bytes(counters), last.get((hca, port))
            if cur and prev and dt > 0 and cur[0] >= prev[0] and cur[1] >= prev[1]:
                tx, rx = (cur[0] - prev[0]) / dt / 1e9, (cur[1] - prev[1]) / dt / 1e9
                ib_writer.writerow([ts, state, hca, port, layer, f"{tx:.3f}", f"{rx:.3f}"])
            last[(hca, port)] = cur
        last_t = now
        gpu_file.flush()
        ib_file.flush()
        tick += 1
        if tick % 150 == 0:
            ports = ib_ports()
            for hca, port, _, counters in ports:
                last.setdefault((hca, port), ib_bytes(counters))


# ---------------------------------------------------------------- reading


def tail_rows(path: Path, size: int = 1 << 16) -> list[list[str]]:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            f.seek(max(0, end - size))
            text = f.read().decode(errors="replace")
    except OSError:
        return []
    lines = text.splitlines()
    if end > size:
        lines = lines[1:]
    return [r for r in csv.reader(io.StringIO("\n".join(lines))) if r and r[0] != "ts"]


def latest(rows: list[list[str]]) -> list[list[str]]:
    return [r for r in rows if r[0] == rows[-1][0]] if rows else []


def show(out: Path) -> None:
    now = time.time()
    for d in sorted(out.glob("*/monitor")):
        for gpu_csv in sorted(d.glob("gpu_node*.csv")):
            node = gpu_csv.stem[len("gpu_node"):]
            rows = latest(tail_rows(gpu_csv))
            if not rows:
                continue
            age = now - float(rows[0][0])
            print(f"== {d.parent.name} node {node}: {age:.0f} s ago, {rows[0][1]}")
            for r in rows:
                mem, total = number(r[4]), number(r[5])
                mem_s = f"{mem / 1024:6.1f} / {total / 1024:.0f} GiB" if mem is not None and total else r[4]
                print(f"  gpu {r[2]}: util {r[3]:>3}%  mem {mem_s}  {r[6]:>7} W  {r[7]:>5} MHz  {r[8]:>3} C  {r[9]}")
            ib = latest(tail_rows(d / f"ib_node{node}.csv"))
            for layer in sorted({r[4] for r in ib}):
                sel = [r for r in ib if r[4] == layer]
                tx = sum(number(r[5]) or 0 for r in sel)
                rx = sum(number(r[6]) or 0 for r in sel)
                print(f"  {layer}: tx {tx:.2f} GB/s, rx {rx:.2f} GB/s over {len(sel)} ports")


def split_state(state: str):
    """'run k3.measure.k3_5d 2' -> ('k3.measure.k3_5d', 2); other states -> None."""
    parts = state.split()
    if len(parts) == 3 and parts[0] == "run" and parts[2].isdigit():
        return parts[1], int(parts[2])
    return None


def task_stats(d: Path) -> dict[tuple[int, str], dict]:
    """Per (node, task) statistics of a monitor directory, from the task's last attempt."""
    stats: dict[tuple[int, str, int], dict] = {}
    for gpu_csv in d.glob("gpu_node*.csv"):
        node = int(gpu_csv.stem[len("gpu_node"):])
        with open(gpu_csv, newline="") as f:
            for r in csv.reader(f):
                if not r or r[0] == "ts" or len(r) < len(GPU_HEADER):
                    continue
                task = split_state(r[1])
                if task is None:
                    continue
                s = stats.setdefault((node, *task), defaultdict(list))
                ts = number(r[0])
                s["ts"].append(ts)
                for key, idx in (("util", 3), ("mem", 4), ("power", 6), ("sm", 7)):
                    v = number(r[idx])
                    if v is not None:
                        s[key].append(v)
                try:
                    s["throttle"].append(1 if int(r[9], 16) & THROTTLE_BITS else 0)
                except ValueError:
                    pass
    for ib_csv in d.glob("ib_node*.csv"):
        node = int(ib_csv.stem[len("ib_node"):])
        per_ts: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        with open(ib_csv, newline="") as f:
            for r in csv.reader(f):
                if not r or r[0] == "ts" or len(r) < len(IB_HEADER):
                    continue
                task = split_state(r[1])
                if task is None:
                    continue
                acc = per_ts[(task, r[0])]
                off = 0 if r[4] == "InfiniBand" else 2
                acc[off] += number(r[5]) or 0.0
                acc[off + 1] += number(r[6]) or 0.0
        for (task, _), acc in per_ts.items():
            s = stats.setdefault((node, *task), defaultdict(list))
            for key, v in zip(("ib_tx", "ib_rx", "eth_tx", "eth_rx"), acc):
                s[key].append(v)
    last_attempt: dict[tuple[int, str], int] = {}
    for node, task, attempt in stats:
        last_attempt[(node, task)] = max(attempt, last_attempt.get((node, task), 0))
    out = {}
    for (node, task), attempt in last_attempt.items():
        s = stats[(node, task, attempt)]

        def mean(key):
            return sum(s[key]) / len(s[key]) if s.get(key) else None

        ts = [t for t in s.get("ts", []) if t is not None]
        out[(node, task)] = dict(
            node=node,
            task=task,
            attempt=attempt,
            seconds=(max(ts) - min(ts)) if ts else None,
            gpu_util_mean=mean("util"),
            mem_max_gib=max(s["mem"]) / 1024 if s.get("mem") else None,
            power_mean_w=mean("power"),
            sm_mhz_min=min(s["sm"]) if s.get("sm") else None,
            throttle_pct=100 * mean("throttle") if s.get("throttle") else None,
            ib_tx_mean=mean("ib_tx"),
            ib_tx_max=max(s["ib_tx"]) if s.get("ib_tx") else None,
            ib_rx_mean=mean("ib_rx"),
            ib_rx_max=max(s["ib_rx"]) if s.get("ib_rx") else None,
            eth_tx_mean=mean("eth_tx"),
            eth_rx_mean=mean("eth_rx"),
        )
    return out


def fmt(v) -> str:
    if v is None:
        return "-"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def summary(out: Path) -> None:
    for d in sorted(out.glob("*/monitor")):
        stats = task_stats(d)
        lines = ["\t".join(SUMMARY_COLUMNS)]
        for key in sorted(stats, key=lambda k: (k[1], k[0])):
            lines.append("\t".join(fmt(stats[key][c]) for c in SUMMARY_COLUMNS))
        (d / "monitor_summary.tsv").write_text("\n".join(lines) + "\n")
        print(f"== {d}/monitor_summary.tsv")
        print("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or default_out()
    if args.show:
        show(out)
    elif args.summary:
        summary(out)
    else:
        sample(out, os.environ["HW"], int(os.environ.get("NODE_RANK", "0")), float(os.environ.get("MN_MON_INTERVAL", "2")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
