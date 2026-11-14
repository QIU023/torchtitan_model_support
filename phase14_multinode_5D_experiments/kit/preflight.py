"""Pre-flight check before the matrix (pipeline task preflight via preflight.sh, or by hand).

python preflight.py gemm --out <dir>   every GPU of this node multiplies bf16 matrices at once for --seconds (30);
    writes <dir>/gemm_node<k>.json: each GPU's TFLOPS and the SM clock, temperature and clock-event reasons
    nvidia-smi reported meanwhile.
python preflight.py verdict <dir> --comm <comm dir> --nodes N   on node 0: every node's GEMM file and the quick
    communication check (run_comm.sh with large messages only); prints the verdict, writes <dir>/VERDICT.txt and exits
    0 when everything passes, 3 when something fails.

A GPU fails when its TFLOPS fall under --min-frac (0.9) of the median over all GPUs, or when a thermal or hardware
slowdown shows up among its clock-event reasons (0x8, 0x20, 0x40, 0x80; the software power cap 0x4 is normal under a
full GEMM and only reported). The network fails when NCCL did not use InfiniBand with GPUDirect RDMA, when an IB port's
raw RDMA write bandwidth is under --min-rdma Gb/s (320, Together's health-check threshold), or when the all_reduce
busbw over all GPUs at the largest message is under --min-ar GB/s (300).
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from monitor import reason_field  # noqa: E402

BAD_REASONS = 0x8 | 0x20 | 0x40 | 0x80
POWER_CAP = 0x4


def gemm_worker(gpu: int, n: int, seconds: float, out: mp.Queue) -> None:
    import torch

    device = torch.device("cuda", gpu) if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    c = torch.empty(n, n, device=device, dtype=dtype)
    for _ in range(5):
        torch.matmul(a, b, out=c)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    iters, start = 0, time.perf_counter()
    while time.perf_counter() - start < seconds:
        for _ in range(10):
            torch.matmul(a, b, out=c)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        iters += 10
    elapsed = time.perf_counter() - start
    out.put((gpu, 2 * n**3 * iters / elapsed / 1e12))


def smi(fields: list[str]) -> list[list[str]]:
    try:
        text = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [[x.strip() for x in line.split(",")] for line in text.splitlines() if line.strip()]


def gemm(args: argparse.Namespace) -> int:
    rank = int(os.environ.get("NODE_RANK", "0"))
    try:
        import torch

        ngpu = torch.cuda.device_count() or 1
    except ImportError:
        ngpu = 1
    ngpu = args.gpus or ngpu
    field = reason_field()
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=gemm_worker, args=(g, args.n, args.seconds, queue)) for g in range(ngpu)]
    for p in procs:
        p.start()
    samples: dict[int, list] = {g: [] for g in range(ngpu)}
    while any(p.is_alive() for p in procs):
        for row in smi(["index", "clocks.sm", "temperature.gpu"] + ([field] if field else [])):
            try:
                g = int(row[0])
                reasons = int(row[3], 16) if field and len(row) > 3 and row[3].startswith("0x") else 0
                samples.setdefault(g, []).append((float(row[1]), float(row[2]), reasons))
            except (ValueError, IndexError):
                pass
        time.sleep(1)
    for p in procs:
        p.join()
    tflops = {}
    while not queue.empty():
        g, value = queue.get()
        tflops[g] = value
    gpus = []
    for g in range(ngpu):
        s = samples.get(g, [])
        reasons = 0
        for _, _, r in s:
            reasons |= r
        gpus.append(dict(
            gpu=g, tflops=tflops.get(g),
            sm_mhz_min=min((x[0] for x in s), default=None),
            sm_mhz_median=statistics.median([x[0] for x in s]) if s else None,
            temp_max=max((x[1] for x in s), default=None),
            reasons=hex(reasons),
        ))
    record = dict(node=rank, host=socket.gethostname(), n=args.n, seconds=args.seconds, gpus=gpus)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"gemm_node{rank}.json").write_text(json.dumps(record, indent=1))
    for x in gpus:
        print(f"node {rank} gpu {x['gpu']}: {x['tflops'] or 0:.0f} TFLOPS, SM {x['sm_mhz_min']}-{x['sm_mhz_median']} MHz, "
              f"max {x['temp_max']} C, reasons {x['reasons']}")
    return 0 if len(tflops) == ngpu else 1


def network_findings(comm: Path, min_rdma: float, min_ar: float) -> list[str]:
    bad = []
    verdict = comm / "verdict_node0.txt"
    text = verdict.read_text(errors="replace") if verdict.exists() else ""
    if not text:
        return [f"no communication verdict in {comm}"]
    if "InfiniBand: yes" not in text:
        bad.append("NCCL did not use InfiniBand")
    if "GPUDirect RDMA: yes" not in text:
        bad.append("no GPUDirect RDMA on the inter-node channels")
    for line in (comm / "rdma_node1.txt").read_text().splitlines() if (comm / "rdma_node1.txt").exists() else []:
        parts = line.split()
        if len(parts) >= 2 and (parts[1] == "failed" or float(parts[1]) < min_rdma):
            bad.append(f"RDMA {line.strip()} (under {min_rdma:.0f} Gb/s)")
    log = comm / "comm_node0.log"
    rows = []
    for line in log.read_text(errors="replace").splitlines() if log.exists() else []:
        m = re.match(r"\s*all_reduce\s+world\s+(\d+)\s+(\d+)\s+[\d.]+\s+([\d.]+)", line)
        if m:
            rows.append((int(m.group(2)), float(m.group(3))))
    if not rows:
        bad.append("no all_reduce busbw over all GPUs in the communication log")
    else:
        size, busbw = max(rows)
        if busbw < min_ar:
            bad.append(f"all_reduce busbw over all GPUs {busbw:.0f} GB/s at {size >> 20} MiB (under {min_ar:.0f})")
    return bad


def verdict(args: argparse.Namespace) -> int:
    out = Path(args.dir)
    records = []
    lines, bad = [], []
    for k in range(args.nodes):
        f = out / f"gemm_node{k}.json"
        if not f.exists():
            bad.append(f"node {k}: no GEMM result")
            continue
        records.append(json.loads(f.read_text()))
    values = [g["tflops"] for r in records for g in r["gpus"] if g["tflops"]]
    median = statistics.median(values) if values else 0.0
    lines.append(f"GEMM bf16 {records[0]['n'] if records else '?'}^3, median over {len(values)} GPUs: {median:.0f} TFLOPS")
    for r in records:
        for g in r["gpus"]:
            reasons = int(g["reasons"], 16)
            notes = []
            if not g["tflops"] or g["tflops"] < args.min_frac * median:
                notes.append(f"{g['tflops'] or 0:.0f} TFLOPS, under {args.min_frac:.0%} of the median")
            if reasons & BAD_REASONS:
                notes.append(f"clock-event reasons {g['reasons']} (thermal or hardware slowdown)")
            line = (f"node {r['node']} ({r['host']}) gpu {g['gpu']} (rank {r['node'] * len(r['gpus']) + g['gpu']}): "
                    f"{g['tflops'] or 0:.0f} TFLOPS, SM min {g['sm_mhz_min']} MHz, max {g['temp_max']} C, reasons {g['reasons']}"
                    + (" power cap" if reasons & POWER_CAP else ""))
            if notes:
                bad.append(line + " -> " + "; ".join(notes))
            lines.append(line)
    bad += network_findings(Path(args.comm), args.min_rdma, args.min_ar)
    head = "PREFLIGHT: PASS" if not bad else "PREFLIGHT: FAIL\n" + "\n".join("  " + b for b in bad)
    text = head + "\n" + "\n".join(lines) + "\n"
    (out / "VERDICT.txt").write_text(text)
    print(text, end="")
    return 0 if not bad else 3


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gemm")
    g.add_argument("--out", required=True)
    g.add_argument("--seconds", type=float, default=30)
    g.add_argument("--n", type=int, default=8192)
    g.add_argument("--gpus", type=int, default=0)
    v = sub.add_parser("verdict")
    v.add_argument("dir")
    v.add_argument("--comm", required=True)
    v.add_argument("--nodes", type=int, default=int(os.environ.get("NNODES", "2")))
    v.add_argument("--min-frac", type=float, default=0.9)
    v.add_argument("--min-rdma", type=float, default=320)
    v.add_argument("--min-ar", type=float, default=300)
    args = parser.parse_args()
    return gemm(args) if args.cmd == "gemm" else verdict(args)


if __name__ == "__main__":
    sys.exit(main())
