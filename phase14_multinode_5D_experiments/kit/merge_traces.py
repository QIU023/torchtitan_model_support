"""Merge one cell's per-rank profiler traces into one file for ui.perfetto.dev: python merge_traces.py <cell> [--out F].

Each rank becomes one process ("rank r, node n") and its CUDA streams become threads; without --cpu only the GPU
events (kernels, copies, GPU annotations) and the ProfilerStep markers are kept, which keeps 16 or 32 ranks loadable.
Times are made absolute with each trace's baseTimeNanoseconds, so ranks on different nodes line up as well as the
nodes' clocks agree; --align-step instead shifts each node so that its first ProfilerStep starts at the same time.
Default output: <cell>/merged_trace.json.gz. --ranks 0,8 keeps only those ranks.
"""

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trace_report import GPU_CATS, load_trace, trace_files  # noqa: E402

GPU_ANNOTATION = "gpu_user_annotation"


def keep(e: dict, cpu: bool) -> bool:
    if e.get("ph") not in ("X", "i", "I"):
        return False
    cat = e.get("cat")
    if cat in GPU_CATS or cat == GPU_ANNOTATION:
        return True
    if str(e.get("name", "")).startswith("ProfilerStep#"):
        return True
    return cpu


def base_us(trace: dict) -> float:
    base = trace.get("baseTimeNanoseconds")
    return float(base) / 1e3 if base else 0.0


def first_step(events: list, base: float):
    starts = [
        float(e["ts"]) + base
        for e in events
        if e.get("ph") == "X" and e.get("cat") != GPU_ANNOTATION and str(e.get("name", "")).startswith("ProfilerStep#")
    ]
    return min(starts) if starts else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cell", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cpu", action="store_true", help="keep the CPU events too")
    parser.add_argument("--ranks", default="", help="comma-separated ranks (default: all)")
    parser.add_argument("--align-step", action="store_true")
    parser.add_argument("--nproc", type=int, default=int(os.environ.get("NPROC", "8")))
    args = parser.parse_args()
    files = trace_files(args.cell)
    only = {int(r) for r in args.ranks.split(",") if r}
    files = [f for f in files if not only or int(f.name.split("_")[0][len("rank"):]) in only]
    if not files:
        print(f"no traces under {args.cell}/tt/profiling/traces")
        return 1

    # Pass 1: the earliest kept event and the first ProfilerStep of every rank.
    first, steps = {}, {}
    for f in files:
        rank = int(f.name.split("_")[0][len("rank"):])
        trace = load_trace(f)
        base = base_us(trace)
        events = trace.get("traceEvents", [])
        times = [float(e["ts"]) + base for e in events if keep(e, args.cpu) and "ts" in e]
        first[rank] = min(times) if times else None
        steps[rank] = first_step(events, base)
    node_step = {}
    for rank, t in steps.items():
        if t is not None:
            node = rank // args.nproc
            node_step[node] = min(t, node_step.get(node, t))
    shift = {rank: 0.0 for rank in first}
    if node_step:
        earliest = min(node_step.values())
        print(
            "first ProfilerStep per node: "
            + ", ".join(f"node {n} +{(t - earliest) / 1e3:.3f} ms" for n, t in sorted(node_step.items())),
            file=sys.stderr,
        )
        if args.align_step:
            for rank in shift:
                shift[rank] = node_step.get(rank // args.nproc, earliest) - earliest
    origin = min(t - shift[rank] for rank, t in first.items() if t is not None)

    # Pass 2: write the kept events, one rank at a time.
    out = args.out or args.cell / "merged_trace.json.gz"
    kept = 0
    with gzip.open(out, "wt", encoding="utf-8") as w:
        w.write('{"displayTimeUnit": "ms", "traceEvents": [\n')
        sep = ""
        for f in files:
            rank = int(f.name.split("_")[0][len("rank"):])
            trace = load_trace(f)
            base = base_us(trace)
            streams = set()
            for e in trace.get("traceEvents", []):
                if not keep(e, args.cpu) or "ts" not in e:
                    continue
                e = dict(e)
                e["pid"] = rank
                e["ts"] = float(e["ts"]) + base - origin - shift[rank]
                if e.get("cat") in GPU_CATS or e.get("cat") == GPU_ANNOTATION:
                    streams.add(e.get("tid"))
                w.write(sep + json.dumps(e))
                sep = ",\n"
                kept += 1
            meta = [
                {"ph": "M", "name": "process_name", "pid": rank,
                 "args": {"name": f"rank {rank}, node {rank // args.nproc}"}},
                {"ph": "M", "name": "process_sort_index", "pid": rank, "args": {"sort_index": rank}},
            ] + [
                {"ph": "M", "name": "thread_name", "pid": rank, "tid": tid, "args": {"name": f"stream {tid}"}}
                for tid in sorted(streams, key=str)
            ]
            for m in meta:
                w.write(sep + json.dumps(m))
                sep = ",\n"
        w.write("\n]}\n")
    print(f"{out}: {len(files)} ranks, {kept} events; open it in https://ui.perfetto.dev")
    return 0


if __name__ == "__main__":
    sys.exit(main())
