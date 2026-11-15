"""Replay one cell's collectives in isolation: torchrun ... comm_replay.py --plan <trace_report_replay.json> --cell <c>.

The plan (trace_report.py) lists the cell's costliest collectives with their member ranks, message sizes, dtype and two
in-training times: train_us, the median kernel time, and nowait_us, the median over calls of the shortest any member
took (the last rank to arrive does not wait). Every rank creates the same groups, then each collective runs --iters
times after --warmup, all instances of a group at once as in training (every EP group, every TP group), and PP's
send runs on every pair of adjacent stages at once. The isolated time is the slowest member's median.

Rank 0 writes the comparison (--out): isolated close to nowait means the in-training transfer ran at what the network
gives; train far above isolated means waiting for other ranks; nowait above isolated means interference in training
(other traffic or compute on the same links or SMs). all_to_all replays even splits of the same total.
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

DTYPES = {"BFloat16": torch.bfloat16, "Float": torch.float32, "Half": torch.float16, "Long": torch.int64,
          "Int": torch.int32}
COLUMNS = ["label", "collective", "msg_mb", "dtype", "instances", "calls_per_rank", "train_us", "nowait_us",
           "isolated_us", "train_over_isolated", "nowait_over_isolated", "isolated_busbw"]


def bus_factor(collective: str, n: int) -> float:
    c = collective.lower()
    if n < 2:
        return 1.0
    if "all_reduce" in c:
        return 2 * (n - 1) / n
    if any(k in c for k in ("gather", "reducescatter", "reduce_scatter", "all_to_all")):
        return (n - 1) / n
    return 1.0


def run_op(op: dict, mine, group, rank: int, device, dtype) -> None:
    c = op["collective"]
    if not mine:
        return
    if c == "send":
        # Every pair of adjacent stages at once; a middle stage sends to the next and receives from the previous.
        p2p = []
        for a, b in mine:
            t = torch.empty(op["in_nelems"], device=device, dtype=dtype)
            p2p.append(dist.P2POp(dist.isend, t, b) if rank == a else dist.P2POp(dist.irecv, t, a))
        for req in dist.batch_isend_irecv(p2p):
            req.wait()
        return
    n = len(mine)
    x = torch.empty(op["in_nelems"], device=device, dtype=dtype)
    if c == "all_to_all":
        dist.all_to_all_single(torch.empty_like(x), x, group=group)
    elif "gather" in c.lower():
        dist.all_gather_into_tensor(torch.empty(op["in_nelems"] * n, device=device, dtype=dtype), x, group=group)
    elif "reducescatter" in c.lower() or "reduce_scatter" in c.lower():
        dist.reduce_scatter_tensor(torch.empty(op["in_nelems"] // n, device=device, dtype=dtype), x, group=group)
    elif "all_reduce" in c:
        dist.all_reduce(x, group=group)
    else:
        raise ValueError(f"no replay for collective {c!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--out", default=None)
    parser.add_argument("--backend", default="nccl")
    args = parser.parse_args()

    dist.init_process_group(args.backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    if args.backend == "nccl":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    plan = json.load(open(args.plan))[args.cell]
    if plan["world"] != world:
        raise SystemExit(f"plan is for {plan['world']} ranks, this run has {world}")
    ops = [op for op in plan["ops"] if op["collective"] in ("send", "all_to_all", "all_reduce")
           or "gather" in op["collective"].lower() or "scatter" in op["collective"].lower()][: args.top]

    groups: dict[tuple, object] = {}
    for op in ops:
        if op["collective"] == "send":
            continue
        for inst in op["instances"]:
            key = tuple(inst)
            if key not in groups:
                groups[key] = dist.new_group(list(key))

    results = []
    for op in ops:
        if op["collective"] == "send":
            mine = [inst for inst in op["instances"] if rank in inst]
            group = None
        else:
            mine = next((inst for inst in op["instances"] if rank in inst), None)
            group = groups.get(tuple(mine)) if mine else None
        dtype = DTYPES.get(op["dtype"], torch.bfloat16)
        if args.backend != "nccl" and dtype in (torch.bfloat16, torch.float16):
            dtype = torch.float32
        times = []
        dist.barrier()
        for i in range(args.warmup + args.iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            run_op(op, mine, group, rank, device, dtype)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= args.warmup and mine:
                times.append((time.perf_counter() - start) * 1e6)
        local = statistics.median(times) if times else None
        gathered = [None] * world
        dist.all_gather_object(gathered, local)
        isolated = max((t for t in gathered if t is not None), default=None)
        size = max(op["in_nelems"], op["out_nelems"]) * torch.tensor([], dtype=DTYPES.get(op["dtype"], torch.bfloat16)).element_size()
        n = len(op["instances"][0])
        results.append(dict(
            label=op["label"], collective=op["collective"], msg_mb=size / 1e6, dtype=op["dtype"],
            instances=len(op["instances"]), calls_per_rank=op["calls_per_rank"],
            train_us=op["train_us"], nowait_us=op["nowait_us"], isolated_us=isolated,
            train_over_isolated=op["train_us"] / isolated if isolated else None,
            nowait_over_isolated=op["nowait_us"] / isolated if isolated and op["nowait_us"] else None,
            isolated_busbw=size / (isolated * 1e-6) * bus_factor(op["collective"], n) / 1e9 if isolated else None,
        ))

    if rank == 0:
        def fmt(v):
            return "-" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)

        text = "\n".join(["\t".join(COLUMNS)] + ["\t".join(fmt(r[c]) for c in COLUMNS) for r in results]) + "\n"
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            open(args.out, "w").write(text)
        print(text, end="")
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
