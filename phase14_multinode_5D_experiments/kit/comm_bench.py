"""Bus bandwidth of the collectives multinode.md 5.1 cites, measured with torch.distributed on NCCL.

torchrun --nnodes 2 --nproc_per_node 8 --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:29501 comm_bench.py --out x.json

Groups: the whole world (16 ranks across the nodes) and each node's 8 ranks. Ops: all_reduce, all_gather,
reduce_scatter and all_to_all over a group, and sendrecv between rank r and its peer on the other node (r +- 8)
or on the same node (r xor 1). Bytes are the nccl-tests "size" (the full buffer); busbw uses nccl-tests' factors:
all_reduce 2(n-1)/n, all_gather / reduce_scatter / alltoall (n-1)/n, sendrecv 1. Rank 0 prints the table.
Sizes run from 256 B to 1 GiB, and rank 0 fits t = alpha + size / beta per op and group (least squares on the
relative error) and prints alpha with the busbw beta implies; through torch.distributed, alpha includes each call's
launch overhead. With --out, the JSON holds the measurements and the fits. Where MPI and ssh between the nodes exist,
nccl-tests (nccl_tests.sh) is the reference; the k8s pods have neither, so there this is the measurement.
"""

import argparse
import json
import os

import torch
import torch.distributed as dist


def _time(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters / 1e3


def _fit(points: list[tuple[int, float]]):
    """t = alpha + size / beta by least squares on the relative error; None unless the slope is positive."""
    w = [1 / (t * t) for _, t in points]
    sw = sum(w)
    sx = sum(wi * s for wi, (s, _) in zip(w, points))
    sxx = sum(wi * s * s for wi, (s, _) in zip(w, points))
    sy = sum(wi * t for wi, (_, t) in zip(w, points))
    sxy = sum(wi * s * t for wi, (s, t) in zip(w, points))
    det = sw * sxx - sx * sx
    if det <= 0:
        return None
    slope = (sw * sxy - sx * sy) / det
    if slope <= 0:
        return None
    return (sxx * sy - sx * sxy) / det, 1 / slope


def _fits(results: list[dict]) -> list[dict]:
    fits = []
    for key in sorted({(r["op"], r["group"], r["ranks"]) for r in results}):
        rows = [r for r in results if (r["op"], r["group"], r["ranks"]) == key]
        fit = _fit([(r["bytes"], r["time_s"]) for r in rows])
        if fit is None:
            continue
        alpha, beta = fit
        factor = rows[0]["busbw"] / rows[0]["algbw"]
        fits.append(dict(op=key[0], group=key[1], ranks=key[2], alpha_us=alpha * 1e6, busbw=beta / 1e9 * factor))
    return fits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-bytes", type=int, default=256)
    parser.add_argument("--max-bytes", type=int, default=1 << 30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    per_node = int(os.environ["LOCAL_WORLD_SIZE"])
    torch.cuda.set_device(local)
    node = rank // per_node
    node_groups = [
        dist.new_group(list(range(n * per_node, (n + 1) * per_node)))
        for n in range(world // per_node)
    ]
    groups = {"world": (None, world), "node": (node_groups[node], per_node)}

    results = []
    size = args.min_bytes
    while size <= args.max_bytes:
        numel = size // 2
        for gname, (group, n) in groups.items():
            if n < 2 or (gname == "node" and world == per_node):
                continue
            x = torch.empty(numel, dtype=torch.bfloat16, device="cuda")
            y = torch.empty(numel, dtype=torch.bfloat16, device="cuda")
            shard = torch.empty(numel // n, dtype=torch.bfloat16, device="cuda")
            ops = {
                "all_reduce": (lambda: dist.all_reduce(x, group=group), 2 * (n - 1) / n),
                "all_gather": (
                    lambda: dist.all_gather_into_tensor(y, shard, group=group),
                    (n - 1) / n,
                ),
                "reduce_scatter": (
                    lambda: dist.reduce_scatter_tensor(shard, x, group=group),
                    (n - 1) / n,
                ),
                "alltoall": (lambda: dist.all_to_all_single(y, x, group=group), (n - 1) / n),
            }
            for op, (fn, factor) in ops.items():
                t = _time(fn, args.warmup, args.iters)
                results.append(dict(op=op, group=gname, ranks=n, bytes=size, time_s=t,
                                    algbw=size / t / 1e9, busbw=size / t / 1e9 * factor))
        for where, peer in (("cross_node", (rank + per_node) % world), ("same_node", rank ^ 1)):
            if where == "cross_node" and world == per_node:
                continue
            x = torch.empty(numel, dtype=torch.bfloat16, device="cuda")
            y = torch.empty(numel, dtype=torch.bfloat16, device="cuda")

            def sendrecv(x=x, y=y, peer=peer):
                for req in dist.batch_isend_irecv(
                    [dist.P2POp(dist.isend, x, peer), dist.P2POp(dist.irecv, y, peer)]
                ):
                    req.wait()

            t = _time(sendrecv, args.warmup, args.iters)
            results.append(dict(op="sendrecv", group=where, ranks=2, bytes=size, time_s=t,
                                algbw=size / t / 1e9, busbw=size / t / 1e9))
        size *= 2

    if rank == 0:
        print(f"{'op':15s} {'group':11s} {'ranks':>5s} {'bytes':>12s} {'algbw GB/s':>11s} {'busbw GB/s':>11s}")
        for r in results:
            print(f"{r['op']:15s} {r['group']:11s} {r['ranks']:5d} {r['bytes']:12d} {r['algbw']:11.1f} {r['busbw']:11.1f}")
        fits = _fits(results)
        print(f"fit {'op':15s} {'group':11s} {'ranks':>5s} {'alpha us':>9s} {'busbw GB/s':>11s}")
        for fit in fits:
            print(f"fit {fit['op']:15s} {fit['group']:11s} {fit['ranks']:5d} {fit['alpha_us']:9.1f} {fit['busbw']:11.1f}")
        if args.out:
            with open(args.out, "w") as f:
                json.dump(dict(results=results, fits=fits), f, indent=1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
