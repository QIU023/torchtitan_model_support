"""Run torchtitan.train with torch.cuda.synchronize() probes at chosen points (SYNC_POINTS env)."""
import functools, os, runpy, sys
import torch

points = set(filter(None, os.environ.get("SYNC_POINTS", "").split(",")))

def probe(name):
    try:
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        print(f"PROBE {name}: FAILED {type(e).__name__}", flush=True)
        raise
    print(f"PROBE {name}: ok", flush=True) if os.environ.get("PROBE_VERBOSE") else None

def wrap(cls, attr, tag):
    orig = getattr(cls, attr)
    @functools.wraps(orig)
    def inner(self, *a, **k):
        if f"{tag}:pre" in points:
            probe(f"{tag}:pre")
        out = orig(self, *a, **k)
        if f"{tag}:post" in points:
            probe(f"{tag}:post")
        return out
    setattr(cls, attr, inner)

from torchtitan.models.minimax_m3 import attention as A, model as M, vision_encoder as V
import torchtitan.models.minimax_m3.attention as A
wrap(A.MSAInnerAttention, "forward", "msa")
wrap(A.MiniMaxSparseAttention, "forward", "sparse_attn")
wrap(A.MSAIndexer, "forward", "indexer")
wrap(V.MiniMaxM3VisionEncoder, "forward", "vision")
wrap(M.MiniMaxM3TransformerBlock, "forward", "block")
orig_sel, orig_mask = A.select_msa_blocks, A.msa_block_mask
def sel(*a, **k):
    out = orig_sel(*a, **k)
    if "select:post" in points: probe("select:post")
    return out
def msk(*a, **k):
    out = orig_mask(*a, **k)
    if "mask:post" in points: probe("mask:post")
    return out
A.select_msa_blocks, A.msa_block_mask = sel, msk
sys.argv = ["torchtitan.train"] + sys.argv[1:]
runpy.run_module("torchtitan.train", run_name="__main__")
