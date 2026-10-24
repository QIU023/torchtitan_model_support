"""Print local shapes inside MiniMaxSparseAttention.forward on rank 0, then run m3_text18_probe."""
import os, runpy, sys
import torch
import torchtitan.models.minimax_m3.attention as A
from torchtitan.models.common.attention import GQAttention

seen = [0]
orig_qkv = None
orig = A.MiniMaxSparseAttention.forward
def forward(self, x_TD, attention_metadata, positions=None, **kw):
    if seen[0] < 1 and int(os.environ.get("RANK", "0")) == 0:
        seen[0] += 1
        q, k, v = self.qkv_linear(x_TD)
        print("PROBE x_TD", tuple(x_TD.shape), type(x_TD).__name__, "q", tuple(q.shape), "positions", None if positions is None else tuple(positions.shape), flush=True)
    return orig(self, x_TD, attention_metadata, positions, **kw)
A.MiniMaxSparseAttention.forward = forward
sys.argv = ["torchtitan.train"] + sys.argv[1:]
runpy.run_module("torchtitan.train", run_name="__main__")
