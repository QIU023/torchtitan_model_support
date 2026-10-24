"""Run torchtitan.train and print per-module-group grad norms at every optimizer step (1 GPU)."""
import re, runpy, sys
from collections import defaultdict
import torch
import torchtitan.components.optim.optim as optim_mod
from torchtitan.models.minimax_m3.model import MiniMaxM3Model

models = []
orig_init = MiniMaxM3Model.__init__
def init(self, *a, **k):
    orig_init(self, *a, **k); models.append(self)
MiniMaxM3Model.__init__ = init

def group_of(name):
    name = re.sub(r"\.\d+\.", ".N.", name, count=1)
    for key in ("vision_encoder", "tok_embeddings", "lm_head", ".indexer.", "router", "routed_experts", "shared_experts", "attention", "feed_forward"):
        if key in name:
            return key.strip(".")
    return "norms/other"

step = [0]
orig_clip = optim_mod.dist_utils.clip_grad_norm_
def clip(parameters, *a, **k):
    step[0] += 1
    sq = defaultdict(float)
    for n, p in models[0].named_parameters():
        if p.grad is not None:
            g = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
            sq[group_of(n)] += g.float().pow(2).sum().item()
    print(f"GN step {step[0]}: " + " ".join(f"{k}={v ** 0.5:.2f}" for k, v in sorted(sq.items(), key=lambda kv: -kv[1])), flush=True)
    return orig_clip(parameters, *a, **k)
optim_mod.dist_utils.clip_grad_norm_ = clip
sys.argv = ["torchtitan.train"] + sys.argv[1:]
runpy.run_module("torchtitan.train", run_name="__main__")
