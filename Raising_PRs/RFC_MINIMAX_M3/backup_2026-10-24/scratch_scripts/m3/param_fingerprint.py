"""Run torchtitan.train, then print per-rank fingerprints of TP-replicated indexer params (M3_FP_DIR)."""
import os, runpy, sys
import torch
import torchtitan.trainer as trainer_mod

orig_train = trainer_mod.Trainer.train
def train(self, *a, **k):
    out = orig_train(self, *a, **k)
    rank = int(os.environ.get("RANK", "0"))
    lines = []
    for part in self.engine.model_parts:
        for name, p in part.named_parameters():
            if any(key in name for key in ("indexer.wk", "indexer.k_norm", "attention.k_norm", "indexer.q_norm")):
                t = p.to_local() if hasattr(p, "to_local") else p
                t = t.detach().float()
                lines.append(f"{name} {t.numel()} {t.sum().item():.10e} {t.abs().sum().item():.10e}")
    with open(os.path.join(os.environ["M3_FP_DIR"], f"rank{rank}.txt"), "w") as f:
        f.write("\n".join(sorted(lines)))
    return out
trainer_mod.Trainer.train = train
sys.argv = ["torchtitan.train"] + sys.argv[1:]
runpy.run_module("torchtitan.train", run_name="__main__")
