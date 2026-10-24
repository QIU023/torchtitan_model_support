"""Run torchtitan.train and save every parameter's full step-1 gradient on rank 0 (M3_GRAD_OUT), non-PP cells only."""
import os
import runpy
import sys

import torch
import torchtitan.components.optim.optim as optim_mod
from torchtitan.models.minimax_m3.model import MiniMaxM3Model

models = []
_orig_init = MiniMaxM3Model.__init__


def _init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    models.append(self)


MiniMaxM3Model.__init__ = _init
_orig_clip = optim_mod.dist_utils.clip_grad_norm_
_done = [False]


def _clip(parameters, *args, **kwargs):
    if not _done[0]:
        _done[0] = True
        grads = {}
        for name, param in models[0].named_parameters():
            grad = param.grad
            if grad is None:
                continue
            if hasattr(grad, "full_tensor"):
                grad = grad.full_tensor()
            grads[name] = grad.detach().float().cpu()
        if int(os.environ.get("RANK", "0")) == 0:
            torch.save(grads, os.environ["M3_GRAD_OUT"])
    return _orig_clip(parameters, *args, **kwargs)


optim_mod.dist_utils.clip_grad_norm_ = _clip
sys.argv = ["torchtitan.train"] + sys.argv[1:]
runpy.run_module("torchtitan.train", run_name="__main__")
