"""Run torchtitan.train and record step-1 evidence in M3_PROBE_DIR: full-precision loss and grad norm per step and
rank, every parameter's full step-1 gradient (M3_DUMP_GRADS=1, non-PP cells), and rank 0's local output of every
decoder block on the first forward."""
import os
import runpy
import sys

import torch
import torchtitan.components.optim.optim as optim_mod
from torchtitan.models.minimax_m3.model import MiniMaxM3Model, MiniMaxM3TransformerBlock
from torchtitan.observability.metrics import MetricsProcessor

OUT = os.environ["M3_PROBE_DIR"]
RANK = int(os.environ.get("RANK", "0"))
models = []
_orig_init = MiniMaxM3Model.__init__


def _init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    models.append(self)


MiniMaxM3Model.__init__ = _init

import torchtitan.trainer as trainer_mod

_engine_parts: list = []
_orig_train = trainer_mod.Trainer.train


def _train(self, *args, **kwargs):
    _engine_parts.extend(self.engine.model_parts)
    return _orig_train(self, *args, **kwargs)


trainer_mod.Trainer.train = _train

_orig_log = MetricsProcessor.log


def _log(self, step, global_avg_loss, global_max_loss, grad_norm, *args, **kwargs):
    with open(os.path.join(OUT, f"loss_rank{RANK}.txt"), "a") as f:
        f.write(f"{step} {float(global_avg_loss)!r} {float(grad_norm)!r}\n")
    return _orig_log(self, step, global_avg_loss, global_max_loss, grad_norm, *args, **kwargs)


MetricsProcessor.log = _log

_orig_clip = optim_mod.dist_utils.clip_grad_norm_
_grads_done = [os.environ.get("M3_DUMP_GRADS") != "1"]
_reduce_inputs: dict = {}
_reduce_done = [os.environ.get("M3_DUMP_REDUCE_INPUTS") != "1"]
if not _reduce_done[0]:
    import torch.distributed.fsdp._fully_shard._fsdp_param_group as _fsdp_group

    _orig_foreach_reduce = _fsdp_group.foreach_reduce

    def _foreach_reduce(fsdp_params, unsharded_grads, *args, **kwargs):
        if not _reduce_done[0]:
            for fsdp_param, grad in zip(fsdp_params, unsharded_grads):
                _reduce_inputs.setdefault(fsdp_param._param_fqn, []).append(grad.detach().cpu().clone())
        return _orig_foreach_reduce(fsdp_params, unsharded_grads, *args, **kwargs)

    _fsdp_group.foreach_reduce = _foreach_reduce
    sys.stderr.write(f"m3_reduce_probe: rank {RANK} foreach_reduce wrapped\n")


def _clip(parameters, *args, **kwargs):
    if not _reduce_done[0]:
        _reduce_done[0] = True
        torch.save(_reduce_inputs, os.path.join(OUT, f"reduce_inputs_rank{RANK}.pt"))
        calls = sorted({len(v) for v in _reduce_inputs.values()})
        sys.stderr.write(f"m3_reduce_probe: rank {RANK} saved {len(_reduce_inputs)} params, calls per param {calls}\n")
    if not _grads_done[0]:
        _grads_done[0] = True
        grads = {}
        for name, param in (item for part in (_engine_parts or models) for item in part.named_parameters()):
            if param.grad is None or name in grads:
                continue
            grad = param.grad
            grads[name] = (grad.full_tensor() if hasattr(grad, "full_tensor") else grad).detach().float().cpu()
        if RANK == 0:
            torch.save(grads, os.path.join(OUT, "grads.pt"))
        if os.environ.get("M3_GRADS_PER_RANK") == "1":
            torch.save(grads, os.path.join(OUT, f"grads_rank{RANK}.pt"))
    if _bwd:
        torch.save(_bwd, os.path.join(OUT, f"bwd_rank{RANK}.pt"))
    if os.environ.get("M3_HOOK_SELECT_ALL") == "1" and RANK == 0:
        torch.save(globals()["_selections"], os.path.join(OUT, "selections_all.pt"))
    if os.environ.get("M3_HOOK_ROUTE_ALL") == "1" and RANK == 0:
        torch.save(_routes, os.path.join(OUT, "routes_all.pt"))
    return _orig_clip(parameters, *args, **kwargs)


optim_mod.dist_utils.clip_grad_norm_ = _clip

_blocks: dict[str, torch.Tensor] = {}
_orig_block_init = MiniMaxM3TransformerBlock.__init__


def _block_init(self, *args, **kwargs):
    _orig_block_init(self, *args, **kwargs)

    def hook(module, inputs, output):
        name = f"block{len(_blocks)}"
        if RANK == 0 and len(_blocks) < 64 and not getattr(module, "_m3_seen", False):
            module._m3_seen = True
            local = output.to_local() if hasattr(output, "to_local") else output
            _blocks[f"{getattr(module, '_m3_name', name)}"] = local.detach().float().cpu()
            torch.save(_blocks, os.path.join(OUT, "blocks.pt"))

    self.register_forward_hook(hook)


MiniMaxM3TransformerBlock.__init__ = _block_init
_orig_model_init2 = MiniMaxM3Model.__init__


def _name_blocks(self, *args, **kwargs):
    _orig_model_init2(self, *args, **kwargs)
    for name, layer in self.layers.items():
        layer._m3_name = f"layers.{name}"


MiniMaxM3Model.__init__ = _name_blocks

_parts: dict[str, object] = {}
_orig_model_init3 = MiniMaxM3Model.__init__


def _hook_parts(self, *args, **kwargs):
    _orig_model_init3(self, *args, **kwargs)
    layer_id = os.environ.get("M3_HOOK_LAYER")
    if layer_id is None or layer_id not in self.layers:
        return
    layer = self.layers[layer_id]

    def save(tag):
        def hook(module, inputs, output):
            if RANK != 0 or tag in _parts:
                return
            outs = output if isinstance(output, tuple) else (output,)
            _parts[tag] = [
                (o.to_local() if hasattr(o, "to_local") else o).detach().float().cpu()
                for o in outs
                if isinstance(o, torch.Tensor)
            ]
            torch.save(_parts, os.path.join(OUT, "parts.pt"))

        return hook

    layer.attention.register_forward_hook(save("attention"))
    layer.attention.inner_attention.register_forward_hook(save("inner_attention"))
    layer.attention.qkv_linear.register_forward_hook(save("qkv"))
    layer.feed_forward.register_forward_hook(save("moe"))
    for name in ("router", "shared_experts", "routed_experts"):
        sub = getattr(layer.feed_forward, name, None)
        if sub is not None:
            sub.register_forward_hook(save(name))


MiniMaxM3Model.__init__ = _hook_parts

if os.environ.get("M3_HOOK_EXPERTS") is not None:
    import torch_remat as remat
    from torchtitan.models.common.moe import moe as moe_mod

    _orig_routed = moe_mod.RoutedExperts.forward
    _target = int(os.environ["M3_HOOK_EXPERTS"])
    _calls = [0]

    def _routed_forward(self, x_TD, topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E, **kw):
        call = _calls[0]
        _calls[0] += 1
        if call != _target - 1:
            return _orig_routed(self, x_TD, topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E, **kw)
        routed_RD, counts_e, metadata = self.token_dispatcher.dispatch(
            x_TD, topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E
        )
        offsets_E = torch.cumsum(counts_e, dim=0, dtype=torch.int32)
        with moe_mod.maybe_set_sparse_mesh():
            gate_up = self.w13(routed_RD.bfloat16(), offsets_E)
            hidden = self.activation_fn(gate_up, offsets=offsets_E)
            out_RD = self.w2(hidden, offsets_E).type_as(routed_RD)
        out_TD = self.token_dispatcher.combine(out_RD, metadata, x_TD)
        loc = lambda t: (t.to_local() if hasattr(t, "to_local") else t).detach().float().cpu()
        torch.save({"routed_in": loc(routed_RD), "counts": loc(counts_e), "gate_up": loc(gate_up), "hidden": loc(hidden),
                    "out_RD": loc(out_RD), "out_TD": loc(out_TD), "ids": loc(topk_expert_ids_TK),
                    "local_counts": loc(num_local_tokens_per_expert_E)}, os.path.join(OUT, f"experts_rank{RANK}.pt"))
        return out_TD

    moe_mod.RoutedExperts.forward = _routed_forward
if os.environ.get("M3_HOOK_COMBINE") is not None:
    from torchtitan.models.common.moe import token_dispatcher as td_mod

    _orig_ssa = td_mod.LocalTokenDispatcher._score_and_scatter_add
    _ssa_target = int(os.environ["M3_HOOK_COMBINE"])
    _ssa_calls = [0]

    def _ssa(self, routed_output_ND, topk_scores_N, token_indices_N, x_TD):
        out = _orig_ssa(self, routed_output_ND, topk_scores_N, token_indices_N, x_TD)
        _ssa_calls[0] += 1
        if RANK == 0 and _ssa_calls[0] == _ssa_target:
            loc = lambda t: (t.to_local() if hasattr(t, "to_local") else t).detach().cpu()
            torch.save({"rows": loc(routed_output_ND).float(), "scores": loc(topk_scores_N).float(),
                        "tokens": loc(token_indices_N), "out": loc(out).float()}, os.path.join(OUT, "combine.pt"))
        return out

    td_mod.LocalTokenDispatcher._score_and_scatter_add = _ssa

_bwd: dict[str, list] = {}
if os.environ.get("M3_HOOK_BWD"):
    _bwd_layers = os.environ["M3_HOOK_BWD"].split(",")
    _orig_model_init4 = MiniMaxM3Model.__init__

    def _cpu(t):
        return (t.to_local() if hasattr(t, "to_local") else t).detach().float().cpu()

    def _grad_hook(tag):
        def hook(grad):
            _bwd.setdefault(tag, []).append(_cpu(grad))

        return hook

    def _module_hook(prefix, input_tags, output_tags):
        import torch_remat

        def hook(module, inputs, output):
            if torch_remat.is_recomputing():
                return
            outs = output if isinstance(output, tuple) else (output,)
            for tag, t in zip(output_tags, outs):
                if tag and isinstance(t, torch.Tensor) and t.requires_grad:
                    _bwd.setdefault(f"fwd:{prefix}.{tag}", []).append(_cpu(t))
                    t.register_hook(_grad_hook(f"grad:{prefix}.{tag}"))
            for tag, t in zip(input_tags, inputs):
                if tag and isinstance(t, torch.Tensor) and t.requires_grad:
                    t.register_hook(_grad_hook(f"grad:{prefix}.{tag}"))

        return hook

    def _hook_bwd(self, *args, **kwargs):
        _orig_model_init4(self, *args, **kwargs)
        for layer_id in _bwd_layers:
            if layer_id not in self.layers:
                continue
            layer = self.layers[layer_id]
            p = f"L{layer_id}"
            layer.register_forward_hook(_module_hook(p, ["block_in"], ["block_out"]))
            layer.attention.register_forward_hook(_module_hook(p, ["attn_in"], ["attn_out"]))
            layer.attention.qkv_linear.register_forward_hook(_module_hook(p, [], ["qkv_q", "qkv_k", "qkv_v"]))
            layer.attention.inner_attention.register_forward_hook(
                _module_hook(p, ["inner_q", "inner_k", "inner_v"], ["inner_out"])
            )
            layer.attention.wo.register_forward_hook(_module_hook(p, ["wo_in"], []))
            layer.feed_forward.register_forward_hook(_module_hook(p, ["ffn_in"], ["ffn_out"]))

    MiniMaxM3Model.__init__ = _hook_bwd

if os.environ.get("M3_HOOK_SELECT") == "1":
    from torchtitan.models.minimax_m3 import attention as msa_mod

    _orig_select = msa_mod.select_msa_blocks
    _selections: list = []
    _save_all = os.environ.get("M3_HOOK_SELECT_ALL") == "1"

    def _select(*args, **kwargs):
        out = _orig_select(*args, **kwargs)
        if RANK == 0 and (_save_all or len(_selections) < 7):
            _selections.append(out.detach().cpu())
            if not _save_all:
                torch.save(_selections, os.path.join(OUT, "selections.pt"))
        return out

    msa_mod.select_msa_blocks = _select

if os.environ.get("M3_FORCE_SELECT"):
    import torch_remat
    from torchtitan.models.minimax_m3 import attention as msa_mod

    _forced = torch.load(os.environ["M3_FORCE_SELECT"])
    _forced_calls = [0]
    _orig_select2 = msa_mod.select_msa_blocks

    def _forced_select(*args, **kwargs):
        out = _orig_select2(*args, **kwargs)
        if torch_remat.is_recomputing():
            raise RuntimeError("select_msa_blocks ran during recompute")
        ref = _forced[_forced_calls[0]]
        _forced_calls[0] += 1
        local_groups = out.shape[1]
        ref = ref[:, RANK * local_groups : (RANK + 1) * local_groups]
        assert ref.shape == out.shape, (ref.shape, out.shape)
        return ref.to(device=out.device, dtype=out.dtype)

    msa_mod.select_msa_blocks = _forced_select

if os.environ.get("M3_STATIC_SHAPES") == "1":
    import torch._dynamo

    torch._dynamo.config.automatic_dynamic_shapes = False
    if os.environ.get("M3_RECOMPILE_LIMIT"):
        torch._dynamo.config.recompile_limit = int(os.environ["M3_RECOMPILE_LIMIT"])
        torch._dynamo.config.accumulated_recompile_limit = 16 * int(os.environ["M3_RECOMPILE_LIMIT"])
        sys.stderr.write(f"m3_static_probe: rank {RANK} recompile_limit {torch._dynamo.config.recompile_limit}\n")

_routes: list = []
if os.environ.get("M3_HOOK_ROUTE_ALL") == "1" or os.environ.get("M3_FORCE_ROUTE"):
    import torch_remat
    from torchtitan.models.common.moe import moe as _moe_mod

    _orig_select_experts = _moe_mod.TokenChoiceTopKRouter._select_experts
    _forced_routes = torch.load(os.environ["M3_FORCE_ROUTE"]) if os.environ.get("M3_FORCE_ROUTE") else None
    _route_calls = [0]

    def _select_experts(self, scores_TE, *args, **kwargs):
        ids = _orig_select_experts(self, scores_TE, *args, **kwargs)
        if torch_remat.is_recomputing():
            raise RuntimeError("routing ran during recompute")
        if _forced_routes is None:
            if RANK == 0:
                _routes.append(ids.detach().cpu())
            return ids
        ref = _forced_routes[_route_calls[0]]
        _route_calls[0] += 1
        local_tokens = ids.shape[0]
        ref = ref[RANK * local_tokens : (RANK + 1) * local_tokens]
        assert ref.shape == ids.shape, (ref.shape, ids.shape)
        return ref.to(device=ids.device, dtype=ids.dtype)

    _moe_mod.TokenChoiceTopKRouter._select_experts = _select_experts

sys.argv = ["torchtitan.train"] + sys.argv[1:]
runpy.run_module("torchtitan.train", run_name="__main__")
