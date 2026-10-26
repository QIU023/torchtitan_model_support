"""Layer-1 MoE in isolation (float32): router, shared expert, routed experts, total."""
import torch
from huggingface_hub import snapshot_download
import scripts.checkpoint_conversion.numerical_tests_minimax_m3 as nt
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.minimax_m3 import build_model_config
from torchtitan.models.minimax_m3.state_dict_adapter import MiniMaxM3StateDictAdapter

torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(42)
dev = torch.device("cuda")
path = snapshot_download(repo_id=nt._HF_REPO_ID, revision=nt._HF_REVISION, allow_patterns=nt._HF_ALLOW_PATTERNS)
cfg = nt._use_released_vocab(build_model_config("debugmodel", attn_backend="flex", parallelism=ParallelismConfig()), path)
tt = nt._build_tt_model(cfg, torch.float32)
hf = nt._build_hf_model(path, cfg, MiniMaxM3StateDictAdapter(cfg, hf_assets_path=None).to_hf(tt.state_dict()), torch.float32, "eager").to(dev)
tt = tt.to(dev)
x = torch.randn(333, cfg.dim, device=dev)
hmoe, tmoe = hf.model.language_model.layers[1].mlp, tt.layers["1"].feed_forward
def rel(a, b): d = (a - b).abs(); return f"max|diff| {d.max().item():.3e} rel {d.max().item() / b.abs().max().item():.2e}"
with torch.no_grad():
    _, hw, hi = hmoe.gate(x)
    ts, ti, _ = tmoe.router(x, tmoe.expert_bias_E if hasattr(tmoe, "expert_bias_E") else None)
    same = (hi.sort(-1).values == ti.sort(-1).values).all().item()
    print("router ids identical:", same)
    order = lambda ids, w: w.gather(-1, ids.argsort(-1))
    print("router weights (x route_scale): ", rel(order(ti, ts.float()), order(hi, hw.float()) * hmoe.routed_scaling_factor))
    print("shared expert:", rel(tmoe.shared_experts(x), hmoe.shared_experts(x)))
    full_hf = hmoe(x[None])[0]
    full_tt = tmoe(x)
    print("full MoE:", rel(full_tt, full_hf))
    routed_hf = hmoe.experts(x, hi, hw) * hmoe.routed_scaling_factor
    routed_tt = full_tt - tmoe.shared_experts(x)
    print("routed part:", rel(routed_tt, routed_hf))
    # Per-expert check with titan's weights through HF's math.
    w13 = dict(tmoe.routed_experts.named_parameters())
    print("titan routed expert params:", {k: tuple(v.shape) for k, v in w13.items()})
    print("HF experts params:", {k: tuple(v.shape) for k, v in hmoe.experts.named_parameters()})
with torch.no_grad():
    ex = hmoe.experts
    gate_up, down = ex.gate_up_proj.data.clone(), ex.down_proj.data.clone()
    ex.gate_up_proj.data = gate_up.bfloat16().float(); ex.down_proj.data = down.bfloat16().float()
    routed_hf_bf16in = ex(x.bfloat16().float(), hi, hw) * hmoe.routed_scaling_factor
    print("routed vs HF on bf16-rounded inputs/weights:", rel(routed_tt, routed_hf_bf16in))
    ex.gate_up_proj.data, ex.down_proj.data = gate_up, down
with torch.no_grad():
    bf = lambda t: t.bfloat16().float()
    ex = hmoe.experts
    out = torch.zeros_like(x)
    xb = bf(x)
    for e in range(ex.num_experts):
        tok, slot = torch.where(hi == e)
        if tok.numel() == 0:
            continue
        gu = bf(xb[tok] @ bf(ex.gate_up_proj[e]).T)
        g, u = gu.chunk(2, dim=-1)
        g = g.clamp(max=ex.swiglu_limit); u = u.clamp(min=-ex.swiglu_limit, max=ex.swiglu_limit)
        act = bf((u + 1.0) * g * torch.sigmoid(g * ex.swiglu_alpha))
        y = bf(act @ bf(ex.down_proj[e]).T)
        out.index_add_(0, tok, y * (hw[tok, slot, None] * hmoe.routed_scaling_factor))
    print("routed vs HF math with titan's bf16 rounding points:", rel(routed_tt, out))
with torch.no_grad():
    re = tmoe.routed_experts
    w13, w2 = re.w13.weight.float(), re.w2.weight.float()          # [E, 2, F, D], [E, D, F]
    out32 = torch.zeros_like(x)
    for e in range(w13.shape[0]):
        tok, slot = torch.where(ti == e)
        if tok.numel() == 0:
            continue
        gu = torch.einsum("nd,kfd->nkf", x[tok], w13[e])            # [n, 2, F]
        act = re.activation_fn(gu)
        y = act @ w2[e].T
        out32.index_add_(0, tok, y * ts[tok, slot, None].float())
    print("titan params + titan activation in fp32 vs HF fp32 routed:", rel(out32, routed_hf))
    print("titan bf16 grouped_mm vs its own fp32 math:", rel(routed_tt, out32))
