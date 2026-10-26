"""Per-layer input hidden-state gap between HF and TorchTitan (float32, flex, HF blocks/experts forced)."""
import sys
import torch
from huggingface_hub import snapshot_download
import scripts.checkpoint_conversion.numerical_tests_minimax_m3 as nt
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.minimax_m3 import build_model_config
from torchtitan.models.minimax_m3.state_dict_adapter import MiniMaxM3StateDictAdapter

force = sys.argv[1] == "force"
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(42)
dtype, dev = torch.float32, torch.device("cuda")
path = snapshot_download(repo_id=nt._HF_REPO_ID, revision=nt._HF_REVISION, allow_patterns=nt._HF_ALLOW_PATTERNS)
cfg = nt._use_released_vocab(build_model_config("debugmodel", attn_backend="flex", parallelism=ParallelismConfig()), path)
tt = nt._build_tt_model(cfg, dtype)
sd = MiniMaxM3StateDictAdapter(cfg, hf_assets_path=None).to_hf(tt.state_dict())

hf_in = {}
orig_build = nt._build_hf_model
def build_and_hook(*a, **k):
    m = orig_build(*a, **k)
    for i, layer in enumerate(m.model.language_model.layers):
        layer.register_forward_pre_hook(lambda _m, args, kwargs, i=i: hf_in.__setitem__(i, (args[0] if args else kwargs["hidden_states"])[0].detach().cpu()), with_kwargs=True)
        layer.self_attn.register_forward_hook(lambda _m, _i, out, i=i: hf_in.__setitem__(f"attn{i}", out[0][0].detach().cpu()))
        layer.mlp.register_forward_hook(lambda _m, _i, out, i=i: hf_in.__setitem__(f"ffn{i}", (out[0] if isinstance(out, tuple) else out).reshape(-1, out[0].shape[-1] if isinstance(out, tuple) else out.shape[-1]).detach().cpu()))
    return m
nt._build_hf_model = build_and_hook
ref = nt.run_hf(path, cfg, sd, 336, dtype, dev, "grouped_mm")
tt_in = {}
for i, layer in tt.layers.items():
    layer.register_forward_pre_hook(lambda _m, args, i=int(i): tt_in.__setitem__(i, args[0].detach().cpu()))
    layer.attention.register_forward_hook(lambda _m, _i, out, i=int(i): tt_in.__setitem__(f"attn{i}", out.detach().cpu()))
    layer.feed_forward.register_forward_hook(lambda _m, _i, out, i=int(i): tt_in.__setitem__(f"ffn{i}", out.detach().cpu()))
nt.run_tt(tt, ref, dtype, dev, {"image_id": 200025, "video_id": 200026}, force_hf_routing=force, force_hf_blocks=force)
for key in [0, "attn0", "ffn0", 1, "attn1", "ffn1", 2, "attn2", "ffn2", 3, 7]:
    a, b = hf_in[key].float(), tt_in[key].float()
    d = (a - b).abs()
    print(f"{str(key):>6}: max|diff| {d.max().item():.3e}  rel {d.max().item() / a.abs().max().item():.2e}  mean|diff| {d.mean().item():.2e}")

ids = ref["input_ids"][0]
img = (ids == 200025)
d = (hf_in["attn0"].float() - tt_in["attn0"].float()).abs().amax(-1)
print("image token span:", img.nonzero().min().item(), "-", img.nonzero().max().item(), "of", ids.numel())
top = d.topk(8).indices.sort().values
print("largest attn0 gaps at positions", top.tolist(), "image?", img[top].tolist())
print("attn0 gap: image tokens max", d[img].max().item(), " text tokens max", d[~img].max().item())
print("text tokens before image max", d[: img.nonzero().min().item()].max().item(), " after image max", d[img.nonzero().max().item() + 1 :].max().item())
