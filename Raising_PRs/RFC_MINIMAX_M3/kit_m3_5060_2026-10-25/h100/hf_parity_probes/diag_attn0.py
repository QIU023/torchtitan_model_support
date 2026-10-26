"""Layer-0 attention: HF eager fp32 vs TorchTitan flex fp32 vs a float64 reference on one input."""
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
import scripts.checkpoint_conversion.numerical_tests_minimax_m3 as nt
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.minimax_m3 import build_model_config
from torchtitan.models.minimax_m3.state_dict_adapter import MiniMaxM3StateDictAdapter
import transformers.models.minimax_m3_vl.modeling_minimax_m3_vl as M

torch.manual_seed(42)
dtype, dev = torch.float32, torch.device("cuda")
path = snapshot_download(repo_id=nt._HF_REPO_ID, revision=nt._HF_REVISION, allow_patterns=nt._HF_ALLOW_PATTERNS)
cfg = nt._use_released_vocab(build_model_config("debugmodel", attn_backend="flex", parallelism=ParallelismConfig()), path)
tt = nt._build_tt_model(cfg, dtype)
hf = nt._build_hf_model(path, cfg, MiniMaxM3StateDictAdapter(cfg, hf_assets_path=None).to_hf(tt.state_dict()), dtype, "grouped_mm").to(dev)
tt = tt.to(dev)
T = 333
x = torch.randn(1, T, cfg.dim, device=dev, dtype=dtype)
pos = torch.arange(T, device=dev)
attn_hf = hf.model.language_model.layers[0].self_attn
cos, sin = hf.model.language_model.rotary_emb(x, pos[None])
with torch.no_grad():
    causal = torch.ones(T, T, device=dev, dtype=torch.bool).tril()
    mask = torch.zeros(1, 1, T, T, device=dev, dtype=dtype).masked_fill(~causal, torch.finfo(dtype).min)
    out_hf = attn_hf(hidden_states=x, position_embeddings=(cos, sin), attention_mask=mask)[0][0]
    meta = tt._get_attention_metadata(positions=pos.to(torch.int32))
    layer = tt.layers["0"]
    out_tt = layer.attention(x[0], meta.get(type(layer.attention.inner_attention)), pos.to(torch.int32))
    # float64 reference built from HF's own projections.
    H, Hk, D = attn_hf.config.num_attention_heads, attn_hf.config.num_key_value_heads, attn_hf.head_dim
    xd = x.double()
    q = attn_hf.q_norm(F.linear(xd, attn_hf.q_proj.weight.double()).view(1, T, H, D).float()).double().transpose(1, 2)
    k = attn_hf.k_norm(F.linear(xd, attn_hf.k_proj.weight.double()).view(1, T, Hk, D).float()).double().transpose(1, 2)
    v = F.linear(xd, attn_hf.v_proj.weight.double()).view(1, T, Hk, D).transpose(1, 2)
    q, k = M.apply_rotary_pos_emb(q, k, cos.double(), sin.double())
    o = F.scaled_dot_product_attention(q, k.repeat_interleave(H // Hk, 1), v.repeat_interleave(H // Hk, 1), is_causal=True)
    out_ref = F.linear(o.transpose(1, 2).reshape(1, T, H * D), attn_hf.o_proj.weight.double())[0]
scale = out_ref.abs().max().item()
for name, out in (("HF eager", out_hf), ("TT flex", out_tt)):
    d = (out.double() - out_ref).abs()
    print(f"{name:9s} vs fp64: max|diff| {d.max().item():.3e}  rel {d.max().item() / scale:.2e}")
d = (out_hf.double() - out_tt.double()).abs()
print(f"HF vs TT : max|diff| {d.max().item():.3e}  rel {d.max().item() / scale:.2e}")
