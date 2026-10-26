"""Split the layer-1 selection gap: input hidden states vs indexer arithmetic (float32, flex)."""
import torch
from huggingface_hub import snapshot_download
import scripts.checkpoint_conversion.numerical_tests_minimax_m3 as nt
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.minimax_m3 import build_model_config
from torchtitan.models.minimax_m3.state_dict_adapter import MiniMaxM3StateDictAdapter

torch.manual_seed(42)
dtype, dev = torch.float32, torch.device("cuda")
path = snapshot_download(repo_id=nt._HF_REPO_ID, revision=nt._HF_REVISION, allow_patterns=nt._HF_ALLOW_PATTERNS)
cfg = nt._use_released_vocab(build_model_config("debugmodel", attn_backend="flex", parallelism=ParallelismConfig()), path)
tt = nt._build_tt_model(cfg, dtype)
sd = MiniMaxM3StateDictAdapter(cfg, hf_assets_path=None).to_hf(tt.state_dict())
hf = nt._build_hf_model(path, cfg, sd, dtype, "grouped_mm").to(dev)
tt = tt.to(dev)

# One shared normalized input for both indexers: random rows, positions 0..T-1.
T = 333
x = torch.randn(1, T, cfg.dim, device=dev, dtype=dtype)
pos = torch.arange(T, device=dev)
hf_layer = hf.model.language_model.layers[1]
indexer = hf_layer.self_attn.indexer
rot = hf.model.language_model.rotary_emb(x, pos[None])
with torch.no_grad():
    q = indexer.q_norm(indexer.q_proj(x).view(1, T, -1, indexer.head_dim)).transpose(1, 2)
    k = indexer.k_norm(indexer.k_proj(x).view(1, T, 1, indexer.head_dim)).transpose(1, 2)
    import transformers.models.minimax_m3_vl.modeling_minimax_m3_vl as M
    cos, sin = rot
    hq, hk = M.apply_rotary_pos_emb(q, k, cos[..., : indexer.head_dim], sin[..., : indexer.head_dim])
    tt_attn = tt.layers["1"].attention
    tq, tk = tt_attn.indexer(x[0], pos, rope=tt_attn.rope)
print("index q: HF", tuple(hq.shape), "TT", tuple(tq.shape))
hq_TGI, hk_TI = hq[0].transpose(0, 1), hk[0, 0]
for name, a, b in (("index_q", hq_TGI, tq), ("index_k", hk_TI, tk)):
    d = (a.float() - b.float()).abs()
    print(f"{name}: max|diff| {d.max().item():.3e}  max|ref| {a.abs().max().item():.3e}  rel {d.max().item() / a.abs().max().item():.2e}")
cos_tt = tt_attn.rope.cache if hasattr(tt_attn.rope, "cache") else None
print("HF cos dtype", cos.dtype, "shape", tuple(cos.shape))
