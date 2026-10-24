"""Locate the sparse-block selection mismatches between HF and TorchTitan (float32)."""
import torch
from huggingface_hub import snapshot_download
import scripts.checkpoint_conversion.numerical_tests_minimax_m3 as nt
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.minimax_m3 import attention as A, build_model_config
from torchtitan.models.minimax_m3.state_dict_adapter import MiniMaxM3StateDictAdapter

torch.manual_seed(42)
dtype, dev = torch.float32, torch.device("cuda")
path = snapshot_download(repo_id=nt._HF_REPO_ID, revision=nt._HF_REVISION, allow_patterns=nt._HF_ALLOW_PATTERNS)
cfg = nt._use_released_vocab(build_model_config("debugmodel", attn_backend="flex", parallelism=ParallelismConfig()), path)
tt = nt._build_tt_model(cfg, dtype)
ref = nt.run_hf(path, cfg, MiniMaxM3StateDictAdapter(cfg, hf_assets_path=None).to_hf(tt.state_dict()), 336, dtype, dev, "grouped_mm")

captured = []  # (index_q_TGI, index_k_TI, titan selection)
orig = A.select_msa_blocks
def rec(index_q_TGI, index_k_TI, metadata, *, topk_blocks):
    out = orig(index_q_TGI, index_k_TI, metadata, topk_blocks=topk_blocks)
    captured.append((index_q_TGI.float().cpu(), index_k_TI.float().cpu(), out.cpu(), metadata))
    return out
A.select_msa_blocks = rec
special = {"image_id": 200025, "video_id": 200026}
nt.run_tt(tt, ref, dtype, dev, special, force_hf_routing=True, force_hf_blocks=False)

B, K = 16, 4
sparse_layers = sorted(ref["blocks"])
for (q, k, tsel, meta), layer in zip(captured, sparse_layers):
    T, G, _ = q.shape
    s = torch.einsum("tgi,ui->tgu", q, k)                                  # [T, G, T]
    causal = torch.arange(T)[None, :] <= torch.arange(T)[:, None]
    s = s.masked_fill(~causal[:, None, :], float("-inf"))
    nb = -(-T // B)
    s = torch.nn.functional.pad(s, (0, nb * B - T), value=float("-inf")).view(T, G, nb, B).amax(-1)
    own = torch.arange(T) // B
    s.scatter_(-1, own[:, None, None].expand(T, G, 1), float("inf"))
    top_s, top_i = s.topk(min(K, nb), dim=-1)
    rsel = top_i.masked_fill(top_s == float("-inf"), -1)
    hsel = ref["blocks"][layer]
    def sets(x): return x.sort(-1).values
    r, h, t = sets(rsel), sets(nt._pad_hf_blocks(hsel, K)), sets(tsel)
    if r.shape[-1] < K: r = sets(nt._pad_hf_blocks(r, K))
    hr = (h == r).all(-1); tr = (t == r).all(-1)
    print(f"layer {layer}: HF==ref {hr.float().mean():.3%}  TT==ref {tr.float().mean():.3%}  HF==TT {(h == t).all(-1).float().mean():.3%}")
    bad = (~hr).nonzero()
    margins = []
    for tok, g in bad.tolist():
        srt = s[tok, g].sort(descending=True).values
        finite = srt[torch.isfinite(srt)]
        margins.append((srt[K - 1] - srt[K]).item())
        hf_only = sorted(set(h[tok, g].tolist()) - set(r[tok, g].tolist())); ref_only = sorted(set(r[tok, g].tolist()) - set(h[tok, g].tolist()))
        if len(margins) <= 3:
            print(f"   hf!=ref t={tok} blk={tok // B} g={g} ref={r[tok,g].tolist()} hf={h[tok,g].tolist()} score[hf_only]={[round(s[tok,g,b].item(),4) for b in hf_only]} score[ref_only]={[round(s[tok,g,b].item(),4) for b in ref_only]} spread={finite.max().item() - finite.min().item() if finite.numel() else 0:.3f}")
    if margins:
        m = torch.tensor(margins)
        print(f"   {len(margins)} mismatches; kth-(k+1)th margin: median {m.median():.2e} max {m.max():.2e}; tokens {sorted(set(bad[:, 0].tolist()))[:12]}...")
