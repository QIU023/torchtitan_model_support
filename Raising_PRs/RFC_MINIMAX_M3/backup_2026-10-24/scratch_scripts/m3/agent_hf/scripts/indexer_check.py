"""CPU-only behavioural checks of HF MiniMax-M3 sparse attention (tiny random config, fp32)."""
import torch
import torch.nn.functional as F

from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLTextConfig
from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLForCausalLM

torch.manual_seed(0)
BS = 4  # block size
TOPK = 2


def make_cfg(sparse_flags, attn_impl):
    return MiniMaxM3VLTextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=32, dense_intermediate_size=48, shared_intermediate_size=24,
        num_hidden_layers=len(sparse_flags), num_attention_heads=8, num_key_value_heads=4, head_dim=16,
        num_local_experts=4, num_experts_per_tok=2, rope_theta=5e6, partial_rotary_factor=0.5,
        sparse_attention_config={"sparse_attention_freq": sparse_flags, "sparse_block_size": BS,
                                 "sparse_topk_blocks": TOPK, "sparse_local_block": 1,
                                 "sparse_num_index_heads": 4, "sparse_index_dim": 16},
        moe_layer_freq=[0] + [1] * (len(sparse_flags) - 1),
        attn_implementation=attn_impl, experts_implementation="eager",
    )


def ref_mask_from_indexer(indexer, hidden, cos, sin, pos):
    """Independent re-implementation of the selection semantics -> bool keep[B, Hq, S, S]."""
    B, S, _ = hidden.shape
    D = indexer.head_dim
    q = indexer.q_norm(indexer.q_proj(hidden).view(B, S, -1, D)).transpose(1, 2)
    k = indexer.k_norm(indexer.k_proj(hidden).view(B, S, 1, D)).transpose(1, 2)
    c = cos.unsqueeze(1); s = sin.unsqueeze(1); r = c.shape[-1]

    def rot(x):
        x1, x2 = x[..., : r // 2], x[..., r // 2: r]
        xr = torch.cat([-x2, x1], -1)
        return torch.cat([x[..., :r] * c + xr * s, x[..., r:]], -1)

    q, k = rot(q), rot(k)
    sc = (q.float() @ k.float().transpose(-1, -2))  # [B, Hidx, S, S]
    kpos = torch.arange(S)
    fut = kpos[None, None, None, :] > pos[:, None, :, None]
    sc = sc.masked_fill(fut, float("-inf"))
    nb = -(-S // BS)
    sc = F.pad(sc, (0, nb * BS - S), value=float("-inf")).view(B, sc.shape[1], S, nb, BS).amax(-1)
    keep = torch.zeros(B, sc.shape[1], S, S, dtype=torch.bool)
    for b in range(B):
        for h in range(sc.shape[1]):
            for i in range(S):
                own = int(pos[b, i]) // BS
                order = sorted(range(nb), key=lambda j: (j != own, -sc[b, h, i, j].item()))
                chosen = [j for j in order[:min(TOPK, nb)] if j == own or sc[b, h, i, j] > float("-inf")]
                for j in range(S):
                    keep[b, h, i, j] = (j // BS in chosen) and j <= pos[b, i]
    return keep.repeat_interleave(2, dim=1)  # 8 q heads / 4 idx heads


S = 24
ids = torch.randint(0, 128, (1, S))

for impl in ("eager", "sdpa"):
    torch.manual_seed(1)
    m = MiniMaxM3VLForCausalLM(make_cfg([0, 1], impl)).float().eval()
    print(f"[{impl}] experts impl:", m.config._experts_implementation, "| layer_types:", m.config.layer_types,
          "| mlp:", m.config.mlp_layer_types)
    captured = {}
    attn = m.model.layers[1].self_attn
    orig_bbm = attn.indexer.build_block_mask

    def spy(block_indices, attention_mask, key_length, dtype, device, position_ids):
        out = orig_bbm(block_indices, attention_mask, key_length, dtype, device, position_ids)
        captured.update(bi=block_indices, mask_in=attention_mask, mask_out=out, pos=position_ids)
        return out

    attn.indexer.build_block_mask = spy
    hooks = {}
    m.model.layers[1].self_attn.register_forward_pre_hook(
        lambda mod, args, kw: hooks.update(h=kw["hidden_states"], pe=kw["position_embeddings"]), with_kwargs=True)
    out = m(ids, use_cache=False)
    print(f"[{impl}] mask_in passed to build_block_mask:", None if captured["mask_in"] is None else
          (captured["mask_in"].dtype, tuple(captured["mask_in"].shape)))
    bi = captured["bi"]
    print(f"[{impl}] block_indices shape {tuple(bi.shape)}; row q=23 per head:", bi[0, :, 23].tolist(),
          "; row q=5:", bi[0, :, 5].tolist())
    keep_hf = captured["mask_out"] == 0
    cos, sin = hooks["pe"]
    keep_ref = ref_mask_from_indexer(attn.indexer, hooks["h"], cos, sin, captured["pos"])
    print(f"[{impl}] HF keep == reference keep:", torch.equal(keep_hf, keep_ref),
          "| mask dtype", captured["mask_out"].dtype, "| min value", captured["mask_out"].min().item())

    # Gradient reachability
    m.train()
    out = m(ids, labels=ids, use_cache=False)
    out.loss.backward()
    g = {n: (p.grad is None) for n, p in m.named_parameters() if "indexer" in n}
    print(f"[{impl}] indexer params with grad None:", g)
    m.zero_grad(set_to_none=True)
    m.eval()

# Dense equivalence: sequence where every query has <= TOPK visible blocks -> sparse == dense
torch.manual_seed(1)
ms = MiniMaxM3VLForCausalLM(make_cfg([0, 1], "eager")).float().eval()
torch.manual_seed(1)
md = MiniMaxM3VLForCausalLM(make_cfg([0, 0], "eager")).float().eval()
sd = {k: v for k, v in ms.state_dict().items() if "indexer" not in k}
print("load dense from sparse (missing/unexpected):", md.load_state_dict(sd, strict=False))
short = ids[:, : BS * TOPK]
long_ = ids
with torch.no_grad():
    print("short seq (<= topk*block) sparse==dense:",
          torch.allclose(ms(short, use_cache=False).logits, md(short, use_cache=False).logits, atol=1e-5))
    ls, ld = ms(long_, use_cache=False).logits, md(long_, use_cache=False).logits
    diff = (ls - ld).abs().amax(-1)[0]
    print("long seq per-position max|sparse-dense| (first 8 should be 0):", [round(x, 4) for x in diff.tolist()])

# Packed sequences: two docs of 12 tokens with position ids resetting
with torch.no_grad():
    pos = torch.cat([torch.arange(12), torch.arange(12)])[None]
    doc2 = ids[:, 12:]
    for name, mm in (("sparse", ms), ("dense", md)):
        packed = mm(ids, position_ids=pos, use_cache=False).logits[:, 12:]
        alone = mm(doc2, use_cache=False).logits
        print(f"packed {name}: doc2 logits match standalone doc2:", torch.allclose(packed, alone, atol=1e-5),
              "max abs diff", (packed - alone).abs().max().item())
    # same with default use_cache (DynamicCache is created -> packed detection disabled)
    packed = md(ids, position_ids=pos).logits[:, 12:]
    print("packed dense with default use_cache=True: doc2 match:", torch.allclose(packed, md(doc2).logits, atol=1e-5))

# Decode path: incremental generation equals full forward on the sparse model
with torch.no_grad():
    full = ms(ids, use_cache=False).logits
    o = ms(ids[:, :20], use_cache=True)
    pkv = o.past_key_values
    print("cache layer classes:", [type(l).__name__ for l in pkv.layers])
    steps = [o.logits[:, -1]]
    for t in range(20, 23):
        o = ms(ids[:, t: t + 1], past_key_values=pkv, use_cache=True)
        steps.append(o.logits[:, -1])
    inc = torch.stack(steps, 1)
    print("decode == full forward (positions 19..22):", torch.allclose(inc, full[:, 19:23], atol=1e-5))
