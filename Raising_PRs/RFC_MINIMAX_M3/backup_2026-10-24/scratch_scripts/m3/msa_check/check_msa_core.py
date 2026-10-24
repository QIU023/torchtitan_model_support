import math, torch, torch.nn.functional as F
from torchtitan.models.minimax_m3.attention import (
    build_msa_metadata, select_msa_blocks, msa_block_mask, MSAIndexerLoss, MSAInnerAttention)
from torchtitan.models.common.attention import FlexInnerAttention
from torch.nn.attention.flex_attention import AuxRequest

torch.manual_seed(0)
dev = "cuda"
B, TILE, TOPK = 16, 128, 4
doc_lens = [150, 70, 36]                     # second doc starts at 150 (not tile aligned)
T = sum(doc_lens)
positions = torch.cat([torch.arange(n) for n in doc_lens]).to(dev)
G, J, K, I = 2, 2, 16, 16
H = G * J
iq = torch.randn(T, G, I, device=dev); ik = torch.randn(T, I, device=dev)
md = build_msa_metadata(positions, block_size=B, tile_size=TILE)
sel = select_msa_blocks(iq, ik, md, topk_blocks=TOPK)

# reference selection
doc_start = (torch.arange(T, device=dev) - positions).tolist()
blk = md.block_T.tolist()
ref_sets, ok = {}, True
for i in range(T):
    for g in range(G):
        cand = {}
        for j in range(doc_start[i], i + 1):
            s = float(iq[i, g] @ ik[j])
            cand[blk[j]] = max(cand.get(blk[j], -math.inf), s)
        cand[blk[i]] = math.inf
        top = sorted(cand, key=lambda b: -cand[b])[:TOPK]
        got = {b for b in sel[i, g].tolist() if b >= 0}
        if set(top) != got:
            ok = False; print("selection mismatch", i, g, sorted(top), sorted(got)); break
    if not ok: break
print("selection matches reference:", ok, "| blocks", md.block_start_N.numel(), "| tiles", len(md.tile_key_start))

# attention vs explicit-mask SDPA
q = torch.randn(T, H, K, device=dev); k = torch.randn(T, G, K, device=dev); v = torch.randn(T, G, K, device=dev)
mask, occ = msa_block_mask(sel, md, num_heads=H)
out, aux = FlexInnerAttention.compiled_flex_attn(q.transpose(0,1)[None], k.transpose(0,1)[None], v.transpose(0,1)[None],
    score_mod=None, block_mask=mask, scale=K**-0.5, enable_gqa=True, return_aux=AuxRequest(lse=True), kernel_options={})
out = out[0].transpose(0, 1)
allowed = torch.zeros(H, T, T, dtype=torch.bool, device=dev)
bT = md.block_T
for h in range(H):
    g = h // J
    a = (sel[:, g, :, None] == bT[None, None, :]).any(1)          # [T, T]
    allowed[h] = a & (torch.arange(T, device=dev)[None, :] <= torch.arange(T, device=dev)[:, None])
kk = k.repeat_interleave(J, dim=1); vv = v.repeat_interleave(J, dim=1)
logits = torch.einsum("thk,uhk->htu", q, kk) * K**-0.5
logits = logits.masked_fill(~allowed, -math.inf)
ref = torch.einsum("htu,uhk->thk", logits.softmax(-1), vv)
ref_lse = torch.logsumexp(logits, -1).transpose(0, 1)
print("flex vs reference: max |out| err", (out - ref).abs().max().item(), "| max |lse| err", (aux.lse[0].transpose(0,1) - ref_lse).abs().max().item())

# KL loss vs reference, and gradient routing
loss = MSAIndexerLoss.Config(coeff=1.0, softmax_scale=K**-0.5, index_softmax_scale=I**-0.5).build().to(dev)
iq_g = iq.clone().requires_grad_(); ik_g = ik.clone().requires_grad_()
carrier = torch.zeros(T, H, K, device=dev, requires_grad=True)
out_c = loss(q, k, iq_g, ik_g, aux.lse[0].transpose(0, 1), md, block_indices_TGS=sel, occupied_XGX=occ,
             carrier=carrier, denominator=torch.tensor(1.0, device=dev))
out_c.sum().backward()
probs = logits.softmax(-1)                                         # [H, T, T]
P = probs.view(G, J, T, T).mean(1)                                 # [G, T, T]
iscores = torch.einsum("tgi,ui->gtu", iq, ik) * I**-0.5
ga = allowed.view(G, J, T, T)[:, 0]
logstu = iscores.masked_fill(~ga, -math.inf).log_softmax(-1).masked_fill(~ga, 0)
ref_kl = (torch.special.xlogy(P, P) - P * logstu).sum() / G
print("KL metric vs reference:", loss.instance_acc.item(), ref_kl.item())
print("grads: index_q", iq_g.grad.abs().sum().item() > 0, "index_k", ik_g.grad.abs().sum().item() > 0, "carrier grad is ones:", torch.equal(carrier.grad, torch.ones_like(carrier)))
