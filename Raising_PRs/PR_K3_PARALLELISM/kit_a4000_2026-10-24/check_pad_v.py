import torch, torch.nn.functional as F
import torch.nn.attention.varlen as varlen
assert varlen.varlen_attn.__name__ == "_varlen_attn_padded_v", varlen.varlen_attn
torch.manual_seed(0)
H, Dq, Dv, lens = 4, 192, 128, [100, 156]
T = sum(lens); cu = torch.tensor([0, lens[0], T], device="cuda", dtype=torch.int32)
q = torch.randn(T, H, Dq, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(T, H, Dq, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(T, H, Dv, device="cuda", dtype=torch.bfloat16, requires_grad=True)
dout = torch.randn(T, H, Dv, device="cuda", dtype=torch.bfloat16)
out, lse = varlen.varlen_attn(q, k, v, cu, cu, max(lens), max(lens), window_size=(-1, 0), return_aux=varlen.AuxRequest(lse=True))
out.backward(dout)
g = [t.grad.float().clone() for t in (q, k, v)]
qf, kf, vf = (t.detach().float().requires_grad_() for t in (q, k, v))
ref, s = [], 0
for n in lens:
    sl = slice(s, s + n); s += n
    ref.append(F.scaled_dot_product_attention(qf[sl].transpose(0, 1), kf[sl].transpose(0, 1), vf[sl].transpose(0, 1), is_causal=True).transpose(0, 1))
ref = torch.cat(ref); ref.backward(dout.float())
print("out", tuple(out.shape), out.is_contiguous(), "lse", tuple(lse.shape))
print("max |out - ref|", (out.float() - ref).abs().max().item(), "ref max", ref.abs().max().item())
for name, a, b in zip("qkv", g, (qf.grad, kf.grad, vf.grad)):
    print(f"max |d{name} - ref| {(a - b).abs().max().item():.3e}  rel {((a - b).norm() / b.norm()).item():.3e}")
