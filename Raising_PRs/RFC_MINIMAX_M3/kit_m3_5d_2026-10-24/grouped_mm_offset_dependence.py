"""Does a row of torch._grouped_mm depend on the group's start offset or the row's position in it? (bf16, this GPU)"""
import torch
torch.manual_seed(0)
dev = "cuda"
D, F = 256, 256
w = torch.randn(3, F, D, device=dev, dtype=torch.bfloat16)
rows = torch.randn(5, D, device=dev, dtype=torch.bfloat16)
ref = rows @ w[1].T
results = []
for lead in (0, 1, 3, 7, 8, 13, 16, 37):            # rows of expert 0 before expert 1's group
    for pos in (0, 1, 5, 11):                         # rows of expert 1 before the target rows
        x = torch.cat([torch.randn(lead, D, device=dev, dtype=torch.bfloat16),
                       torch.randn(pos, D, device=dev, dtype=torch.bfloat16), rows,
                       torch.randn(9, D, device=dev, dtype=torch.bfloat16)])
        offs = torch.tensor([lead, lead + pos + 5, lead + pos + 5 + 9], device=dev, dtype=torch.int32)
        out = torch._grouped_mm(x, w.transpose(-2, -1), offs=offs)[lead + pos : lead + pos + 5]
        results.append((lead, pos, torch.equal(out, ref), (out.float() - ref.float()).abs().max().item()))
bad = [r for r in results if not r[2]]
print(f"{len(results) - len(bad)}/{len(results)} placements bitwise equal to a plain mm")
for lead, pos, _, diff in bad[:8]:
    print(f"  group start {lead}, row offset {pos}: max|diff| {diff:.3e}")
