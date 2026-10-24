import torch
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.minimax_m3 import build_model_config

torch.manual_seed(0)
dev = "cuda"
for flavor in ("debugmodel", "debugmodel_flash"):
    config = build_model_config(flavor, seq_len=512, attn_backend="flex", parallelism=ParallelismConfig())
    model = config.build().to(dev)
    with torch.device(dev):
        model.init_states(buffer_device=torch.device(dev))
    model.train()
    T, image_id = 512, 2000
    positions = torch.cat([torch.arange(300), torch.arange(212)]).to(dev)
    tokens = torch.randint(0, 1990, (T,), device=dev)
    tokens[10:14] = image_id                         # one 4x4-patch image -> 4 merged tokens
    grid = torch.tensor([[1, 4, 4]], device=dev)
    pixels = torch.randn(16, 1176, device=dev)
    meta = model._get_attention_metadata(positions=positions)
    logits = model(tokens, positions=positions, attention_metadata=meta, pixel_values=pixels, grid_thw=grid,
                   special_tokens={"image_id": image_id, "video_id": 2001}, aux_loss_denominators=torch.tensor([float(T)], device=dev))
    loss = torch.nn.functional.cross_entropy(logits.float(), torch.roll(tokens, -1))
    loss.backward()
    named = dict(model.named_parameters())
    idx = [n for n in named if ".indexer." in n]
    no_grad = [n for n, p in named.items() if p.grad is None or p.grad.abs().sum() == 0]
    print(flavor, "| logits", tuple(logits.shape), "| loss", round(loss.item(), 4), "| params", sum(p.numel() for p in named.values()))
    print("  indexer params", len(idx), "all with grad:", all(named[n].grad is not None and named[n].grad.abs().sum() > 0 for n in idx))
    print("  params without grad:", no_grad[:8], "... total", len(no_grad))
