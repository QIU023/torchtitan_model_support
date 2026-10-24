import sys, torch
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.minimax_m3 import build_model_config
torch.manual_seed(0)
dev = "cuda"; T = int(sys.argv[1]); dtype = getattr(torch, sys.argv[2]); with_image = sys.argv[3] == "1"
config = build_model_config("debugmodel", seq_len=T, parallelism=ParallelismConfig())
model = config.build().to(dev)
with torch.device(dev):
    model.init_states(buffer_device=torch.device(dev))
model = model.to(dtype); model.train()
positions = torch.cat([torch.arange(T - 700), torch.arange(700)]).to(dev)
tokens = torch.randint(0, 1990, (T,), device=dev)
kw = {}
if with_image:
    tokens[10:14] = 2000
    kw = dict(pixel_values=torch.randn(16, 1176, device=dev, dtype=dtype), grid_thw=torch.tensor([[1, 4, 4]], device=dev))
meta = model._get_attention_metadata(positions=positions)
logits = model(tokens, positions=positions, attention_metadata=meta, special_tokens={"image_id": 2000, "video_id": 2001},
               aux_loss_denominators=torch.tensor([float(T)], device=dev), **kw)
logits.float().mean().backward()
torch.cuda.synchronize(); print("OK", T, dtype, with_image)
