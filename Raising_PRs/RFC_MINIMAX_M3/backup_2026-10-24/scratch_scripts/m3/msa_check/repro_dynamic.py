import sys, torch
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.minimax_m3 import build_model_config
torch.manual_seed(0); dev = "cuda"; T = 2048
config = build_model_config("debugmodel", seq_len=T, parallelism=ParallelismConfig())
model = config.build().to(dev)
with torch.device(dev):
    model.init_states(buffer_device=torch.device(dev))
model = model.to(torch.bfloat16); model.train()
only_vision = sys.argv[1] == "vision"
for step, (h, w) in enumerate([(4, 4), (6, 8), (10, 12), (14, 6)]):
    if only_vision:
        out = model.vision_encoder(torch.randn(h * w, 1176, device=dev, dtype=torch.bfloat16), grid_thw=torch.tensor([[1, h, w]], device=dev))
        out.float().sum().backward(); torch.cuda.synchronize(); print("vision step", step, (h, w), "ok", flush=True); continue
    positions = torch.cat([torch.arange(T - 700), torch.arange(700)]).to(dev)
    tokens = torch.randint(0, 1990, (T,), device=dev); n = h * w // 4; tokens[10:10 + n] = 2000
    meta = model._get_attention_metadata(positions=positions)
    logits = model(tokens, positions=positions, attention_metadata=meta, pixel_values=torch.randn(h * w, 1176, device=dev, dtype=torch.bfloat16),
                   grid_thw=torch.tensor([[1, h, w]], device=dev), special_tokens={"image_id": 2000, "video_id": 2001},
                   aux_loss_denominators=torch.tensor([float(T)], device=dev))
    logits.float().mean().backward(); torch.cuda.synchronize(); print("step", step, (h, w), "ok", flush=True)
