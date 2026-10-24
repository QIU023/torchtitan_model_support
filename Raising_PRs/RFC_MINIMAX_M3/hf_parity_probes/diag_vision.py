"""Vision tower: HF fp32 and TorchTitan fp32 against HF float64 on the same pixels."""
import copy
import torch
from huggingface_hub import snapshot_download
import scripts.checkpoint_conversion.numerical_tests_minimax_m3 as nt
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.minimax_m3 import build_model_config
from torchtitan.models.minimax_m3.state_dict_adapter import MiniMaxM3StateDictAdapter

print("cudnn.allow_tf32 =", torch.backends.cudnn.allow_tf32, " matmul.allow_tf32 =", torch.backends.cuda.matmul.allow_tf32)
torch.manual_seed(42)
dev = torch.device("cuda")
path = snapshot_download(repo_id=nt._HF_REPO_ID, revision=nt._HF_REVISION, allow_patterns=nt._HF_ALLOW_PATTERNS)
cfg = nt._use_released_vocab(build_model_config("debugmodel", attn_backend="flex", parallelism=ParallelismConfig()), path)
tt = nt._build_tt_model(cfg, torch.float32)
hf = nt._build_hf_model(path, cfg, MiniMaxM3StateDictAdapter(cfg, hf_assets_path=None).to_hf(tt.state_dict()), torch.float32, "grouped_mm").to(dev)
tt = tt.to(dev)
grid = torch.tensor([[1, 24, 24]], device=dev)
pix = torch.randn(576, 1176, device=dev)

def hf_features(model, pixels):
    return model.model.get_image_features(pixels, grid, return_dict=True).pooler_output
with torch.no_grad():
    hf64 = copy.deepcopy(hf).double()
    ref = torch.cat(list(hf_features(hf64, pix.double())), 0) if isinstance(hf_features(hf64, pix.double()), (list, tuple)) else hf_features(hf64, pix.double())
    out_hf = hf_features(hf, pix); out_hf = torch.cat(list(out_hf), 0) if isinstance(out_hf, (list, tuple)) else out_hf
    out_tt = tt.vision_encoder(pix, grid_thw=grid)
    torch.backends.cudnn.allow_tf32 = False
    out_hf_notf32 = hf_features(hf, pix); out_hf_notf32 = torch.cat(list(out_hf_notf32), 0) if isinstance(out_hf_notf32, (list, tuple)) else out_hf_notf32
scale = ref.abs().max().item()
for name, out in (("HF fp32 (cudnn tf32 on)", out_hf), ("HF fp32 (cudnn tf32 off)", out_hf_notf32), ("TT fp32", out_tt)):
    d = (out.double() - ref).abs()
    print(f"{name:26s} vs HF fp64: max|diff| {d.max().item():.3e}  rel {d.max().item() / scale:.2e}")
