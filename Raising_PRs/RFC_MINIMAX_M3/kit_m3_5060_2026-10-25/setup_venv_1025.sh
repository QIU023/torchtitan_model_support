#!/bin/bash
# 10-25 5060 box (8 x RTX 5060 Ti 16 GB, sm120, driver 580 / CUDA 13.0): venv_1025 = the latest cu130 nightly torch and
# torchvision, then the M3 branch's requirements.txt and requirements-dev.txt, the VLM extras and the torchao nightly, as the
# A4000 kit_a4000_2026-10-24/setup.sh did. torch and torchvision are held by a constraints file.
exec > /workspace/kit_m3_5060/setup_1025.log 2>&1
set -eux
export UV_CACHE_DIR=/workspace/.uvcache UV_HTTP_TIMEOUT=900
V=/workspace/venv_1025
T=/workspace/wt_m3
uv venv --python 3.12 $V
. $V/bin/activate
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130
python - > /workspace/kit_m3_5060/venv_1025_torch_pins.txt <<'PY'
import torch, torchvision
print(f"torch=={torch.__version__}")
print(f"torchvision=={torchvision.__version__}")
PY
cat /workspace/kit_m3_5060/venv_1025_torch_pins.txt
cd $T
uv pip install -c /workspace/kit_m3_5060/venv_1025_torch_pins.txt -r requirements.txt -r requirements-dev.txt
uv pip install -c /workspace/kit_m3_5060/venv_1025_torch_pins.txt -r .ci/docker/requirements-vlm.txt
USE_CPP=0 uv pip install -c /workspace/kit_m3_5060/venv_1025_torch_pins.txt --pre --upgrade torchao --index-url https://download.pytorch.org/whl/nightly/cu130
uv pip freeze > /workspace/kit_m3_5060/venv_1025_freeze.txt
python - <<'PY'
import torch, torchvision, triton
print("venv_1025", torch.__version__, torch.version.git_version[:10], "cuda", torch.version.cuda, "tv", torchvision.__version__, "triton", triton.__version__)
print("devices", torch.cuda.device_count(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
torch.backends.cuda.matmul.fp32_precision = "bfx9"
print("bfx9 ok")
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print("matmul", float((x @ x).float().abs().mean()))
import attn_gym, spmd_types, torch_remat, torch_checkpointing, torchao
print("attn_gym spmd_types torch_remat torch_checkpointing torchao import ok")
PY
echo SETUP_OK
