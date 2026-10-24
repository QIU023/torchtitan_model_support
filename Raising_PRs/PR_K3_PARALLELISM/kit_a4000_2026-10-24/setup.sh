#!/bin/bash
# 10-24 A4000 box (8 x RTX A4000 16 GB, sm86, PCIe, driver 580 / CUDA 13.0): venv_1024 = the latest cu130 nightly torch and
# torchvision, then main's requirements.txt and requirements-dev.txt, the VLM extras and the torchao nightly, the way the B200
# lane installs them (flash-attn-4, flash-attn-3 and dist_moe are SM90/SM100 lane extras, not installed). torch and
# torchvision are held by a constraints file so the later installs cannot swap the nightly for a PyPI release.
exec > /workspace/setup_1024.log 2>&1
set -eux
export UV_CACHE_DIR=/workspace/.uvcache UV_HTTP_TIMEOUT=900
V=/workspace/venv_1024
T=/workspace/tt_main
uv venv --python 3.12 $V
. $V/bin/activate
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130
python - > /workspace/venv_1024_torch_pins.txt <<'PY'
import torch, torchvision
print(f"torch=={torch.__version__}")
print(f"torchvision=={torchvision.__version__}")
PY
cat /workspace/venv_1024_torch_pins.txt
cd $T
uv pip install -c /workspace/venv_1024_torch_pins.txt -r requirements.txt -r requirements-dev.txt
uv pip install -c /workspace/venv_1024_torch_pins.txt -r .ci/docker/requirements-vlm.txt
USE_CPP=0 uv pip install -c /workspace/venv_1024_torch_pins.txt --pre --upgrade torchao --index-url https://download.pytorch.org/whl/nightly/cu130
uv pip freeze > /workspace/venv_1024_freeze.txt
python - <<'PY'
import torch, torchvision, triton
print("venv_1024", torch.__version__, torch.version.git_version[:10], "cuda", torch.version.cuda, "tv", torchvision.__version__, "triton", triton.__version__)
print("devices", torch.cuda.device_count(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
import torch.distributed.pipelining.stage as s
print("has _make_forward_chunk_state", hasattr(s._PipelineStageBase, "_make_forward_chunk_state"))
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print("matmul", float((x @ x).float().abs().mean()))
import attn_gym, spmd_types, torch_remat, torch_checkpointing, torchao
print("attn_gym spmd_types torch_remat torch_checkpointing torchao import ok")
PY
echo SETUP_OK
