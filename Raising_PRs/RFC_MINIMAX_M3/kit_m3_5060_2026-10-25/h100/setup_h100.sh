#!/bin/bash
# M3 kit on the H100 box: clone the fork's two M3 heads over HTTPS, build a venv with the torch nightly the 5060 runs used
# (2.16.0.dev20261024) for the driver's CUDA (cu130 when the driver reports CUDA >= 13, else cu126), then the tree's
# requirements, the VLM extras and the torchao nightly; check the imports. Writes $B/SETUP_OK on success.
. "$(dirname "$0")/env_h100.sh"
mkdir -p $B $R
exec > $B/setup.log 2>&1
set -eux
nvidia-smi
CUDA_DRV=$(nvidia-smi | grep -o "CUDA Version: [0-9.]*" | grep -o "[0-9]*\.[0-9]*")
if [ "${CUDA_DRV%%.*}" -ge 13 ]; then CU=cu130; else CU=cu126; fi
echo "driver CUDA $CUDA_DRV -> $CU"
command -v uv || { python3 -m pip install --user uv; export PATH=$HOME/.local/bin:$PATH; }
if [ ! -d $B/tt ]; then
  git clone --filter=blob:none --no-checkout https://github.com/QIU023/torchtitan.git $B/tt
fi
git -C $B/tt fetch origin minimax_m3 minimax_m3_5d
[ -d $TB ] || git -C $B/tt worktree add --detach $TB 9a5ab9ebe
[ -d $T5 ] || git -C $B/tt worktree add --detach $T5 89fc094d1
git -C $TB log --oneline -1; git -C $T5 log --oneline -1
[ -d $V ] || uv venv --python 3.12 $V
. $V/bin/activate
IDX=https://download.pytorch.org/whl/nightly/$CU
uv pip install --pre "torch==2.16.0.dev20261024+$CU" "torchvision==0.30.0.dev20261024+$CU" --index-url $IDX \
  || uv pip install --pre torch torchvision --index-url $IDX
python -c "import torch, torchvision; print(f'torch=={torch.__version__}'); print(f'torchvision=={torchvision.__version__}')" > $B/torch_pins.txt
cat $B/torch_pins.txt
cd $T5
uv pip install -c $B/torch_pins.txt -r requirements.txt -r requirements-dev.txt
uv pip install -c $B/torch_pins.txt -r .ci/docker/requirements-vlm.txt
USE_CPP=0 uv pip install -c $B/torch_pins.txt --pre --upgrade torchao --index-url $IDX
uv pip freeze > $B/venv_freeze.txt
python - <<'PY'
import torch, torchvision, triton
print("venv", torch.__version__, torch.version.git_version[:10], "cuda", torch.version.cuda, "tv", torchvision.__version__, "triton", triton.__version__)
print("devices", torch.cuda.device_count(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print("matmul", float((x @ x).float().abs().mean()))
import attn_gym, spmd_types, torch_remat, torch_checkpointing, torchao
print("attn_gym spmd_types torch_remat torch_checkpointing torchao import ok")
PY
touch $B/SETUP_OK
echo SETUP_OK
