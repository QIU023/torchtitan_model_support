#!/bin/bash
# Run once on every node (or once on a shared filesystem): clone the fork, check out phase14_multinode_5d, build a
# venv with the torch nightly the M3 / K3 kits used (cu130 when the driver reports CUDA >= 13, else cu126), install
# the tree's requirements, then check the GPU, its compute capability and the imports. Writes $B/SETUP_OK.
. "$(dirname "$0")/env.sh"
mkdir -p $B $OUT
exec > >(tee $B/setup_$(hostname).log) 2>&1
set -eux
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
CUDA_DRV=$(nvidia-smi | grep -o "CUDA Version: [0-9.]*" | grep -o "[0-9]*\.[0-9]*")
if [ "${CUDA_DRV%%.*}" -ge 13 ]; then CU=cu130; else CU=cu126; fi
command -v uv || { python3 -m pip install --user uv; export PATH=$HOME/.local/bin:$PATH; }
[ -d $TT ] || git clone --filter=blob:none --no-checkout https://github.com/QIU023/torchtitan.git $TT
git -C $TT fetch origin $BRANCH
[ -d $T ] || git -C $TT worktree add --detach $T origin/$BRANCH
git -C $T log --oneline -1
[ "$(git -C $T rev-parse --short=9 HEAD)" = "$BRANCH_SHA" ] || echo "WARNING: $T is not at $BRANCH_SHA"
[ -d $V ] || uv venv --python 3.12 $V
. $V/bin/activate
IDX=https://download.pytorch.org/whl/nightly/$CU
uv pip install --pre "torch==$TORCH_NIGHTLY+$CU" "torchvision==$TORCHVISION_NIGHTLY+$CU" --index-url $IDX \
  || uv pip install --pre torch torchvision --index-url $IDX
python -c "import torch, torchvision; print(f'torch=={torch.__version__}'); print(f'torchvision=={torchvision.__version__}')" > $B/torch_pins.txt
cd $T
uv pip install -c $B/torch_pins.txt -r requirements.txt -r requirements-dev.txt
uv pip install -c $B/torch_pins.txt -r .ci/docker/requirements-vlm.txt
USE_CPP=0 uv pip install -c $B/torch_pins.txt --pre --upgrade torchao --index-url $IDX
uv pip freeze > $B/venv_freeze_$(hostname).txt
python - <<'PY'
import torch, triton
cap = torch.cuda.get_device_capability(0)
print("torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__)
print("devices", torch.cuda.device_count(), torch.cuda.get_device_name(0), cap)
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print("matmul", float((x @ x).float().abs().mean()))
import attn_gym, spmd_types, torch_remat, torchao
print("imports ok")
if cap < (9, 0):
    print("compute capability", cap, "< 9.0: run with HW=a100 so the KDA gate uses impl='reference'")
PY
touch $B/SETUP_OK
echo SETUP_OK
