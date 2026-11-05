#!/bin/bash
# Run once on every node (or once on a shared filesystem): clone the fork, check out phase14_multinode_5d, build a
# venv with the torch nightly the M3 / K3 kits used (cu130 / cu128 / cu126 by the driver's CUDA version), install
# the tree's requirements, then check the GPU, its compute capability and the imports. Writes $B/SETUP_OK.
. "$(dirname "$0")/env.sh"
mkdir -p $B $OUT
exec > >(tee $B/setup_$(hostname).log) 2>&1
set -eux
# Container images (Nebius pods, vast): git, a C compiler for Triton's launcher, and the InfiniBand user-space
# libraries NCCL loads (without libibverbs and the mlx5 provider NCCL silently falls back to TCP sockets).
if command -v apt-get >/dev/null && [ "$(id -u)" = 0 ]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends git curl ca-certificates build-essential rsync tmux iproute2 \
    rdma-core ibverbs-providers libibverbs1 ibverbs-utils infiniband-diags
fi
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
CUDA_DRV=$(nvidia-smi | grep -o "CUDA Version: [0-9.]*" | grep -o "[0-9]*\.[0-9]*")
DRV_MAJOR=${CUDA_DRV%%.*}
DRV_MINOR=${CUDA_DRV#*.}
if [ "$DRV_MAJOR" -ge 13 ]; then CU=cu130
elif [ "$DRV_MAJOR" -eq 12 ] && [ "${DRV_MINOR%%.*}" -ge 8 ]; then CU=cu128
else CU=cu126; fi
export PATH=$HOME/.local/bin:$PATH
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh || python3 -m pip install --user uv
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
# Interconnect report (informational): HCAs and their state, GPUDirect RDMA module, locked memory, /dev/shm.
set +e
ulimit -l unlimited 2>/dev/null
echo "ib devices: $(ibv_devinfo -l 2>/dev/null | grep -c mlx5)"
ibstat 2>/dev/null | grep -E "CA '|State:|Rate:" | paste - - - | head -n 16
echo "nvidia_peermem: $(grep -c nvidia_peermem /proc/modules)"
echo "ulimit -l: $(ulimit -l)"
df -h /dev/shm | tail -n 1
set -e
touch $B/SETUP_OK
echo SETUP_OK
