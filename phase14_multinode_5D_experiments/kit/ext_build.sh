#!/bin/bash
# Build communication libraries onto the shared volume, once (k8s.sh build ..., or bootstrap.sh for MN_EP_LIBS).
#   ext_build.sh nccl [ref] [patch]  NCCL at a git ref (default: the version torch bundles, so the first swap changes
#                                    nothing), optionally patched, plus NCCL's example tuner and profiler plugins, into
#                                    $B/nccl/<ref>[-<patch>]/{lib,plugins}; runs pick it with MN_NCCL_LIB=<dir>/lib/libnccl.so.2
#   ext_build.sh hybridep | deepep   DeepEP's hybrid-ep branch or DeepEP v2, at titan CI's pinned commits, into the venv
# Extensions build with the CUDA toolkit torch was built with; the container's is older, so the matching one is
# installed from NVIDIA's apt repository once and kept at $B/cuda-<version>.
. "$(dirname "$0")/env.sh"
set -eux
. $V/bin/activate
CUV=$(python -c "import torch; print(torch.version.cuda)")
CUDA_DIR=$B/cuda-$CUV
if [ ! -x $CUDA_DIR/bin/nvcc ]; then
  apt-get update && apt-get install -y --no-install-recommends cuda-toolkit-${CUV/./-}
  cp -a /usr/local/cuda-$CUV $CUDA_DIR
fi
export CUDA_HOME=$CUDA_DIR PATH=$CUDA_DIR/bin:$PATH
ARCH=$(python -c "import torch; m, n = torch.cuda.get_device_capability(0); print(f'{m}.{n}')")
case ${1:-} in
  nccl)
    REF=${2:-v$(python -c "import torch; print('.'.join(map(str, torch.cuda.nccl.version())))")-1}
    SRC=$B/nccl/src
    [ -d $SRC/.git ] || git clone https://github.com/NVIDIA/nccl.git $SRC
    git -C $SRC fetch --tags origin
    git -C $SRC checkout --force $REF
    git -C $SRC clean -fdx
    [ -z "${3:-}" ] || git -C $SRC apply "$3"
    OUTD=$B/nccl/$REF${3:+-$(basename "$3" .patch)}
    SM=${ARCH/./}
    make -C $SRC -j$(nproc) src.build BUILDDIR=$OUTD CUDA_HOME=$CUDA_HOME NVCC_GENCODE="-gencode=arch=compute_$SM,code=sm_$SM"
    mkdir -p $OUTD/plugins
    for d in ext-tuner/example ext-profiler/example; do
      [ -d $SRC/$d ] || continue
      make -C $SRC/$d CUDA_HOME=$CUDA_HOME
      cp $SRC/$d/*.so $OUTD/plugins/
    done
    echo "built: MN_NCCL_LIB=$OUTD/lib/libnccl.so.2; plugins in $OUTD/plugins: $(ls $OUTD/plugins | tr '\n' ' ')"
    ;;
  hybridep | deepep)
    apt-get install -y --no-install-recommends libibverbs-dev rdma-core libmlx5-1
    uv pip install "nvidia-nvshmem-cu${CUV%%.*}" || true
    D=$B/ext_src/$1
    if [ $1 = hybridep ]; then BR=(--branch hybrid-ep); COMMIT=f725d29699f5bda9ba789456bb9579af69844685; MOD=HybridEPBuffer
    else BR=(); COMMIT=01dc3aaac82068020353dce2c302e38153c0bfaa; MOD=ElasticBuffer; fi
    [ -d $D/.git ] || git clone "${BR[@]}" https://github.com/deepseek-ai/DeepEP.git $D
    git -C $D fetch origin
    git -C $D checkout $COMMIT
    git -C $D submodule update --init --recursive
    # CUDA 13 keeps the CCCL headers under include/cccl.
    [ ! -d $CUDA_HOME/include/cccl/cuda ] || [ -e $CUDA_HOME/include/cuda ] || ln -s $CUDA_HOME/include/cccl/cuda $CUDA_HOME/include/cuda
    (cd $D && TORCH_CUDA_ARCH_LIST=$ARCH uv pip install --no-build-isolation .)
    python -c "from deep_ep import $MOD; print('$MOD ok')"
    touch $B/EXT_OK_$1
    ;;
  *) echo "usage: ext_build.sh nccl [ref] [patch] | hybridep | deepep" >&2; exit 1 ;;
esac
