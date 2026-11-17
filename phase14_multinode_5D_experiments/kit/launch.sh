#!/bin/bash
# One cell on NNODES x NPROC GPUs. Run the same command on every node (on the k8s pods, pipeline.py does it).
# Usage: launch.sh <run_dir>. Reads the MN_* variables of torchtitan_recipes/multinode_5d.py, plus MASTER_ADDR
# (node 0), MN_PORT (rendezvous port; run_matrix.sh gives every cell its own) and MN_CACHE (compile cache shared by
# the cells of one table). Static rendezvous: node 0 hosts the store and global rank = NODE_RANK * NPROC + local rank,
# so node 0 holds ranks 0-7 (pipeline stage 0 when pp = 2).
. "$(dirname "$0")/env.sh"
set -uo pipefail
D=$1
mkdir -p $D
: "${MASTER_ADDR:?set MASTER_ADDR to the address of node 0}"
PORT=${MN_PORT:-$MASTER_PORT}
CACHE=${MN_CACHE:-$OUT/cache/$HW}/node$NODE_RANK
mkdir -p $CACHE
# InfiniBand memory registration needs an unlimited locked-memory limit.
ulimit -l unlimited 2>/dev/null || true
. $V/bin/activate
cd $T
{ env | grep -E '^(MN_|HW=|NCCL_|UCX_|SHARP_|GLOO_|NODE_RANK=|MASTER_ADDR=)' | sort; echo "ulimit_l=$(ulimit -l)"; } \
  > $D/env_node$NODE_RANK.txt
git -C $T rev-parse HEAD > $D/tree.txt
# NCCL's transport choice (IB or sockets, GPUDirect RDMA) goes to per-process files beside the logs, so every run
# checks the network without a separate test (summarize.py reads them); MN_NCCL_INFO=0 turns it off.
NCCL_LOG=()
if [ "${MN_NCCL_INFO:-1}" = 1 ]; then
  mkdir -p $D/nccl
  NCCL_LOG=(NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET NCCL_DEBUG_FILE=$D/nccl/%h.%p.log)
fi
# MN_NCCL_LIB preloads a replacement NCCL (ext_build.sh nccl) ahead of the one torch bundles; summary.tsv's nccl
# column shows the version that ran. HybridEP and DeepEP get the CUDA toolkit ext_build.sh kept on the volume.
EXTRA=()
# A cold compile in step 1 can outlast the 300 s first-step NCCL timeout (an 8-26 cold KDA compile took 7 minutes).
[ "${MN_COMPILE:-1}" = 0 ] || EXTRA+=(MN_NCCL_INIT_TIMEOUT=${MN_NCCL_INIT_TIMEOUT:-1800})
[ -z "${MN_NCCL_LIB:-}" ] || EXTRA+=(LD_PRELOAD=$MN_NCCL_LIB)
if [ "${MN_EP_BACKEND:-standard}" != standard ]; then
  CUDA_DIR=$(ls -d $B/cuda-* 2>/dev/null | tail -n 1)
  [ -z "$CUDA_DIR" ] || EXTRA+=(CUDA_HOME=$CUDA_DIR PATH=$CUDA_DIR/bin:$PATH)
fi
# Every rank logs its step line, so the memory of each pipeline stage is visible; lines carry the global rank.
env "${NCCL_LOG[@]}" "${EXTRA[@]}" LOG_RANK=$(seq -s, 0 $((NNODES * NPROC - 1))) PYTHONPATH=$T \
TORCHINDUCTOR_CACHE_DIR=$CACHE/ic TRITON_CACHE_DIR=$CACHE/tc \
PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True} PYTHONUNBUFFERED=1 \
timeout ${MN_TIMEOUT:-5400} torchrun --nnodes=$NNODES --nproc_per_node=$NPROC --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR --master_port=$PORT --tee 3 --log_line_prefix_template '[rank${rank}]:' \
  -m torchtitan.train --module torchtitan_recipes.multinode_5d --config cell --output-dir $D/tt \
  > $D/node$NODE_RANK.log 2>&1 < /dev/null
rc=$?
echo $rc > $D/rc_node$NODE_RANK
exit $rc
