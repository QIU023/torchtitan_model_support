#!/bin/bash
# One cell on NNODES x NPROC GPUs. Run the same command on every node, or once per node under srun.
# Usage: launch.sh <run_dir>. Reads the MN_* variables of torchtitan_recipes/multinode_5d.py, plus MASTER_ADDR
# (node 0), MN_JOB (rendezvous id, unique per cell) and MN_CACHE (compile cache shared by the cells of one table).
. "$(dirname "$0")/env.sh"
set -uo pipefail
D=$1
mkdir -p $D
NODE_RANK=${NODE_RANK:-${SLURM_NODEID:-0}}
: "${MASTER_ADDR:?set MASTER_ADDR to the address of node 0}"
CACHE=${MN_CACHE:-$OUT/cache/$HW}/node$NODE_RANK
mkdir -p $CACHE
. $V/bin/activate
cd $T
env | grep -E '^(MN_|HW=|NCCL_)' | sort > $D/env_node$NODE_RANK.txt
git -C $T rev-parse HEAD > $D/tree.txt
# Every rank logs its step line, so the memory of each pipeline stage is visible.
LOG_RANK=$(seq -s, 0 $((NNODES * NPROC - 1))) PYTHONPATH=$T \
TORCHINDUCTOR_CACHE_DIR=$CACHE/ic TRITON_CACHE_DIR=$CACHE/tc \
timeout ${MN_TIMEOUT:-5400} torchrun --nnodes=$NNODES --nproc_per_node=$NPROC --node_rank=$NODE_RANK \
  --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT --rdzv_id=${MN_JOB:-p14} \
  --role rank --tee 3 \
  -m torchtitan.train --module torchtitan_recipes.multinode_5d --config cell --output-dir $D/tt \
  > $D/node$NODE_RANK.log 2>&1 < /dev/null
rc=$?
echo $rc > $D/rc_node$NODE_RANK
exit $rc
