#!/bin/bash
# Replay one measured cell's collectives in isolation (comm_replay.py) over all nodes; run the same command on every
# node, or k8s.sh replay <stage> <row>. Needs the stage's trace_report_replay.json (trace_report.py writes it).
# Usage: replay.sh <cell dir>. Output in <cell>/replay/: replay.tsv from node 0, a log, NCCL INFO and rc per node.
. "$(dirname "$0")/env.sh"
set -u
C=$1
R=$C/replay
mkdir -p $R
: "${MASTER_ADDR:?set MASTER_ADDR to the address of node 0}"
. $V/bin/activate
env ${MN_NCCL_LIB:+LD_PRELOAD=$MN_NCCL_LIB} NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET NCCL_DEBUG_FILE=$R/nccl_%h.%p.log \
  timeout ${MN_TIMEOUT:-1800} torchrun --nnodes=$NNODES --nproc_per_node=$NPROC --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR --master_port=${MN_PORT:-$((MASTER_PORT + 950))} \
  $K/comm_replay.py --plan $(dirname $C)/trace_report_replay.json --cell $(basename $C) --out $R/replay.tsv \
  > $R/replay_node$NODE_RANK.log 2>&1 < /dev/null
rc=$?
echo $rc > $R/rc_node$NODE_RANK
exit $rc
