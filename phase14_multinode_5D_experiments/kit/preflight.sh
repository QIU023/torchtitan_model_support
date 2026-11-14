#!/bin/bash
# Pipeline task preflight, on every node at once: this node's GPUs (preflight.py gemm, about 30 s), then the quick
# communication check over all nodes (run_comm.sh with messages of 256 MiB and up, plus the raw RDMA test), about two
# minutes. Node 0's pipeline then runs preflight.py verdict and stops before the matrix when something fails.
. "$(dirname "$0")/env.sh"
set -u
P=$OUT/$HW/preflight
mkdir -p $P
. $V/bin/activate
python $K/preflight.py gemm --out $P --seconds ${MN_PREFLIGHT_SECONDS:-30} > $P/gemm_node$NODE_RANK.log 2>&1 || exit 1
MN_COMM_TAG=preflight MN_COMM_RDMA=1 MN_COMM_MIN_BYTES=268435456 bash $K/run_comm.sh > $P/comm_node$NODE_RANK.log 2>&1
