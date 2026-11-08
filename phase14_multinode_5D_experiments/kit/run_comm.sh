#!/bin/bash
# Multi-node communication check; run the same command on both nodes (on the k8s pods, pipeline.py runs it first).
# comm_bench.py over all NNODES x NPROC ranks with NCCL_DEBUG=INFO, then a verdict from this node's NCCL log:
# which transport NCCL picked (IB or TCP sockets), whether the inter-node channels use GPUDirect RDMA, and on node 0
# the busbw of the large messages. Output in $OUT/$HW/comm/.
. "$(dirname "$0")/env.sh"
set -u
: "${MASTER_ADDR:?set MASTER_ADDR to the address of node 0}"
R=$OUT/$HW/comm
mkdir -p $R
L=$R/comm_node$NODE_RANK.log
ulimit -l unlimited 2>/dev/null || true
. $V/bin/activate
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,ENV timeout ${MN_TIMEOUT:-1800} \
  torchrun --nnodes=$NNODES --nproc_per_node=$NPROC --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR --master_port=${MN_PORT:-$((MASTER_PORT + 900))} \
  $K/comm_bench.py --out $R/comm_bench.json > $L 2>&1 < /dev/null
rc=$?
{
  echo "node $NODE_RANK rc=$rc ulimit_l=$(ulimit -l) log=$L"
  echo "NCCL transport: $(grep -m1 -oE 'NET/(IB|Socket) : Using[^;]*' $L)"
  if grep -q "Using network IB" $L; then echo "InfiniBand: yes"
  else echo "InfiniBand: NO, NCCL fell back to sockets (privileged pod? /dev/infiniband? NCCL_IB_HCA? libibverbs?)"; fi
  if grep -qE "NET/IB/[0-9]+/GDRDMA" $L; then echo "GPUDirect RDMA: yes"
  else echo "GPUDirect RDMA: not seen (nvidia_peermem or dma-buf missing; traffic is staged through host memory)"; fi
  if [ "$NODE_RANK" = 0 ]; then
    echo "busbw of messages >= 256 MiB (GB/s):"
    awk '$1 ~ /^(all_reduce|all_gather|reduce_scatter|alltoall|sendrecv)$/ && $4 >= 268435456' $L
  fi
} | tee $R/verdict_node$NODE_RANK.txt
exit $rc
