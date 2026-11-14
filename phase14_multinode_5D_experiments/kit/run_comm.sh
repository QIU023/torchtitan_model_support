#!/bin/bash
# Multi-node communication check; run the same command on both nodes (on the k8s pods, pipeline.py runs it first).
# comm_bench.py over all NNODES x NPROC ranks with NCCL_DEBUG=INFO, then a verdict from this node's NCCL log:
# which transport NCCL picked (IB or TCP sockets), whether the inter-node channels use GPUDirect RDMA, and on node 0
# the busbw of the large messages and the alpha-beta fit of every op and group. Output in $OUT/$HW/comm/; the run with
# NCCL's own choices also measures the raw RDMA bandwidth of every IB port first (rdma_bw.sh). MN_COMM_TAG=<tag> with
# NCCL_* variables is one tuning variant (comm_sweep.tsv), written to $OUT/$HW/comm_<tag>/ (comm_compare.py compares).
. "$(dirname "$0")/env.sh"
set -u
: "${MASTER_ADDR:?set MASTER_ADDR to the address of node 0}"
R=$OUT/$HW/comm${MN_COMM_TAG:+_$MN_COMM_TAG}
mkdir -p $R
L=$R/comm_node$NODE_RANK.log
ulimit -l unlimited 2>/dev/null || true
. $V/bin/activate
# The raw RDMA test runs with NCCL's own choices, and in any run that asks for it (MN_COMM_RDMA=1, the preflight).
if [ -z "${MN_COMM_TAG:-}" ] || [ "${MN_COMM_RDMA:-0}" = 1 ]; then
  MN_COMM_DIR=$R bash $K/rdma_bw.sh > $R/rdma_run_node$NODE_RANK.log 2>&1
fi
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,ENV,TUNING timeout ${MN_TIMEOUT:-1800} \
  torchrun --nnodes=$NNODES --nproc_per_node=$NPROC --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR --master_port=${MN_PORT:-$((MASTER_PORT + 900))} \
  $K/comm_bench.py --min-bytes ${MN_COMM_MIN_BYTES:-256} --out $R/comm_bench.json > $L 2>&1 < /dev/null
rc=$?
{
  echo "node $NODE_RANK rc=$rc ulimit_l=$(ulimit -l) log=$L"
  echo "NCCL variables: $(env | grep '^NCCL_' | grep -v '^NCCL_DEBUG' | sort | tr '\n' ' ')"
  if [ -s $R/rdma_node$NODE_RANK.txt ]; then
    echo "raw RDMA write bandwidth per IB port (ib_write_bw, host memory, node 1 to node 0):"
    cat $R/rdma_node$NODE_RANK.txt
  fi
  echo "NCCL transport: $(grep -m1 -oE 'NET/(IB|Socket) : Using[^;]*' $L)"
  if grep -q "Using network IB" $L; then echo "InfiniBand: yes"
  else echo "InfiniBand: NO, NCCL fell back to sockets (privileged pod? /dev/infiniband? NCCL_IB_HCA? libibverbs?)"; fi
  if grep -qE "NET/IB/[0-9]+/GDRDMA" $L; then echo "GPUDirect RDMA: yes"
  else echo "GPUDirect RDMA: not seen (nvidia_peermem or dma-buf missing; traffic is staged through host memory)"; fi
  if [ "$NODE_RANK" = 0 ]; then
    echo "busbw of messages >= 256 MiB (GB/s):"
    awk '$1 ~ /^(all_reduce|all_gather|reduce_scatter|alltoall|sendrecv)$/ && $4 >= 268435456' $L
    echo "alpha-beta fit, t = alpha + size / beta (through torch.distributed, alpha includes each call's launch overhead):"
    grep '^fit ' $L
  fi
} | tee $R/verdict_node$NODE_RANK.txt
exit $rc
