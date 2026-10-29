#!/bin/bash
# nccl-tests for multinode.md 5.1 (goal E2: intra-node and cross-node busbw). Run on node 0; needs MPI and
# passwordless ssh to node 1. Usage: NODE0=<host> NODE1=<host> nccl_tests.sh
# Builds nccl-tests at the commit multinode.md cites (afd59ab) against the venv's NCCL, then runs 1 GiB-per-rank sweeps.
. "$(dirname "$0")/env.sh"
set -eux
: "${NODE0:?}" "${NODE1:?}"
NT=$B/nccl-tests
R=$OUT/$HW/nccl_tests
mkdir -p $R
. $V/bin/activate
NCCL_HOME=${NCCL_HOME:-$(python -c "import nvidia.nccl, os; print(os.path.dirname(nvidia.nccl.__file__))")}
if [ ! -x $NT/build/all_reduce_perf ]; then
  [ -d $NT ] || git clone https://github.com/NVIDIA/nccl-tests.git $NT
  git -C $NT checkout afd59ab
  make -C $NT MPI=1 MPI_HOME=${MPI_HOME:-/usr/lib/x86_64-linux-gnu/openmpi} NCCL_HOME=$NCCL_HOME CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} -j
fi
export LD_LIBRARY_PATH=$NCCL_HOME/lib:${LD_LIBRARY_PATH:-}
SWEEP="-b 8M -e 8G -f 2 -g 1"
MPI="mpirun --allow-run-as-root --bind-to none -x LD_LIBRARY_PATH -x NCCL_DEBUG -x NCCL_IB_HCA -x NCCL_SOCKET_IFNAME -x NCCL_NVLS_ENABLE"
# Intra-node, 8 ranks on node 0.
for t in all_reduce all_gather reduce_scatter alltoall; do
  $MPI -np 8 -H $NODE0:8 $NT/build/${t}_perf $SWEEP > $R/${t}_8.txt
done
# Cross-node: one pair, then all 16 ranks.
$MPI -np 2 -H $NODE0:1,$NODE1:1 $NT/build/sendrecv_perf $SWEEP > $R/sendrecv_2_cross.txt
for t in all_reduce all_gather alltoall; do
  $MPI -np 16 -H $NODE0:8,$NODE1:8 $NT/build/${t}_perf $SWEEP > $R/${t}_16.txt
done
grep -H "Avg bus bandwidth" $R/*.txt
