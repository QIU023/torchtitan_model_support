#!/bin/bash
# Per container, on every start (bootstrap.sh runs it; setup.sh too): the system packages a fresh container image
# lacks, then an interconnect report. Packages: git, a C compiler for Triton's launcher, and the InfiniBand user-space
# libraries NCCL loads (without libibverbs and the mlx5 provider NCCL silently falls back to TCP sockets).
set -x
if command -v apt-get >/dev/null && [ "$(id -u)" = 0 ]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends git curl ca-certificates build-essential rsync tmux iproute2 procps \
    rdma-core ibverbs-providers libibverbs1 ibverbs-utils infiniband-diags
fi
# The clone and the worktree on a shared filesystem may belong to another uid than this container's user.
git config --global --add safe.directory '*' 2>/dev/null
set +x
ulimit -l unlimited 2>/dev/null
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | sort | uniq -c
echo "ib devices: $(ibv_devinfo -l 2>/dev/null | grep -c mlx5)"
ibstat 2>/dev/null | grep -E "CA '|State:|Rate:" | paste - - - | head -n 16
echo "nvidia_peermem: $(grep -c nvidia_peermem /proc/modules)"
echo "ulimit -l: $(ulimit -l)"
df -h /dev/shm | tail -n 1
exit 0
