#!/bin/bash
# Raw RDMA write bandwidth of every active InfiniBand port between the two nodes, without NCCL: perftest's
# ib_write_bw in host memory, node 1 writing to node 0, one port after the other (port k of node 1 to port k of node 0).
# run_comm.sh calls it on both nodes at once before the NCCL measurement; node 0 serves, node 1 connects to MASTER_ADDR.
# Node 1 writes one line per port to rdma_node1.txt in MN_COMM_DIR ($OUT/$HW/comm); the raw output is in rdma_<hca>_node<k>.log.
. "$(dirname "$0")/env.sh"
set -u
command -v ib_write_bw > /dev/null || { echo "ib_write_bw not installed (perftest), skipped"; exit 0; }
# Nodes 0 and 1 stand for every pair: with more nodes, the others would contend for node 0's server ports.
[ "$NODE_RANK" -le 1 ] || { echo "node $NODE_RANK: the RDMA check runs between nodes 0 and 1 only"; exit 0; }
: "${MASTER_ADDR:?set MASTER_ADDR to the address of node 0}"
R=${MN_COMM_DIR:-$OUT/$HW/comm}
mkdir -p $R
out=$R/rdma_node$NODE_RANK.txt
: > $out
port=${MN_RDMA_PORT:-18515}
for dev in /sys/class/infiniband/*; do
  hca=$(basename $dev)
  [ "$(cat $dev/ports/1/link_layer 2>/dev/null)" = InfiniBand ] && grep -q ACTIVE $dev/ports/1/state 2>/dev/null || continue
  log=$R/rdma_${hca}_node$NODE_RANK.log
  if [ "$NODE_RANK" = 0 ]; then
    timeout 90 ib_write_bw -d $hca -p $port -D 5 -F --report_gbits > $log 2>&1
  else
    # The client keeps trying for longer than the server waits, so the two sides cannot drift apart by a port.
    for _ in $(seq 40); do
      sleep 3
      timeout 60 ib_write_bw -d $hca -p $port -D 5 -F --report_gbits $MASTER_ADDR > $log 2>&1 && break
    done
    echo "$hca $(awk '$1 ~ /^[0-9]+$/ && NF >= 5 { bw = $4 } END { print (bw ? bw " Gb/s" : "failed") }' $log)" >> $out
  fi
  port=$((port + 1))
done
cat $out
