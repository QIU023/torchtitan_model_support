#!/bin/bash
# On node 0, without a shared filesystem: pull node 1's run directories next to node 0's (node 1 writes node1.log,
# rc_node1, env_node1.txt, so nothing collides), then summarize every stage. Needs ssh from node 0 to node 1.
# Usage: NODE1=<address of node 1> [NODE1_PORT=22] collect.sh
. "$(dirname "$0")/env.sh"
set -eu
: "${NODE1:?set NODE1 to the address of node 1}"
rsync -a --exclude 'tt/profiling' -e "ssh -p ${NODE1_PORT:-22} -o StrictHostKeyChecking=no" \
  root@$NODE1:$OUT/$HW/ $OUT/$HW/
for stage in $OUT/$HW/*/; do
  [ -d "$stage" ] || continue
  python $K/summarize.py $stage > $stage/summary.tsv && echo "== $stage" && column -t -s $'\t' $stage/summary.tsv
done
