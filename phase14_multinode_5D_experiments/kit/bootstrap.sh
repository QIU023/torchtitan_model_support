#!/bin/bash
# Container entry of the phase 14 pods (nebius/ and together/ manifests). On every start: the container's packages;
# once per shared filesystem: the venv and the titan branch (one pod runs setup.sh under a lock, the other waits);
# then the unattended pipeline (pipeline.py), relaunched if it crashes, until it finishes or $B/STOP exists.
# Afterwards the pod idles so kubectl exec and kit/k8s.sh keep working. MN_AUTORUN=0 skips the pipeline.
KIT=$(cd "$(dirname "$0")" && pwd)
. $KIT/env.sh
mkdir -p $B $OUT
bash $KIT/setup_node.sh > $B/setup_node_$(hostname).log 2>&1
echo "$(date +%T) $(hostname): container packages ready, $(grep -m1 'ib devices' $B/setup_node_$(hostname).log)"
# A lock left behind by a pod that was preempted mid-setup is broken when that pod restarts (same hostname) or after
# 90 minutes; a lock whose owner file is not written yet counts as fresh.
while [ ! -f $B/SETUP_OK ]; do
  if mkdir $B/.setup_lock 2>/dev/null; then
    echo "$(hostname) $(date +%s)" > $B/.setup_lock/owner
    MN_NODE_READY=1 bash $KIT/setup.sh > /dev/null 2>&1
    rm -rf $B/.setup_lock
    [ -f $B/SETUP_OK ] || { echo "setup failed, see $B/setup_$(hostname).log; retrying in 5 minutes"; sleep 300; }
  else
    if read -r owner since < $B/.setup_lock/owner 2>/dev/null; then
      if [ "$owner" = "$(hostname)" ] || [ $(($(date +%s) - since)) -gt 5400 ]; then rm -rf $B/.setup_lock; fi
    fi
    sleep 30
  fi
done
echo "$(date +%T) $(hostname): shared setup ready"
export B OUT HW GATE FLAVOR_k3 FLAVOR_m3 ASSETS_k3 ASSETS_m3 MASTER_PORT NNODES NPROC NODE_RANK MASTER_ADDR K T V
while [ "${MN_AUTORUN:-1}" = 1 ] && [ ! -f $OUT/$HW/pipeline/ALL_DONE ] && [ ! -f $B/STOP ]; do
  $V/bin/python $KIT/pipeline.py >> $B/pipeline_$(hostname).log 2>&1 || { echo "pipeline exited non-zero"; sleep 60; }
done
echo "$(date +%T) $(hostname): idle (pipeline done, stopped, or MN_AUTORUN=0)"
sleep infinity
