#!/bin/bash
# Drive the two phase 14 pods (p14_pods.yaml) from your own machine; kubectl must point at the Nebius cluster.
#   k8s.sh status                      pods and nodes, SETUP_OK, last progress lines of both pods
#   k8s.sh shell <0|1>                 interactive shell in pod 0 or 1
#   k8s.sh both '<command>'            the same command in both pods at once, detached (survives kubectl disconnects)
#   k8s.sh comm                        kit/run_comm.sh in both pods: IB / GPUDirect RDMA verdict and busbw
#   k8s.sh run <k3|m3> <stage> [row..] kit/run_matrix.sh in both pods (stage: smoke | probe | warm | measure)
#   k8s.sh logs [n]                    last n (40) lines of the latest detached command's output in both pods
#   k8s.sh sync                        copy this kit directory into both pods (after editing matrices or scripts)
#   k8s.sh collect [dir]               copy both pods' results (no traces, no caches) to dir and summarize every stage
# Environment: NS (default), JOB (p14), HW (overrides the pods' HW for both and run), PYTHON (python3).
set -euo pipefail
NS=${NS:-default}
JOB=${JOB:-p14}
PHASE_POD=/root/logbook/phase14_multinode_5D_experiments
KIT_POD=$PHASE_POD/kit
RUNS_POD=/root/p14/runs
HERE=$(cd "$(dirname "$0")/../.." && pwd)

pod() {
  kubectl get pods -n "$NS" -l "job-name=$JOB,batch.kubernetes.io/job-completion-index=$1" \
    -o jsonpath='{.items[0].metadata.name}'
}
inpod() {
  local p
  p=$(pod "$1")
  shift
  kubectl exec -n "$NS" "$p" -- bash -lc "$*"
}
detached() {  # <tag> <command>: write the command to a file in each pod, then start it with setsid in both
  local tag=$1 cmd=$2 idx p
  [ -n "${HW:-}" ] && cmd="export HW=$HW; $cmd"
  for idx in 0 1; do
    p=$(pod $idx)
    printf '%s\n' "$cmd" | kubectl exec -i -n "$NS" "$p" -- bash -c "mkdir -p $RUNS_POD && cat > $RUNS_POD/cmd_$tag.sh"
    kubectl exec -n "$NS" "$p" -- bash -lc \
      "echo $tag > $RUNS_POD/LATEST; setsid nohup bash -l $RUNS_POD/cmd_$tag.sh > $RUNS_POD/out_${tag}_node$idx.log 2>&1 < /dev/null &"
    echo "pod $idx ($p): started $tag, output $RUNS_POD/out_${tag}_node$idx.log"
  done
}

case "${1:-status}" in
  status)
    kubectl get pods -n "$NS" -l "job-name=$JOB" -o wide
    for idx in 0 1; do
      echo "== pod $idx"
      inpod $idx "ls /root/p14/SETUP_OK 2>/dev/null || echo 'setup not finished (tail /root/p14/setup_*.log)'; \
        tail -n 3 $RUNS_POD/*/*/progress_node*.txt 2>/dev/null; cat $RUNS_POD/*/comm/verdict_node*.txt 2>/dev/null" || true
    done ;;
  shell)
    kubectl exec -it -n "$NS" "$(pod "${2:?pod index 0 or 1}")" -- bash -l ;;
  both)
    detached "both_$(date +%H%M%S)" "${2:?command}" ;;
  comm)
    detached comm "bash $KIT_POD/run_comm.sh" ;;
  run)
    model=${2:?k3 or m3}
    stage=${3:?smoke, probe, warm or measure}
    shift 3
    detached "${model}_${stage}" "bash $KIT_POD/run_matrix.sh $model $stage $*" ;;
  logs)
    for idx in 0 1; do
      echo "== pod $idx"
      inpod $idx "tail -n ${2:-40} $RUNS_POD/out_\$(cat $RUNS_POD/LATEST)_node$idx.log" || true
    done ;;
  sync)
    for idx in 0 1; do
      tar -C "$HERE" -cf - kit | kubectl exec -i -n "$NS" "$(pod $idx)" -- tar -C $PHASE_POD -xf -
      echo "pod $idx: kit synced"
    done ;;
  collect)
    dest=${2:-./results_nebius_$(date +%Y%m%d_%H%M)}
    mkdir -p "$dest"
    for idx in 0 1; do
      kubectl exec -n "$NS" "$(pod $idx)" -- tar -C $RUNS_POD -cf - --exclude=tt --exclude=cache . | tar -C "$dest" -xf -
    done
    for stage in "$dest"/*/*/; do
      [ -d "$stage" ] || continue
      ls "$stage"/*/node0.log >/dev/null 2>&1 || continue
      "${PYTHON:-python3}" "$HERE/kit/summarize.py" "$stage" > "$stage/summary.tsv" && echo "== $stage" && cat "$stage/summary.tsv"
    done
    echo "results in $dest" ;;
  *)
    sed -n '2,12p' "$0"; exit 1 ;;
esac
