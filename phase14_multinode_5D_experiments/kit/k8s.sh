#!/bin/bash
# Watch and steer the two phase 14 pods (nebius/p14_pods.yaml or together/p14_pods.yaml) from your own machine;
# kubectl must point at the cluster. The pods run the pipeline by themselves (bootstrap.sh, pipeline.py).
#   k8s.sh status              pods, setup, both heartbeats, the last pipeline decisions, REPORT.txt once finished
#   k8s.sh logs [n]            last n (30) pipeline log lines of both nodes
#   k8s.sh shell <0|1>         interactive shell in pod 0 or 1
#   k8s.sh stop                stop once the current attempt is over (pods stay up)
#   k8s.sh kill                stop now: the current attempt is killed and will be rerun later
#   k8s.sh start               start the pipeline again (after stop or kill)
#   k8s.sh retask              kill, forget the frozen task list (finished tasks stay finished), start; picks up
#                              matrix edits made with sync and a scope set with scope
#   k8s.sh scope [VAR=value ...]   override the manifest's MN_* values (MN_ROWS, MN_MODELS, MN_STAGES, MN_ORDER,
#                              MN_COMM_SWEEP, ...) from the next retask on; no argument clears the override
#   k8s.sh sync                copy this kit directory to the shared filesystem; scripts take effect at the next
#                              attempt, matrix edits after retask
#   k8s.sh collect [dir]       copy the results (no traces, no caches) to dir and summarize every stage
#   k8s.sh comm                communication verdicts (IB, GPUDirect RDMA, raw RDMA per port, busbw, alpha-beta) and
#                              the comparison of the NCCL tuning variants
#   k8s.sh top                 latest GPU (utilization, memory, power, SM clock) and InfiniBand rates of every node
#   k8s.sh report [stage]      per cell of a stage (measure): step time, tps, MFU, peak memory, compute / exposed
#                              communication by process group / idle from the traces, InfiniBand rates
#   k8s.sh trace <stage> <row> [dir]   one cell's ranks merged into one trace in dir, for ui.perfetto.dev
#   k8s.sh tb [stage]          TensorBoard of a stage (measure) on pod 0, forwarded to localhost:6006
#   k8s.sh fetch <path> [dir]  copy a file or directory under runs/ to dir (memory snapshots, raw traces)
#   k8s.sh both '<command>'    one command in both pods at once, detached (manual work; stop the pipeline first)
# Environment: NS (default), JOB (p14), BASE (/mnt/shared/p14, MN_BASE in the manifest), PYTHON (python3).
set -euo pipefail
# Git Bash on Windows rewrites /mnt/... arguments of native programs such as kubectl.exe into Windows paths.
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'
NS=${NS:-default}
JOB=${JOB:-p14}
BASE=${BASE:-/mnt/shared/p14}
PHASE=$BASE/logbook/phase14_multinode_5D_experiments
KIT=$PHASE/kit
RUNS=$BASE/runs
PY=$BASE/venv/bin/python
HERE=$(cd "$(dirname "$0")/.." && pwd)

pods() {
  kubectl get pods -n "$NS" -l "job-name=$JOB" --field-selector=status.phase=Running \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}'
}
pod() {
  kubectl get pods -n "$NS" -l "job-name=$JOB,batch.kubernetes.io/job-completion-index=$1" \
    --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}'
}
any() { pods | head -n 1; }
on() {  # <pod> <command>
  local p=$1
  shift
  kubectl exec -n "$NS" "$p" -- bash -c "$*"
}
detached() {  # <tag> <command>: run in every running pod, detached from kubectl
  local tag=$1 cmd=$2 p
  for p in $(pods); do
    printf '%s\n' "$cmd" | kubectl exec -i -n "$NS" "$p" -- bash -c "mkdir -p $RUNS && cat > $RUNS/cmd_${tag}_\$(hostname).sh"
    on "$p" "setsid nohup bash $RUNS/cmd_${tag}_\$(hostname).sh > $RUNS/out_${tag}_\$(hostname).log 2>&1 < /dev/null &"
    echo "$p: started $tag (output $RUNS/out_${tag}_<hostname>.log)"
  done
}
killall_runs() {
  local p
  on "$(any)" "touch $BASE/STOP"
  # The bracketed patterns keep pkill from matching the bash -c that runs it.
  for p in $(pods); do
    on "$p" "pkill -f 'pipelin[e].py'; pkill -f 'torchtita[n].train|comm_benc[h].py'; true"
  done
}

case "${1:-status}" in
  status)
    kubectl get pods -n "$NS" -l "job-name=$JOB" -o wide
    on "$(any)" "
      [ -f $BASE/SETUP_OK ] && echo 'setup: done' || { echo 'setup: running'; tail -n 3 $BASE/setup_*.log 2>/dev/null; }
      [ -f $BASE/STOP ] && echo 'STOP is set'
      for s in $RUNS/*/pipeline; do
        [ -d \$s ] || continue
        echo \"== \$s: \$(ls \$s/done 2>/dev/null | wc -l) done, \$(ls \$s/giveup 2>/dev/null | wc -l) given up\"
        for n in 0 1; do echo \"node\$n: \$(cat \$s/hb/node\$n 2>/dev/null)\"; done
        tail -n 6 \$s/log_node0.txt 2>/dev/null
        cat \$s/../comm/verdict_node0.txt 2>/dev/null | head -n 3
        [ -f \$s/REPORT.txt ] && cat \$s/REPORT.txt
      done; true" ;;
  logs)
    on "$(any)" "for s in $RUNS/*/pipeline; do for n in 0 1; do echo \"== \$s node\$n\"; tail -n ${2:-30} \$s/log_node\$n.txt 2>/dev/null; done; done; true" ;;
  shell)
    kubectl exec -it -n "$NS" "$(pod "${2:?pod index 0 or 1}")" -- bash ;;
  stop)
    on "$(any)" "touch $BASE/STOP" && echo "STOP set: both pipelines exit once the current attempt is over" ;;
  kill)
    killall_runs
    echo "killed; the pods stay up. k8s.sh start resumes (the killed attempt is rerun)" ;;
  start)
    on "$(any)" "rm -f $BASE/STOP"
    for p in $(pods); do
      if on "$p" "pgrep -f 'pipelin[e].py' > /dev/null"; then echo "$p: pipeline already running"; continue; fi
      printf '%s\n' "bash $KIT/bootstrap.sh" | kubectl exec -i -n "$NS" "$p" -- bash -c "cat > /tmp/p14_start.sh"
      on "$p" "setsid nohup bash /tmp/p14_start.sh > /tmp/p14_start.log 2>&1 < /dev/null &"
      echo "$p: pipeline started"
    done ;;
  retask)
    killall_runs
    echo "waiting 75 s for the pods' bootstrap loops to see STOP"
    sleep 75
    on "$(any)" "rm -f $RUNS/*/pipeline/tasks.json $RUNS/*/pipeline/ALL_DONE $RUNS/*/pipeline/REPORT.txt"
    "$0" start ;;
  sync)
    tar -C "$HERE" -cf - kit | kubectl exec -i -n "$NS" "$(any)" -- tar -C $PHASE -xf -
    # Files written on Windows may end lines in CRLF, which bash and the matrix reader in the pods do not take.
    on "$(any)" "find $KIT -type f \( -name '*.sh' -o -name '*.py' -o -name '*.tsv' \) -exec sed -i 's/\r\$//' {} +"
    echo "kit synced to $KIT" ;;
  collect)
    dest=${2:-./results_$(date +%Y%m%d_%H%M)}
    mkdir -p "$dest"
    kubectl exec -n "$NS" "$(any)" -- tar -C $RUNS -cf - --exclude=tt --exclude=cache . | tar -C "$dest" -xf -
    for stage in "$dest"/*/*/; do
      [ -d "$stage" ] || continue
      ls "$stage"/*/node0.log > /dev/null 2>&1 || continue
      "${PYTHON:-python3}" "$HERE/kit/summarize.py" "$stage" > "$stage/summary.tsv" && echo "== $stage" && cat "$stage/summary.tsv"
    done
    echo "results in $dest" ;;
  scope)
    shift
    if [ $# = 0 ]; then
      on "$(any)" "rm -f $BASE/scope.env"
      echo "scope cleared: the manifest's values apply from the next retask"
    else
      printf 'export %q\n' "$@" | kubectl exec -i -n "$NS" "$(any)" -- bash -c "cat > $BASE/scope.env"
      on "$(any)" "cat $BASE/scope.env"
      echo "takes effect at the next retask"
    fi ;;
  comm)
    on "$(any)" "for h in $RUNS/*/; do [ -d \${h}comm ] || continue; echo \"== \$h\"
      cat \${h}comm/verdict_node*.txt; $PY $KIT/comm_compare.py \$h; done; true" ;;
  top)
    on "$(any)" "$PY $KIT/monitor.py --show --out $RUNS" ;;
  report)
    on "$(any)" "for s in $RUNS/*/${2:-measure}; do [ -d \$s ] && $PY $KIT/trace_report.py \$s; done
      $PY $KIT/monitor.py --summary --out $RUNS > /dev/null; true" ;;
  trace)
    stage=${2:?stage} row=${3:?row} dest=${4:-.}
    cell=$(on "$(any)" "ls -d $RUNS/*/$stage/$row 2>/dev/null | head -n 1")
    [ -n "$cell" ] || { echo "no $stage/$row under $RUNS"; exit 1; }
    on "$(any)" "$PY $KIT/merge_traces.py $cell"
    mkdir -p "$dest"
    kubectl exec -n "$NS" "$(any)" -- cat "$cell/merged_trace.json.gz" > "$dest/${row}_${stage}_trace.json.gz"
    echo "$dest/${row}_${stage}_trace.json.gz: open it in https://ui.perfetto.dev" ;;
  tb)
    p=$(pod 0)
    on "$p" "pkill -f 'tensorboar[d] --logdir'; logdir=\$(ls -d $RUNS/*/${2:-measure} | head -n 1)
      setsid nohup $BASE/venv/bin/tensorboard --logdir \$logdir --host 127.0.0.1 --port 6006 > /tmp/p14_tb.log 2>&1 < /dev/null &"
    sleep 5
    echo "TensorBoard of ${2:-measure} on $p: open http://localhost:6006 (ctrl-c ends the forward)"
    kubectl port-forward -n "$NS" "$p" 6006:6006 ;;
  fetch)
    dest=${3:-.}
    mkdir -p "$dest"
    kubectl exec -n "$NS" "$(any)" -- tar -C $RUNS -cf - "${2:?path under runs/}" | tar -C "$dest" -xf -
    echo "copied $2 to $dest" ;;
  both)
    detached "both_$(date +%H%M%S)" "${2:?command}" ;;
  *)
    awk 'NR > 1 && /^#/ { print; next } NR > 1 { exit }' "$0"
    exit 1 ;;
esac
