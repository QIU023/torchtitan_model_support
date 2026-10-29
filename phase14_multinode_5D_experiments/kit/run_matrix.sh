#!/bin/bash
# Every row of one model's matrix on this hardware, for one stage. Run the same command on every node (NODE_RANK and
# MASTER_ADDR set), or under srun with one task per node.
# Usage: run_matrix.sh <k3|m3> <smoke|probe|warm|measure> [row ...]
#   smoke    debug flavor, test tokenizer, c4_test, short sequence, 5 steps: every layout starts and steps
#   probe    downscale flavor, c4_test, 3 steps at the row's length: per-rank memory (rows that OOM get shorter)
#   warm     downscale flavor, c4, 10 steps of every row on the one compile cache the measured runs share
#   measure  downscale flavor, c4, MEAS_STEPS (30) steps, profiler trace every PROFILE_FREQ (25) steps
. "$(dirname "$0")/env.sh"
set -u
MODEL=$1
STAGE=$2
shift 2
ONLY=" $* "
NODE_RANK=${NODE_RANK:-${SLURM_NODEID:-0}}
FLAVOR_VAR=FLAVOR_$MODEL
ASSETS_VAR=ASSETS_$MODEL
P=$OUT/$HW/$STAGE
mkdir -p $P
note() { echo "$(date +%T) $*" | tee -a $P/progress_node$NODE_RANK.txt; }

while IFS=$'\t' read -r name cell s_h100 s_h200 s_a100 extra; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  [[ "$ONLY" != "  " && "$ONLY" != *" $name "* ]] && continue
  case $HW in h100) seq=$s_h100 ;; h200) seq=$s_h200 ;; *) seq=$s_a100 ;; esac
  envs=(MN_MODEL=$MODEL MN_CELL=$cell MN_GATE=$GATE MN_JOB=p14_${HW}_${MODEL}_${STAGE}_$name)
  skip=0
  for item in ${extra:-}; do
    case $item in
      nosmoke) [ $STAGE = smoke ] && skip=1 ;;
      *:*) [ "${item%%:*}" = $HW ] && envs+=("${item#*:}") ;;
      *) envs+=("$item") ;;
    esac
  done
  if [ $skip = 1 ]; then note "skip $name (debug model cannot take this layout)"; continue; fi
  pp=$(echo "$cell" | grep -o 'pp=[0-9]*' | cut -d= -f2)
  pp=${pp:-1}
  case $STAGE in
    smoke)
      envs+=(MN_FLAVOR=debugmodel MN_SEQ=$([ $MODEL = k3 ] && echo 4096 || echo 2048)
             MN_MICROBATCHES=$((pp > 1 ? 2 * pp : 1)) MN_STEPS=5 MN_DATASET=c4_test
             MN_CACHE=$OUT/cache/$HW/smoke_$MODEL) ;;
    probe)
      envs+=(MN_FLAVOR=${!FLAVOR_VAR} MN_ASSETS=${!ASSETS_VAR} MN_SEQ=$seq MN_STEPS=3 MN_DATASET=c4_test
             MN_CACHE=$OUT/cache/$HW/probe_$MODEL) ;;
    warm)
      envs+=(MN_FLAVOR=${!FLAVOR_VAR} MN_ASSETS=${!ASSETS_VAR} MN_SEQ=$seq MN_STEPS=10 MN_DATASET=c4
             MN_CACHE=$OUT/cache/$HW/$MODEL) ;;
    measure)
      envs+=(MN_FLAVOR=${!FLAVOR_VAR} MN_ASSETS=${!ASSETS_VAR} MN_SEQ=$seq MN_STEPS=${MEAS_STEPS:-30}
             MN_PROFILE=${PROFILE_FREQ:-25} MN_DATASET=c4 MN_CACHE=$OUT/cache/$HW/$MODEL) ;;
    *) echo "stage must be smoke, probe, warm or measure" >&2; exit 1 ;;
  esac
  [ -n "${MN_FLAVOR_OVERRIDE:-}" ] && [ $STAGE != smoke ] && envs+=(MN_FLAVOR=$MN_FLAVOR_OVERRIDE)
  note "start $name cell=$cell seq=$seq ${envs[*]}"
  env "${envs[@]}" bash $K/launch.sh $P/$name
  note "end $name rc=$?"
done < $K/matrix_$MODEL.tsv
python $K/summarize.py $P > $P/summary_node$NODE_RANK.tsv 2>&1 || true
