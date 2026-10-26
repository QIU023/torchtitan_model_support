#!/bin/bash
# MiniMax M3 4D matrix on the H100 or H200 box (kit venv, tree minimax_m3_5d = 89fc094d1): FSDP only (no HSDP cells, user 10-25), CP not run; cells wider than the box are skipped.
# Protocol of the 10-24 kit (run_m3_text18.sh): c4_test text, seq 512, 4096 tokens per step, AdamW, seed 42,
# deterministic; one single-GPU seed checkpoint every cell loads; one shared compile cache that a serial 10-step warm-up
# of every cell fills; measured 20-step runs on copies of that cache, packed onto free GPUs; every stream's reference
# runs twice (noise floor). step1_probe.py records full-precision loss / grad norm per rank and, for non-PP cells,
# rank 0's step-1 gradients.
. "$(dirname "$0")/env_h100.sh"
T=$T5; O=$R/m4d${RUN_TAG:+_$RUN_TAG}
MEAS=${MEAS_STEPS:-20}; export M3_DATASET=${M3_DATASET:-c4_test}
NG=$(nvidia-smi -L | wc -l)
mkdir -p $O; P=$O/progress.txt
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
. $V/bin/activate
# name|gpus|M3_CELL|stream
CELLS="dp1|1|dp=1|dp1
pp2|2|pp=2|dp1
pp2_vpp2|2|pp=2,vpp=2|dp1
pp4|4|pp=4|dp1
pp8|8|pp=8|dp1
tp2_ep2|2|tp=2,ep=2|dp1
tp4_ep4|4|tp=4,ep=4|dp1
tp2_ep2_pp2|4|tp=2,ep=2,pp=2|dp1
tp2_ep2_pp4|8|tp=2,ep=2,pp=4|dp1
tp4_ep4_pp2|8|tp=4,ep=4,pp=2|dp1
fsdp2|2|dp=2|dp2
ep2_fsdp2|2|dp=2,ep=2|dp2
fsdp2_pp2|4|dp=2,pp=2|dp2
ep2_fsdp2_pp2|4|dp=2,ep=2,pp=2|dp2
fsdp2_pp4|8|dp=2,pp=4|dp2
ep2_fsdp2_pp4|8|dp=2,ep=2,pp=4|dp2
fsdp2_tp2_ep2|4|dp=2,tp=2,ep=2|dp2
fsdp2_tp2_ep4|4|dp=2,tp=2,ep=4|dp2
fsdp2_tp4_ep4|8|dp=2,tp=4,ep=4|dp2
fsdp2_tp4_ep8|8|dp=2,tp=4,ep=8|dp2
ep2_fsdp2_tp2_pp2|8|dp=2,tp=2,ep=2,pp=2|dp2
ep4_fsdp2_tp2_pp2|8|dp=2,tp=2,ep=4,pp=2|dp2
fsdp4|4|dp=4|dp4
ep4_fsdp4|4|dp=4,ep=4|dp4
fsdp4_pp2|8|dp=4,pp=2|dp4
ep4_fsdp4_pp2|8|dp=4,ep=4,pp=2|dp4
fsdp4_tp2_ep2|8|dp=4,tp=2,ep=2|dp4
fsdp8|8|dp=8|dp8
ep8_fsdp8|8|dp=8,ep=8|dp8"
echo "$CELLS" > $O/cells.txt

run() {  # <phase> <name> <gpus csv> <M3_CELL> <cache dir> <steps> [env...]
  local ph=$1 name=$2 gpus=$3 spec=$4 C=$5 st=$6; shift 6
  local D=$O/$ph/$name n; n=$(echo $gpus | tr ',' '\n' | wc -l)
  rm -rf $D; mkdir -p $D $D/tmp
  ( cd $T && env "$@" M3_CELL="$spec" M3_STEPS=$st M3_SEED=$O/seed/step-0 M3_PROBE_DIR=$D PYTHONPATH=$SHIM:$K:$T \
      TORCHINDUCTOR_CACHE_DIR=$C/ic TRITON_CACHE_DIR=$C/tc TMPDIR=$D/tmp CUDA_VISIBLE_DEVICES=$gpus \
      timeout 3000 torchrun --nproc_per_node=$n --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
      $K/step1_probe.py --module m3_text18_probe --config cell --output-dir $D/out > $D/run.log 2>&1 < /dev/null )
  local rc=$?; rm -rf $D/out $D/tmp
  local last; last=$(cat $D/loss_rank*.txt 2>/dev/null | awk '$2 >= 0' | sort -n | tail -1)
  note "$ph $name gpus=$gpus rc=$rc shim=$(grep -a -c m3_5060_shim $D/run.log) seed_loaded=$(grep -a -c 'Loading the checkpoint' $D/run.log) mesh=[$(grep -a -o 'Building device mesh with parallelism: [^\x1b]*' $D/run.log | head -1 | cut -c40-)] last=[$last]"
}

pack() {  # <phase> <steps> <cells...> as name|gpus|spec : runs cells of one size side by side on disjoint GPUs
  local ph=$1 st=$2; shift 2
  local size slots i=0 pids=() cell name g spec start gpus
  size=$(echo "$1" | cut -d'|' -f2); slots=$((NG / size))
  for cell in "$@"; do
    IFS='|' read -r name g spec <<< "$cell"
    start=$(( (i % slots) * size )); gpus=$(seq -s, $start $((start + size - 1)))
    local C=$O/cc/$name; rm -rf $C; mkdir -p $O/cc; cp -r $O/cache $C
    local dump=1; [[ $spec == *pp=* ]] && dump=0
    ( run $ph $name $gpus "$spec" $C $st M3_DUMP_GRADS=$dump; rm -rf $C ) &
    pids+=($!); i=$((i + 1))
    if [ $((i % slots)) -eq 0 ]; then wait "${pids[@]}"; pids=(); fi
  done
  wait "${pids[@]}"
}

note "tree $(git -C $T rev-parse --short HEAD) dirty=$(git -C $T status --short | wc -l) torch $(python -c 'import torch;print(torch.__version__)') gpus $NG dataset $M3_DATASET measured_steps $MEAS"
rm -rf $O/seed $O/seed_run $O/cache
( cd $T && M3_SEED_OUT=$O/seed TORCHINDUCTOR_CACHE_DIR=$O/cache_seed/ic TRITON_CACHE_DIR=$O/cache_seed/tc PYTHONPATH=$SHIM:$K:$T \
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
    -m torchtitan.train --module m3_text18_probe --config seed --output-dir $O/seed_run > $O/seed.log 2>&1 < /dev/null )
[ -d $O/seed/step-0 ] || { note "no seed, stop"; exit 1; }
rm -rf $O/seed_run $O/cache_seed
note "seed written"

# Warm-up: every cell 10 steps, one after another, on the one shared cache.
while IFS='|' read -r name g spec stream; do
  [ "$g" -le "$NG" ] || { note "skip $name: needs $g GPUs, box has $NG"; continue; }
  run warm $name $(seq -s, 0 $((g - 1))) "$spec" $O/cache 10
done < $O/cells.txt
note "warm-up done, cache $(du -sh $O/cache | cut -f1)"

# Measured: 20 steps on copies of the warm cache; each stream's reference twice.
for size in 1 2 4 8; do
  [ $size -le $NG ] || continue
  list=()
  while IFS='|' read -r name g spec stream; do
    [ "$g" = "$size" ] || continue
    list+=("$name|$g|$spec")
    case $name in dp1|fsdp2|fsdp4|fsdp8) list+=("${name}_again|$g|$spec") ;; esac
  done < $O/cells.txt
  [ ${#list[@]} -gt 0 ] && pack measured $MEAS "${list[@]}"
done
note M4D_DONE
