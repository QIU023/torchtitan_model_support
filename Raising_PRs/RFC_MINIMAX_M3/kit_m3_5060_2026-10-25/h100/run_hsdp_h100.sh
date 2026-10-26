#!/bin/bash
# Baseline body's dp4 table on the H100 box: fsdp4 (twice, noise floor) against HSDP 2 x 2 on the baseline tree, same
# protocol as the 5060 matrix: one single-GPU seed, one cache warmed by 10 steps of each layout, 20 measured steps on
# copies of it, full-precision loss / grad norm and rank 0's step-1 gradients (step1_probe.py), FA selection off (shim).
. "$(dirname "$0")/env_h100.sh"
T=$TB; O=$R/hsdp${RUN_TAG:+_$RUN_TAG}; mkdir -p $O; P=$O/progress.txt
MEAS=${MEAS_STEPS:-20}; export M3_DATASET=${M3_DATASET:-c4_test}
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
. $V/bin/activate
NG=$(nvidia-smi -L | wc -l)
note "tree $(git -C $T rev-parse --short HEAD) dirty=$(git -C $T status --short | wc -l) torch $(python -c 'import torch;print(torch.__version__)') gpus $NG dataset $M3_DATASET measured_steps $MEAS"
rm -rf $O/seed $O/seed_run $O/cache
( cd $T && M3_SEED_OUT=$O/seed TORCHINDUCTOR_CACHE_DIR=$O/cache_seed/ic TRITON_CACHE_DIR=$O/cache_seed/tc PYTHONPATH=$SHIM:$K:$T \
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
    -m torchtitan.train --module m3_text18_probe --config seed --output-dir $O/seed_run > $O/seed.log 2>&1 < /dev/null )
[ -d $O/seed/step-0 ] || { note "no seed, stop"; exit 1; }
rm -rf $O/seed_run $O/cache_seed
note "seed written"
run() {  # <phase> <name> <gpus> <spec> <cache dir> <steps> [env...]
  local ph=$1 name=$2 gpus=$3 spec=$4 C=$5 st=$6; shift 6
  local D=$O/$ph/$name n; n=$(echo $gpus | tr ',' '\n' | wc -l)
  rm -rf $D; mkdir -p $D $D/tmp
  ( cd $T && env "$@" M3_CELL="$spec" M3_STEPS=$st M3_SEED=$O/seed/step-0 M3_PROBE_DIR=$D PYTHONPATH=$SHIM:$K:$T \
      TORCHINDUCTOR_CACHE_DIR=$C/ic TRITON_CACHE_DIR=$C/tc TMPDIR=$D/tmp CUDA_VISIBLE_DEVICES=$gpus timeout 3000 \
      torchrun --nproc_per_node=$n --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
      $K/step1_probe.py --module m3_text18_probe --config cell --output-dir $D/out > $D/run.log 2>&1 < /dev/null )
  local rc=$?; rm -rf $D/out $D/tmp
  note "$ph $name gpus=$gpus rc=$rc shim=$(grep -a -c m3_5060_shim $D/run.log) seed_loaded=$(grep -a -c 'Loading the checkpoint' $D/run.log) mesh=[$(grep -a -o 'Building device mesh with parallelism: [^\x1b]*' $D/run.log | head -1 | cut -c40-)] last=[$(cat $D/loss_rank0.txt 2>/dev/null | tail -1)]"
}
run warm fsdp4 0,1,2,3 "dp=4" $O/cache 10
run warm hsdp2x2 0,1,2,3 "dpr=2,dp=2" $O/cache 10
for c in fsdp4 fsdp4_again hsdp2x2; do rm -rf $O/cc_$c; cp -r $O/cache $O/cc_$c; done
if [ $NG -ge 8 ]; then
  run measured fsdp4 0,1,2,3 "dp=4" $O/cc_fsdp4 $MEAS M3_DUMP_GRADS=1 &
  run measured hsdp2x2 4,5,6,7 "dpr=2,dp=2" $O/cc_hsdp2x2 $MEAS M3_DUMP_GRADS=1 &
  wait
else
  run measured fsdp4 0,1,2,3 "dp=4" $O/cc_fsdp4 $MEAS M3_DUMP_GRADS=1
  run measured hsdp2x2 0,1,2,3 "dpr=2,dp=2" $O/cc_hsdp2x2 $MEAS M3_DUMP_GRADS=1
fi
run measured fsdp4_again 0,1,2,3 "dp=4" $O/cc_fsdp4_again $MEAS M3_DUMP_GRADS=1
rm -rf $O/cc_fsdp4 $O/cc_fsdp4_again $O/cc_hsdp2x2
( cd $T && python $K/compare_grads.py $O/measured/fsdp4/grads.pt $O/measured/fsdp4_again/grads.pt > $O/grads_fsdp4_twice.txt 2>&1
  python $K/compare_grads.py $O/measured/fsdp4/grads.pt $O/measured/hsdp2x2/grads.pt > $O/grads_fsdp4_vs_hsdp2x2.txt 2>&1 )
[ -s $O/measured/hsdp2x2/loss_rank0.txt ] && [ -s $O/measured/fsdp4_again/loss_rank0.txt ] && note HSDP_DONE || note "HSDP_FAILED"
