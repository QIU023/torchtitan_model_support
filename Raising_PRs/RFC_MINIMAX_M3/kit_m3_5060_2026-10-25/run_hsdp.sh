#!/bin/bash
# MiniMax M3 HSDP check on the 5060 box (venv_1025): fsdp4 against HSDP 2 x 2, both dp4 and so the same samples, on the
# debug model with the 5D matrix's c4_test text cell (m3_text18_probe: seq 512, 4096 tokens per step, AdamW, seed 42,
# deterministic). Every cell loads one single-GPU seed checkpoint; one cache warmed by 10 steps of each layout; 10
# measured steps per cell on a copy of it with full-precision loss / grad norm and the step-1 gradients (step1_probe).
# Noise floor: fsdp4 run twice.
T=/workspace/wt_m3; O=/workspace/kit_m3_5060/hsdp; K=/workspace/kit_m3_5060/kit
mkdir -p $O; P=$O/progress.txt
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
. /workspace/venv_1025/bin/activate
note "tree $(git -C $T rev-parse --short HEAD) dirty=$(git -C $T status --short | wc -l) torch $(python -c 'import torch;print(torch.__version__)')"
rm -rf $O/seed $O/seed_run
( cd $T && M3_SEED_OUT=$O/seed TORCHINDUCTOR_CACHE_DIR=$O/cache_seed/ic TRITON_CACHE_DIR=$O/cache_seed/tc PYTHONPATH=/workspace/kit_m3_5060/shim:$K:$T \
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
    -m torchtitan.train --module m3_text18_probe --config seed --output-dir $O/seed_run > $O/seed.log 2>&1 )
[ -d $O/seed/step-0 ] || { note "no seed, stop"; exit 1; }
rm -rf $O/seed_run $O/cache_seed
note "seed written"
run() {  # <phase> <name> <gpus> <spec> <cache dir> <steps> [env...]
  local ph=$1 name=$2 gpus=$3 spec=$4 C=$5 st=$6; shift 6
  local D=$O/$ph/$name; rm -rf $D; mkdir -p $D
  local n=$(echo $gpus | tr ',' '\n' | wc -l)
  ( cd $T && env "$@" M3_CELL="$spec" M3_STEPS=$st M3_SEED=$O/seed/step-0 M3_PROBE_DIR=$D PYTHONPATH=/workspace/kit_m3_5060/shim:$K:$T \
      TORCHINDUCTOR_CACHE_DIR=$C/ic TRITON_CACHE_DIR=$C/tc CUDA_VISIBLE_DEVICES=$gpus NCCL_NVLS_ENABLE=0 timeout 3000 \
      torchrun --nproc_per_node=$n --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
      $K/step1_probe.py --module m3_text18_probe --config cell --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; rm -rf $D/out
  note "$ph $name rc=$rc shim=$(grep -a -c m3_5060_shim $D/run.log) seed_loaded=$(grep -a -c 'Loading the checkpoint' $D/run.log) mesh=$(grep -a -o 'Building device mesh with parallelism: [^\x1b]*' $D/run.log | head -1 | cut -c40-) last=$(tail -1 $D/loss_rank0.txt 2>/dev/null)"
}
W=$O/cache_warm; rm -rf $W
run warm fsdp4 0,1,2,3 "dp=4" $W 10
run warm hsdp2x2 0,1,2,3 "dpr=2,dp=2" $W 10
for c in fsdp4_a fsdp4_b hsdp2x2; do rm -rf $O/cc_$c; cp -r $W $O/cc_$c; done
run measured fsdp4_a 0,1,2,3 "dp=4" $O/cc_fsdp4_a 10 M3_DUMP_GRADS=1 &
run measured hsdp2x2 4,5,6,7 "dpr=2,dp=2" $O/cc_hsdp2x2 10 M3_DUMP_GRADS=1 &
wait
run measured fsdp4_b 0,1,2,3 "dp=4" $O/cc_fsdp4_b 10 M3_DUMP_GRADS=1
rm -rf $O/cc_fsdp4_a $O/cc_fsdp4_b $O/cc_hsdp2x2
note HSDP_DONE
