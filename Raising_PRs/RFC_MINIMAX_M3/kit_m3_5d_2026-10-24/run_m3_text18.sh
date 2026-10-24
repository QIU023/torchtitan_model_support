#!/bin/bash
# M3 debug model on the K3 text matrix (11 non-CP cells), one shared warm cache, 8 x A4000.
# usage: run_m3_text18.sh <tree> <out dir>
# Seed from a single-GPU build; warm-up (10 steps) of every cell on the shared cache; measured runs (20 steps) on it;
# noise floor: dp1 again on the warm cache, and dp1 on a fresh cache.
T=$1 O=$2
K=$(cd "$(dirname "$0")" && pwd)
mkdir -p $O; P=$O/progress.txt
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
CACHE="TORCHINDUCTOR_CACHE_DIR=$O/cache/ic TRITON_CACHE_DIR=$O/cache/tc"
cell() {  # <phase> <name> <gpus> <spec> [env...]
  local phase=$1 name=$2 gpus=$3 spec=$4; shift 4
  $K/run_cell.sh $T $O/$phase/$name $gpus "$spec" M3_SEED=$O/seed/step-0 "$@" > /dev/null
  note "$phase $name $(cat $O/$phase/$name/result.txt) seed_loaded=$(grep -a -c 'Loading the checkpoint' $O/$phase/$name/run.log)"
}
note "tree $(git -C $T rev-parse --short HEAD) dirty=$(git -C $T status --short | wc -l) torch $(/workspace/venv_1024/bin/python -c 'import torch;print(torch.__version__)')"
rm -rf $O/seed
( . /workspace/venv_1024/bin/activate; cd $T && env M3_SEED_OUT=$O/seed TORCHINDUCTOR_CACHE_DIR=$O/cache/ic TRITON_CACHE_DIR=$O/cache/tc PYTHONPATH=$K:$T CUDA_VISIBLE_DEVICES=0 \
  torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint=localhost:0 -m torchtitan.train --module m3_text18_probe --config seed \
  --output-dir $O/seed_run > $O/seed.log 2>&1 )
ls -d $O/seed/step-0 > /dev/null 2>&1 || { note "no seed, stop"; echo failed > $O/DONE; exit 1; }
note "seed written"
small() {  # <phase> <steps>
  local ph=$1 st=$2
  ( cell $ph dp1 0 "dp=1" M3_STEPS=$st $CACHE; cell $ph fsdp2 0,1 "dp=2" M3_STEPS=$st $CACHE; cell $ph pp2 0,1 "pp=2" M3_STEPS=$st $CACHE ) &
  ( cell $ph tp2_ep2 6,7 "tp=2,ep=2" M3_STEPS=$st $CACHE; cell $ph ep2_fsdp2 6,7 "dp=2,ep=2" M3_STEPS=$st $CACHE ) &
  ( cell $ph pp4 2,3,4,5 "pp=4" M3_STEPS=$st $CACHE; cell $ph tp4_ep4 2,3,4,5 "tp=4,ep=4" M3_STEPS=$st $CACHE ) &
  wait
  for c in "pp8|pp=8" "ep8_fsdp8|dp=8,ep=8" "ep2_fsdp2_tp2_pp2|dp=2,tp=2,ep=2,pp=2" "ep4_fsdp2_tp2_pp2|dp=2,tp=2,ep=4,pp=2"; do
    cell $ph ${c%%|*} 0,1,2,3,4,5,6,7 "${c#*|}" M3_STEPS=$st $CACHE
  done
}
small warm 10
small measured 20
cell noise dp1_rerun 0 "dp=1" M3_STEPS=20 $CACHE
cell noise dp1_fresh_cache 1 "dp=1" M3_STEPS=20 TORCHINDUCTOR_CACHE_DIR=$O/cache_fresh/ic TRITON_CACHE_DIR=$O/cache_fresh/tc
note done; echo done > $O/DONE
