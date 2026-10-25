#!/bin/bash
# The 1-ulp step-1 loss gap between tp2 x ep2 x pp2 and tp2 x ep2: both again with automatic_dynamic_shapes off
# (M3_STATIC_SHAPES=1), one after the other on one copy of the warm cache so both read the same static flex kernels.
T=/workspace/wt_m3_5d; O=/workspace/kit_m3_5060/m4d; K=/workspace/kit_m3_5060/kit; SHIM=/workspace/kit_m3_5060/shim
P=$O/progress.txt; note() { echo "$(date +%H:%M:%S) $*" >> $P; }
. /workspace/venv_1025/bin/activate
C=$O/cc_static; rm -rf $C; cp -r $O/cache $C
run() {  # <name> <gpus> <M3_CELL>
  local name=$1 gpus=$2 spec=$3 D=$O/static/$1 n; n=$(echo $gpus | tr ',' '\n' | wc -l)
  rm -rf $D; mkdir -p $D $D/tmp
  ( cd $T && M3_STATIC_SHAPES=1 M3_RECOMPILE_LIMIT=256 M3_CELL="$spec" M3_STEPS=1 M3_SEED=$O/seed/step-0 M3_PROBE_DIR=$D PYTHONPATH=$SHIM:$K:$T \
      TORCHINDUCTOR_CACHE_DIR=$C/ic TRITON_CACHE_DIR=$C/tc TMPDIR=$D/tmp CUDA_VISIBLE_DEVICES=$gpus NCCL_NVLS_ENABLE=0 \
      timeout 2400 torchrun --nproc_per_node=$n --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
      $K/step1_probe.py --module m3_text18_probe --config cell --output-dir $D/out > $D/run.log 2>&1 < /dev/null )
  local rc=$?; rm -rf $D/out $D/tmp
  note "static $name rc=$rc limit=$(grep -a -c "flex recompiled" $D/run.log) shim=$(grep -a -c m3_5060_shim $D/run.log) seed_loaded=$(grep -a -c 'Loading the checkpoint' $D/run.log) step1=[$(cat $D/loss_rank*.txt | awk '$2 >= 0' | sort -u | tr '\n' ' ')]"
}
note "# static-shape check, retry with every compile at limit 256"
run tp2_ep2 0,1 "tp=2,ep=2"
run tp2_ep2_pp2 0,1,2,3 "tp=2,ep=2,pp=2"
rm -rf $C
note STATIC_DONE
