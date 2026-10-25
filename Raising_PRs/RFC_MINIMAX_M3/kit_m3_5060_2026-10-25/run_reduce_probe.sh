#!/bin/bash
# Step 1 of fsdp4 and HSDP 2 x 2 again, each on a fresh copy of the HSDP campaign's warm cache, dumping every rank's
# per-micro-batch gradients as they enter FSDP's reduction (M3_DUMP_REDUCE_INPUTS) and the reduced step-1 gradients.
T=/workspace/wt_m3; O=/workspace/kit_m3_5060/hsdp; K=/workspace/kit_m3_5060/kit; P=$O/progress.txt
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
. /workspace/venv_1025/bin/activate
run() {  # <name> <gpus> <spec>
  local name=$1 gpus=$2 spec=$3 D=$O/reduce_probe/$1 C=$O/cc_probe_$1
  rm -rf $D $C; mkdir -p $D; cp -r $O/cache_warm $C
  ( cd $T && env M3_DUMP_GRADS=1 M3_DUMP_REDUCE_INPUTS=1 M3_CELL="$spec" M3_STEPS=1 M3_SEED=$O/seed/step-0 M3_PROBE_DIR=$D \
      PYTHONPATH=/workspace/kit_m3_5060/shim:$K:$T TORCHINDUCTOR_CACHE_DIR=$C/ic TRITON_CACHE_DIR=$C/tc \
      CUDA_VISIBLE_DEVICES=$gpus NCCL_NVLS_ENABLE=0 timeout 3000 \
      torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
      $K/step1_probe.py --module m3_text18_probe --config cell --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; rm -rf $D/out $C
  note "reduce_probe $name rc=$rc wrapped=$(grep -a -c 'foreach_reduce wrapped' $D/run.log) saved=$(grep -a -o 'saved [0-9]* params, calls per param \[[0-9, ]*\]' $D/run.log | sort | uniq -c | tr -s ' ' | tr '\n' ';') step1=$(head -1 $D/loss_rank0.txt 2>/dev/null)"
}
note "# reduce probe, tree $(git -C $T rev-parse --short HEAD) dirty=$(git -C $T status --short | wc -l)"
run fsdp4 0,1,2,3 "dp=4" &
run hsdp2x2 4,5,6,7 "dpr=2,dp=2" &
wait
note REDUCE_PROBE_DONE
