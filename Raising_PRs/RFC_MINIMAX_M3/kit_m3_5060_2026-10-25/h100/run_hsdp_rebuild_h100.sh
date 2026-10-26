#!/bin/bash
# Locate the HSDP vs fsdp4 step-1 gradient gap on this box: rerun both 1 step on copies of the dp4 table's warm cache with
# every rank's per-micro-batch gradients dumped at FSDP's reduction (M3_DUMP_REDUCE_INPUTS), then rebuild both layouts'
# reductions on CPU (reduce_emulate.py) and check the probe left the step-1 gradients unchanged.
. "$(dirname "$0")/env_h100.sh"
O=$R/hsdp_s100; D=$O/rebuild; mkdir -p "${D:?}"; P=$O/progress.txt; note() { echo "$(date +%H:%M:%S) $*" >> $P; }
. $V/bin/activate
one() {  # <name> <spec>
  local name=$1 spec=$2 C=$O/cc_rebuild_$1
  rm -rf "${D:?}/${name:?}" "${C:?}"; mkdir -p $D/$name/tmp; cp -r $O/cache $C
  ( cd $TB && M3_DUMP_REDUCE_INPUTS=1 M3_DUMP_GRADS=1 M3_CELL="$spec" M3_STEPS=1 M3_SEED=$O/seed/step-0 M3_PROBE_DIR=$D/$name \
      PYTHONPATH=$SHIM:$K:$TB TORCHINDUCTOR_CACHE_DIR=$C/ic TRITON_CACHE_DIR=$C/tc TMPDIR=$D/$name/tmp CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 2400 \
      torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
      $K/step1_probe.py --module m3_text18_probe --config cell --output-dir $D/$name/out > $D/$name/run.log 2>&1 < /dev/null )
  local rc=$?
  rm -rf "${D:?}/${name:?}/out" "${D:?}/${name:?}/tmp" "${C:?}"
  note "rebuild $name rc=$rc wrapped=$(grep -a -c 'foreach_reduce wrapped' $D/$name/run.log) step1=[$(head -1 $D/$name/loss_rank0.txt 2>/dev/null)]"
}
one fsdp4 "dp=4"
one hsdp2x2 "dpr=2,dp=2"
( cd $TB && python $K/compare_grads.py $O/measured/fsdp4/grads.pt $D/fsdp4/grads.pt > $D/probe_vs_measured_fsdp4.txt 2>&1
  python $K/compare_grads.py $O/measured/hsdp2x2/grads.pt $D/hsdp2x2/grads.pt > $D/probe_vs_measured_hsdp2x2.txt 2>&1
  python -I $K/reduce_emulate.py $D > $D/reduce_emulate.txt 2>&1 )
note "REBUILD_DONE"
