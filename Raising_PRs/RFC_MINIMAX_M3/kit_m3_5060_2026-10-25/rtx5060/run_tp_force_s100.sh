#!/bin/bash
# TP attribution on the 100-step matrix's warm cache: record the single GPU's step-1 block selections and expert choices,
# then impose them on tp2 x ep2 and tp4 x ep4 (1 step each on cache copies) and compare step-1 gradients.
. "$(dirname "$0")/env_h100.sh"
O=$R/m4d_s100; D=$O/tpforce; mkdir -p "${D:?}"; P=$O/progress.txt; note() { echo "$(date +%H:%M:%S) $*" >> $P; }
. $V/bin/activate
run() {  # <name> <gpus> <spec> [env...]
  local name=$1 gpus=$2 spec=$3; shift 3
  local C=$O/cc_tpforce_$name n; n=$(echo $gpus | tr ',' '\n' | wc -l)
  rm -rf "${D:?}/${name:?}" "${C:?}"; mkdir -p $D/$name/tmp; cp -r $O/cache $C
  ( cd $T5 && env "$@" M3_CELL="$spec" M3_STEPS=1 M3_SEED=$O/seed/step-0 M3_PROBE_DIR=$D/$name M3_DUMP_GRADS=1 PYTHONPATH=$SHIM:$K:$T5 \
      TORCHINDUCTOR_CACHE_DIR=$C/ic TRITON_CACHE_DIR=$C/tc TMPDIR=$D/$name/tmp CUDA_VISIBLE_DEVICES=$gpus timeout 2400 \
      torchrun --nproc_per_node=$n --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
      $K/step1_probe.py --module m3_text18_probe --config cell --output-dir $D/$name/out > $D/$name/run.log 2>&1 < /dev/null )
  local rc=$?
  rm -rf "${D:?}/${name:?}/out" "${D:?}/${name:?}/tmp" "${C:?}"
  note "tpforce $name rc=$rc step1=[$(head -1 $D/$name/loss_rank0.txt 2>/dev/null)]"
}
run dp1_record 0 "dp=1" M3_HOOK_SELECT=1 M3_HOOK_SELECT_ALL=1 M3_HOOK_ROUTE_ALL=1
REC=$D/dp1_record
if [ "$(nvidia-smi -L | wc -l)" -ge 6 ]; then
  run tp2_ep2_forced 0,1 "tp=2,ep=2" M3_FORCE_SELECT=$REC/selections_all.pt M3_FORCE_ROUTE=$REC/routes_all.pt &
  run tp4_ep4_forced 2,3,4,5 "tp=4,ep=4" M3_FORCE_SELECT=$REC/selections_all.pt M3_FORCE_ROUTE=$REC/routes_all.pt &
  wait
else
  run tp2_ep2_forced 0,1 "tp=2,ep=2" M3_FORCE_SELECT=$REC/selections_all.pt M3_FORCE_ROUTE=$REC/routes_all.pt
  run tp4_ep4_forced 0,1,2,3 "tp=4,ep=4" M3_FORCE_SELECT=$REC/selections_all.pt M3_FORCE_ROUTE=$REC/routes_all.pt
fi
( cd $T5
  python $K/compare_grads.py $O/measured/dp1/grads.pt $REC/grads.pt > $D/record_vs_measured_dp1.txt 2>&1
  for c in tp2_ep2 tp4_ep4; do
    python $K/compare_grads.py $O/measured/dp1/grads.pt $O/measured/$c/grads.pt > $D/${c}_unforced.txt 2>&1
    python $K/compare_grads.py $O/measured/dp1/grads.pt $D/${c}_forced/grads.pt > $D/${c}_forced.txt 2>&1
  done )
note "TPFORCE_DONE"
