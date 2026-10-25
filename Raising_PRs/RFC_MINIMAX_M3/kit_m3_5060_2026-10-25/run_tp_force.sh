#!/bin/bash
# TP attribution on the 5060 (the 10-24 forced experiment): record the single GPU's step-1 block selections and expert
# choices, then impose them on tp2 x ep2 and tp4 x ep4; every run is 1 step on a copy of the 4D matrix's warm cache.
T=/workspace/wt_m3_5d; O=/workspace/kit_m3_5060/m4d; K=/workspace/kit_m3_5060/kit; SHIM=/workspace/kit_m3_5060/shim
P=$O/progress.txt; note() { echo "$(date +%H:%M:%S) $*" >> $P; }
. /workspace/venv_1025/bin/activate
run() {  # <name> <gpus> <M3_CELL> [env...]
  local name=$1 gpus=$2 spec=$3; shift 3
  local D=$O/tpforce/$name C=$O/cc_tpforce_$name n; n=$(echo $gpus | tr ',' '\n' | wc -l)
  rm -rf $D $C; mkdir -p $D $D/tmp; cp -r $O/cache $C
  ( cd $T && env "$@" M3_CELL="$spec" M3_STEPS=1 M3_SEED=$O/seed/step-0 M3_PROBE_DIR=$D M3_DUMP_GRADS=1 PYTHONPATH=$SHIM:$K:$T \
      TORCHINDUCTOR_CACHE_DIR=$C/ic TRITON_CACHE_DIR=$C/tc TMPDIR=$D/tmp CUDA_VISIBLE_DEVICES=$gpus NCCL_NVLS_ENABLE=0 \
      timeout 2400 torchrun --nproc_per_node=$n --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
      $K/step1_probe.py --module m3_text18_probe --config cell --output-dir $D/out > $D/run.log 2>&1 < /dev/null )
  local rc=$?; rm -rf $D/out $D/tmp $C
  note "tpforce $name rc=$rc shim=$(grep -a -c m3_5060_shim $D/run.log) seed_loaded=$(grep -a -c 'Loading the checkpoint' $D/run.log) files=[$(ls $D | grep -v run.log | tr '\n' ' ')] step1=[$(cat $D/loss_rank0.txt 2>/dev/null)]"
}
note "# TP forced selections and routing"
run dp1_record 0 "dp=1" M3_HOOK_SELECT=1 M3_HOOK_SELECT_ALL=1 M3_HOOK_ROUTE_ALL=1
[ -f $O/tpforce/dp1_record/selections_all.pt ] && [ -f $O/tpforce/dp1_record/routes_all.pt ] || { note "no recordings, stop"; exit 1; }
R=$O/tpforce/dp1_record
run tp2_ep2_forced 0,1 "tp=2,ep=2" M3_FORCE_SELECT=$R/selections_all.pt M3_FORCE_ROUTE=$R/routes_all.pt &
run tp4_ep4_forced 2,3,4,5 "tp=4,ep=4" M3_FORCE_SELECT=$R/selections_all.pt M3_FORCE_ROUTE=$R/routes_all.pt &
wait
note TPFORCE_DONE
