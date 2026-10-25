#!/bin/bash
# The 1-ulp step-1 loss gap between tp2 x ep2 x pp2 and tp2 x ep2: record every micro-batch loss of step 1 in both
# (M3_DUMP_MB_LOSSES), default settings as in the matrix, one after the other on one copy of the warm cache.
T=/workspace/wt_m3_5d; O=/workspace/kit_m3_5060/m4d; K=/workspace/kit_m3_5060/kit; SHIM=/workspace/kit_m3_5060/shim
P=$O/progress.txt; note() { echo "$(date +%H:%M:%S) $*" >> $P; }
. /workspace/venv_1025/bin/activate
C=$O/cc_mbloss; rm -rf $C; cp -r $O/cache $C
run() {  # <name> <gpus> <M3_CELL>
  local name=$1 gpus=$2 spec=$3 D=$O/mbloss/$1 n; n=$(echo $gpus | tr ',' '\n' | wc -l)
  rm -rf $D; mkdir -p $D $D/tmp
  ( cd $T && M3_DUMP_MB_LOSSES=1 M3_CELL="$spec" M3_STEPS=1 M3_SEED=$O/seed/step-0 M3_PROBE_DIR=$D PYTHONPATH=$SHIM:$K:$T \
      TORCHINDUCTOR_CACHE_DIR=$C/ic TRITON_CACHE_DIR=$C/tc TMPDIR=$D/tmp CUDA_VISIBLE_DEVICES=$gpus NCCL_NVLS_ENABLE=0 \
      timeout 2400 torchrun --nproc_per_node=$n --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
      $K/step1_probe.py --module m3_text18_probe --config cell --output-dir $D/out > $D/run.log 2>&1 < /dev/null )
  local rc=$?; rm -rf $D/out $D/tmp
  note "mbloss $name rc=$rc armed=$(grep -a -c m3_mb_loss_probe $D/run.log) files=[$(ls $D | grep mb_losses | tr '\n' ' ')] step1=[$(cat $D/loss_rank*.txt | awk '$2 >= 0' | sort -u | tr '\n' ' ')]"
}
note "# per-micro-batch losses for the tp2 x ep2 PP 1-ulp gap"
run tp2_ep2 0,1 "tp=2,ep=2"
run tp2_ep2_pp2 0,1,2,3 "tp=2,ep=2,pp=2"
rm -rf $C
note MBLOSS_DONE
