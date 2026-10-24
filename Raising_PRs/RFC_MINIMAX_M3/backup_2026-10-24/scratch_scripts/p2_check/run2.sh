#!/bin/bash
S=/tmp/claude-0/-workspace/e7080317-3a4f-4454-b734-9868986c7b7e/scratchpad/p2_check; . /workspace/venv_1024/bin/activate
P=$S/progress2.txt; rm -f $P
pair() {  # <config> <ngpu> <gpus>
  local cfg=$1 n=$2 gpus=$3
  for tree in base p2; do
    local T=/workspace/tt_main; [ $tree = p2 ] && T=/workspace/wt_m3
    local extra="--export-result=$S/r2_base_$cfg.txt"; [ $tree = p2 ] && extra="--import-result=$S/r2_base_$cfg.txt --assert-equal"
    ( cd $T && PYTHONPATH=$S/cfg CUDA_VISIBLE_DEVICES=$gpus timeout 2400 python scripts/loss_compare.py . . --baseline-module=p2check_configs --baseline-config=$cfg \
        --steps=10 --baseline-ngpus=$n --test-ngpus=$n --job-dump-folder=$S/r2_${tree}_$cfg $extra > $S/r2_${tree}_$cfg.log 2>&1 )
    echo "$tree $cfg rc=$?" >> $P
  done
}
pair qwen35_dense_nocg 1 0
pair qwen35_moe_nocg 8 0,1,2,3,4,5,6,7
echo done >> $P
