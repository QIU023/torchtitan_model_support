#!/bin/bash
# MoE P2 check on ONE warmed cache: warm both trees (10 steps each) on cache C, then measured runs on copies of C:
# base export, P2 vs base, base again vs base (noise floor). 8 GPUs, dp_shard2 x tp2 x pp2, deterministic.
S=/tmp/claude-0/-workspace/e7080317-3a4f-4454-b734-9868986c7b7e/scratchpad/p2_check; . /workspace/venv_1024/bin/activate
P=$S/progress3.txt; rm -f $P; C=$S/r3_cache; rm -rf $C ${C}_*; mkdir -p $C
run() {  # <name> <tree> <cache> <extra args...>
  local name=$1 T=$2 cache=$3; shift 3
  ( cd $T && PYTHONPATH=$S/cfg TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 2400 \
      python scripts/loss_compare.py . . --baseline-module=p2check_configs --baseline-config=qwen35_moe_nocg --steps=10 \
      --baseline-ngpus=8 --test-ngpus=8 --job-dump-folder=$S/r3_$name "$@" > $S/r3_$name.log 2>&1 )
  echo "$name rc=$?" >> $P
}
run warm_base /workspace/tt_main $C --export-result=$S/r3_warm_base.txt
run warm_p2 /workspace/wt_m3 $C --export-result=$S/r3_warm_p2.txt
for i in 1 2 3; do cp -r $C ${C}_$i; done
run base /workspace/tt_main ${C}_1 --export-result=$S/r3_base.txt
run p2 /workspace/wt_m3 ${C}_2 --import-result=$S/r3_base.txt --assert-equal
run base_again /workspace/tt_main ${C}_3 --import-result=$S/r3_base.txt --assert-equal
echo done >> $P
