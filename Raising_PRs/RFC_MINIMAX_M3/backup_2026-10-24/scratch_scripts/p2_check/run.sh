#!/bin/bash
# P2 bitwise check: Qwen3.5 debug recipes, 10 steps, deterministic, base tree (upstream main 3ed5aa97) exported, P1+P2 tree compared with --assert-equal.
S=/tmp/claude-0/-workspace/e7080317-3a4f-4454-b734-9868986c7b7e/scratchpad/p2_check; . /workspace/venv_1024/bin/activate
M=torchtitan_recipes.tests.models.qwen3_5
pair() {  # <config> <gpu>
  local cfg=$1 gpu=$2
  ( cd /workspace/tt_main && CUDA_VISIBLE_DEVICES=$gpu timeout 2400 python scripts/loss_compare.py . . --baseline-module=$M --baseline-config=$cfg \
      --steps=10 --baseline-ngpus=1 --test-ngpus=1 --job-dump-folder=$S/base_$cfg --export-result=$S/base_$cfg.txt > $S/base_$cfg.log 2>&1 )
  echo "base $cfg rc=$?" >> $S/progress.txt
  ( cd /workspace/wt_m3 && CUDA_VISIBLE_DEVICES=$gpu timeout 2400 python scripts/loss_compare.py . . --baseline-module=$M --baseline-config=$cfg \
      --steps=10 --baseline-ngpus=1 --test-ngpus=1 --job-dump-folder=$S/p2_$cfg --import-result=$S/base_$cfg.txt --assert-equal > $S/p2_$cfg.log 2>&1 )
  echo "p2 $cfg assert-equal rc=$?" >> $S/progress.txt
}
( pair qwen35_debugmodel 0 ) &
( pair qwen35_debugmodel_moe 1 ) &
( cd /workspace/wt_m3 && CUDA_VISIBLE_DEVICES=2 timeout 2400 python -m pytest -q -p no:cacheprovider tests/unit_tests/gpu/models/qwen3_5/test_rope_compile.py > $S/rope_compile.log 2>&1; echo "rope_compile rc=$? $(tail -1 $S/rope_compile.log)" >> $S/progress.txt ) &
wait
echo done >> $S/progress.txt
