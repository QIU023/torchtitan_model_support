#!/bin/bash
# CI's CPU suite after the rebase onto e111f10d4, one tree after another (the suites bind the fixed ports 29511 and
# 12356): new main, the baseline, the 5D branch; then each branch against main test by test, and the outcomes of
# main's new test_rope_cache_tying.py on all three trees.
K=/workspace/kit_m3_5060; O=$K/cpu_full; P=$K/runs5060/rebase_e111/cpu_progress.txt; mkdir -p "$(dirname $P)"
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
for t in /workspace/tt_main_e111:main_e111f10d4 /workspace/wt_m3:minimax_m3_a7db544ff /workspace/wt_m3_5d:minimax_m3_5d_43ccfadb5; do
  note "start ${t#*:} at $(git -C ${t%%:*} rev-parse --short HEAD) dirty=$(git -C ${t%%:*} status --short | wc -l)"
  $K/run_cpu_full.sh ${t%%:*} ${t#*:}
  note "${t#*:} $(cat $O/${t#*:}.done)"
done
for b in minimax_m3_a7db544ff minimax_m3_5d_43ccfadb5; do
  /workspace/venv_1025/bin/python $K/kit/junit_compare.py $O/main_e111f10d4.xml $O/$b.xml tests.unit_tests.cpu.models.minimax_m3 MiniMaxM3 test_rope_cache_tying > $O/compare_$b.txt 2>&1
  note "$b vs main: $(sed -n 2p $O/compare_$b.txt); $(grep "matching 'test_rope_cache_tying'" $O/compare_$b.txt)"
done
/workspace/venv_1025/bin/python $K/kit/junit_compare.py $O/main_e111f10d4.xml $O/main_e111f10d4.xml test_rope_cache_tying > $O/compare_main_self.txt 2>&1
note "main: $(grep "matching 'test_rope_cache_tying'" $O/compare_main_self.txt)"
note CPU_REBASE_DONE
