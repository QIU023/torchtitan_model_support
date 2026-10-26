#!/bin/bash
# The 4D layouts that fit on the H100 box's 4 GPUs (user 10-26), after the FSDP2 chain: the FSDP-only matrix at 100 steps
# on c4_test (cells wider than the box are skipped), its 4312-format tables and summary, the EP gradient check and the
# TP forced-selection attribution on the same warm cache.
. "$(dirname "$0")/env_h100.sh"
P=$R/chain_4d_h100.txt; mkdir -p "${R:?}"; note() { echo "$(date +%H:%M:%S) $*" >> $P; }
note "waiting for the FSDP2 chain"
until grep -q "CHAIN_FSDP2_DONE\|SETUP_FAILED" $R/chain_fsdp2.txt 2>/dev/null; do sleep 20; done
note "start, after: $(tail -1 $R/chain_fsdp2.txt)"
RUN_TAG=s100 MEAS_STEPS=100 M3_DATASET=c4_test bash $K/run_m3_4d_h100.sh
O=$R/m4d_s100
. $V/bin/activate
( cd $T5 && python3 $K/tables_4312.py $O auto > $O/table_4312.md 2>&1
  M4D_STEPS=100 M4D_COLS=1,10,50,100 python $K/summarize_m4d.py $O > $O/summary.md 2>&1
  python $K/ep_param_diff.py $O fsdp2:ep2_fsdp2 fsdp4:ep4_fsdp4 > $O/ep_param_diff.txt 2>&1 )
note "matrix: $(tail -1 $O/progress.txt)"
bash $K/run_tp_force_s100.sh
note CHAIN_4D_H100_DONE
