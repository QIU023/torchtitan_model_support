#!/bin/bash
# 100-step rerun on the H200 in PR 4312's protocol (user 10-26): c4_test text, the baseline's dp4 HSDP table and the
# FSDP-only 4D cells, 100 measured steps, tables at steps 1 / 10 / 50 / 100; starts once the 20-step chain has ended.
. "$(dirname "$0")/env_h100.sh"
P=$R/chain_s100.txt; note() { echo "$(date +%H:%M:%S) $*" >> $P; }
until grep -q "CHAIN_DONE\|CHAIN_INCOMPLETE\|SETUP_FAILED" $R/chain.txt 2>/dev/null; do sleep 20; done
note "100-step chain start, after: $(tail -1 $R/chain.txt)"
export M3_DATASET=c4_test MEAS_STEPS=100 RUN_TAG=s100
bash $K/run_hsdp_h100.sh
note "hsdp_s100: $(tail -1 $R/hsdp_s100/progress.txt)"
bash $K/run_m3_4d_h100.sh
note "m4d_s100: $(tail -1 $R/m4d_s100/progress.txt)"
. $V/bin/activate
( cd $T5 && M4D_STEPS=100 M4D_COLS=1,10,50,100 python $K/summarize_m4d.py $R/m4d_s100 > $R/m4d_s100/summary.md 2>&1
  M4D_COLS=1,10,50,100 python $K/tables_m4d.py $R/m4d_s100 > $R/m4d_s100/tables_en.md 2>&1 )
if grep -q HSDP_DONE $R/hsdp_s100/progress.txt && grep -q M4D_DONE $R/m4d_s100/progress.txt && [ -s $R/m4d_s100/summary.md ]; then note CHAIN_S100_DONE; else note CHAIN_S100_INCOMPLETE; fi
