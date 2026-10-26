#!/bin/bash
# The whole H100 job: setup (once), the baseline's dp4 table, the 5D 4D matrix, then the summaries as text on the box.
. "$(dirname "$0")/env_h100.sh"
mkdir -p $R; P=$R/chain.txt
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
note "chain start"
[ -f $B/SETUP_OK ] || bash $K/setup_h100.sh
[ -f $B/SETUP_OK ] || { note "SETUP_FAILED, see $B/setup.log"; exit 1; }
note "setup ok: $(cat $B/torch_pins.txt | tr '\n' ' ')"
bash $K/run_hsdp_h100.sh
note "hsdp: $(tail -1 $R/hsdp/progress.txt)"
bash $K/run_m3_4d_h100.sh
note "m4d: $(tail -1 $R/m4d/progress.txt)"
. $V/bin/activate
( cd $T5 && python $K/summarize_m4d.py $R/m4d > $R/m4d/summary.md 2>&1; python $K/tables_m4d.py $R/m4d > $R/m4d/tables_en.md 2>&1 )
if grep -q HSDP_DONE $R/hsdp/progress.txt && grep -q M4D_DONE $R/m4d/progress.txt && [ -s $R/m4d/summary.md ]; then note CHAIN_DONE; else note CHAIN_INCOMPLETE; fi
