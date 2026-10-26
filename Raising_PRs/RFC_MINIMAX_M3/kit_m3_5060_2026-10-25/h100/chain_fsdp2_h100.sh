#!/bin/bash
# Everything the FSDP2 baseline body cites, on the H100 box: main's worktree, the CPU suites in the background, then the
# dp4 HSDP table (100 steps), its reduction rebuild, HF parity, the GPU unit tests and models-suite cell, the P2 check,
# and pre-commit last; text results under $R.
. "$(dirname "$0")/env_h100.sh"
P=$R/chain_fsdp2.txt; mkdir -p "${R:?}"; note() { echo "$(date +%H:%M:%S) $*" >> $P; }
note "chain start"
while [ ! -f $B/SETUP_OK ] && pgrep -f "[s]etup_h100.sh" > /dev/null; do sleep 10; done
[ -f $B/SETUP_OK ] || { note "SETUP_FAILED, see $B/setup.log"; exit 1; }
note "setup ok: $(cat $B/torch_pins.txt | tr '\n' ' ')"
MAIN=3ed5aa976fdf38e05fd50885ab192ea66ed6c5b7
if [ ! -d $B/tt_main ]; then
  git -C $B/tt fetch -q https://github.com/pytorch/torchtitan.git $MAIN && git -C $B/tt worktree add --detach $B/tt_main $MAIN > /dev/null 2>&1
fi
note "main tree $(git -C $B/tt_main rev-parse --short HEAD 2>/dev/null || echo missing)"
bash $K/run_cpu_h100.sh &
CPU_PID=$!
RUN_TAG=s100 MEAS_STEPS=100 M3_DATASET=c4_test bash $K/run_hsdp_h100.sh
python3 $K/tables_4312.py $R/hsdp_s100 dp4:fsdp4:fsdp4_again,hsdp2x2 > $R/hsdp_s100/table_4312.md 2>&1
note "dp4 table: $(tail -1 $R/hsdp_s100/progress.txt)"
bash $K/run_hsdp_rebuild_h100.sh
bash $K/run_parity_h100.sh
bash $K/run_tests_fsdp2_h100.sh
bash $K/run_p2_h100.sh
wait $CPU_PID
bash $K/run_precommit_h100.sh
note CHAIN_FSDP2_DONE
