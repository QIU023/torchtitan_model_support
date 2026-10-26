#!/bin/bash
# Pull the H200 box's text results into the logbook kit every minute until the 100-step chain ends (no dumps, caches or seeds).
L=/workspace/torchtitan_attention_residual/Raising_PRs/RFC_MINIMAX_M3/kit_m3_5060_2026-10-25/results_h200
mkdir -p $L
S="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20"
for i in $(seq 1 240); do
  timeout 120 rsync -a -e "$S" --prune-empty-dirs \
    --exclude 'cache*' --exclude 'cc*' --exclude 'seed*' --exclude '*.pt' --exclude 'tmp' --exclude 'out' \
    --include '*/' --include '*.txt' --include '*.md' --include '*.log' --exclude '*' \
    root@217.18.55.94:/home/m3/runs/ $L/runs/ 2>/dev/null
  timeout 60 rsync -a -e "$S" root@217.18.55.94:/home/m3/setup.log root@217.18.55.94:/home/m3/torch_pins.txt root@217.18.55.94:/home/m3/venv_freeze.txt $L/ 2>/dev/null
  if grep -q "CHAIN_S100_DONE\|CHAIN_S100_INCOMPLETE" $L/runs/chain_s100.txt 2>/dev/null; then tail -3 $L/runs/chain_s100.txt; exit 0; fi
  sleep 60
done
echo "pull loop timed out"
