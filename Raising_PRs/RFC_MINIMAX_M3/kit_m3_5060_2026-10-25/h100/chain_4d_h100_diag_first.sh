#!/bin/bash
# After the FSDP2 chain: the fp32 HF parity localization probes, then the 4-GPU 4D matrix chain.
. "$(dirname "$0")/env_h100.sh"
until grep -q "CHAIN_FSDP2_DONE\|SETUP_FAILED" $R/chain_fsdp2.txt 2>/dev/null; do sleep 20; done
bash $K/run_diag_h100.sh
bash $K/chain_4d_h100.sh
