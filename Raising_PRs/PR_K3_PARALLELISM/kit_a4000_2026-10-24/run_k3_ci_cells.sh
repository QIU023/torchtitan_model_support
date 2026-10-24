#!/bin/bash
# The five Kimi K3 CI cells of main on 8 x A4000 (venv_1024), through run_tests as the CI lanes call it: b200 kimi_k3_mm (4 GPUs),
# kimi_k3_mm_muon (2, loss_compare --assert-equal against the B200 golden), kimi_k3_fsdp2_tp2_ep2_pp2_vpp4 (8); h100
# kimi_k3_mm_allgather_kv_cp and kimi_k3_mm_ulysses_cp (2 each). Sequential run_tests mode with the GPUs pinned here, so the
# independent cells share the box; each cell gets a fresh output dir. Smoke only: default inductor and triton caches.
# shims/sitecustomize.py (on PYTHONPATH) runs the KDA bounded gate as impl="reference": the fused gate needs sm90.
V=/workspace/venv_1024; T=/workspace/tt_main; R=${1:-/workspace/k3_ci_1024}; P=$R/progress.txt
K=$(cd "$(dirname "$0")" && pwd)
mkdir -p $R
note() { echo "$(date -u +%H:%M:%S) $*" >> $P; }
. $V/bin/activate
cell() {  # <suite> <test_name> <gpus>
  local suite=$1 name=$2 gpus=$3
  ( cd $T && PYTHONPATH=$K/shims CUDA_VISIBLE_DEVICES=$gpus TORCH_SHOW_CPP_STACKTRACES=1 NCCL_NVLS_ENABLE=0 timeout 3600 \
      python -m tests.integration_tests.run_tests $R/$name --test_suite $suite --test_name $name --ngpu 8 --no-parallel \
      > $R/$name.log 2>&1 )
  local rc=$?
  note "$name rc=$rc $(sed 's/\x1b\[[0-9;]*m//g' $R/$name.log | grep -a -o 'step: *[0-9]* *loss: *[0-9.]*' | tail -1)" \
    "$(grep -a -m1 -o -E '[A-Za-z]*(Error|Exception): .*' $R/$name.log | cut -c1-200)"
}
note "tt_main $(git -C $T rev-parse --short HEAD) dirty=$(git -C $T status --short | wc -l) torch $(python -c 'import torch; print(torch.__version__)') shim $K/shims"
( cell b200 kimi_k3_mm 0,1,2,3 ) &
( cell b200 kimi_k3_mm_muon 4,5 ) &
( cell h100 kimi_k3_mm_allgather_kv_cp 6,7; cell h100 kimi_k3_mm_ulysses_cp 6,7 ) &
wait
cell b200 kimi_k3_fsdp2_tp2_ep2_pp2_vpp4 0,1,2,3,4,5,6,7
note done
