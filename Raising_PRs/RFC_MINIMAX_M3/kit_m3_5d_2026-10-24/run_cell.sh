#!/bin/bash
# One M3 matrix cell: run_cell.sh <tree> <out dir> <gpus> <M3_CELL> [extra env...]
T=$1 O=$2 G=$3 C=$4; shift 4
K=$(cd "$(dirname "$0")" && pwd)
N=$(echo $G | tr ',' '\n' | wc -l)
mkdir -p $O
. /workspace/venv_1024/bin/activate
cd $T && env "$@" M3_CELL="$C" PYTHONPATH=$K:$T CUDA_VISIBLE_DEVICES=$G NCCL_NVLS_ENABLE=0 timeout 3000 \
  torchrun --nproc_per_node=$N --rdzv_backend c10d --rdzv_endpoint=localhost:0 --local-ranks-filter $(seq -s, 0 $((N - 1))) --role rank --tee 3 \
  -m torchtitan.train --module m3_text18_probe --config cell --output-dir $O/out > $O/run.log 2>&1
rc=$?; rm -rf $O/out
echo "rc=$rc $(sed 's/\x1b\[[0-9;]*m//g' $O/run.log | grep -a -o 'step: *[0-9]* *loss: *[0-9.]* *grad_norm: *[0-9.]*' | tail -1)" > $O/result.txt
cat $O/result.txt
