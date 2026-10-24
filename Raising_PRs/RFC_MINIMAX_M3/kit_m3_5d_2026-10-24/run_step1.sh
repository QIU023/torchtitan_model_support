#!/bin/bash
# Step-1 probe for one cell: run_step1.sh <tree> <out dir> <gpus> <M3_CELL> <seed> <cache dir> <dump grads 0|1>
T=$1 O=$2 G=$3 C=$4 SEED=$5 CACHE=$6 GRADS=$7
K=$(cd "$(dirname "$0")" && pwd)
N=$(echo $G | tr ',' '\n' | wc -l)
rm -rf $O; mkdir -p $O
. /workspace/venv_1024/bin/activate
cd $T && M3_CELL="$C" M3_STEPS=1 M3_SEED=$SEED M3_PROBE_DIR=$O M3_DUMP_GRADS=$GRADS PYTHONPATH=${M3_EXTRA_PATH:+$M3_EXTRA_PATH:}$K:$T \
  TORCHINDUCTOR_CACHE_DIR=$CACHE/ic TRITON_CACHE_DIR=$CACHE/tc CUDA_VISIBLE_DEVICES=$G NCCL_NVLS_ENABLE=0 timeout 2400 \
  torchrun --nproc_per_node=$N --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank \
  $K/step1_probe.py --module ${M3_MODULE:-m3_text18_probe} --config cell --output-dir $O/out > $O/run.log 2>&1
echo "rc=$? $(cat $O/loss_rank*.txt 2>/dev/null | sort -u | head -2 | tr '\n' ' ')"
rm -rf $O/out
