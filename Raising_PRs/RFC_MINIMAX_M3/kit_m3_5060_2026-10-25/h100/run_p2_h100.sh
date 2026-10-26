#!/bin/bash
# P2 check on the H100 box (scripts/loss_compare.py, 10 steps, deterministic, CUDA graphs off): main 3ed5aa976 against
# the baseline (whose P2 commit moves partial RoPE into RoPE), the Qwen3.5 dense model on 1 GPU and the MoE model on
# 4 GPUs (tp 2 x pp 2 x ep 2); per model one cache warmed by both trees, then main, the baseline and main again on copies.
. "$(dirname "$0")/env_h100.sh"
O=$R/p2; mkdir -p "${O:?}"; P=$O/progress.txt; note() { echo "$(date +%H:%M:%S) $*" >> $P; }
. $V/bin/activate
run() {  # <tag> <cfg> <ngpu> <gpus> <name> <tree> <cache> [loss_compare args...]
  local tag=$1 cfg=$2 n=$3 gpus=$4 name=$5 T=$6 C=$7; shift 7
  ( cd $T && M3_FA_OFF=1 PYTHONPATH=$K/p2cfg:$SHIM:$T TORCHINDUCTOR_CACHE_DIR=$C/ic TRITON_CACHE_DIR=$C/tc CUDA_VISIBLE_DEVICES=$gpus timeout 2400 \
      python scripts/loss_compare.py . . --baseline-module=p2check_configs --baseline-config=$cfg --steps=10 \
      --baseline-ngpus=$n --test-ngpus=$n --job-dump-folder=$O/${tag}_$name "$@" > $O/${tag}_$name.log 2>&1 < /dev/null )
  note "$tag $name rc=$?"
}
model() {  # <tag> <cfg> <ngpu> <gpus>
  local tag=$1 cfg=$2 n=$3 gpus=$4
  local C=$O/cache_$tag
  rm -rf "${C:?}" "${C:?}_1" "${C:?}_2" "${C:?}_3"; mkdir -p $C
  run $tag $cfg $n $gpus warm_main $B/tt_main $C --export-result=$O/${tag}_warm_main.txt
  run $tag $cfg $n $gpus warm_base $TB $C --export-result=$O/${tag}_warm_base.txt
  for i in 1 2 3; do cp -r $C ${C}_$i; done
  run $tag $cfg $n $gpus main $B/tt_main ${C}_1 --export-result=$O/${tag}_main.txt
  run $tag $cfg $n $gpus baseline $TB ${C}_2 --import-result=$O/${tag}_main.txt --assert-equal
  run $tag $cfg $n $gpus main_again $B/tt_main ${C}_3 --import-result=$O/${tag}_main.txt --assert-equal
  rm -rf "${C:?}" "${C:?}_1" "${C:?}_2" "${C:?}_3"
}
model dense qwen35_dense_nocg 1 0
model moe qwen35_moe_nocg_4gpu 4 0,1,2,3
note P2_DONE
