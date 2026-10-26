#!/bin/bash
# HF parity of the baseline tree on the H100 box: bf16 (varlen full attention) free and with HF's block and expert
# choices imposed, fp32 (flex) free and imposed; FA selection off (M3_FA_OFF shim), each run on its own GPU and cache.
. "$(dirname "$0")/env_h100.sh"
O=$R/parity; mkdir -p "${O:?}"; P=$O/progress.txt; note() { echo "$(date +%H:%M:%S) $*" >> $P; }
. $V/bin/activate
one() {  # <name> <gpu> [args...]
  local name=$1 gpu=$2; shift 2
  ( cd $TB && M3_FA_OFF=1 PYTHONPATH=$SHIM:$TB CUDA_VISIBLE_DEVICES=$gpu TORCHINDUCTOR_CACHE_DIR=$O/cache_$name/ic TRITON_CACHE_DIR=$O/cache_$name/tc \
      timeout 3000 python -m scripts.checkpoint_conversion.numerical_tests_minimax_m3 "$@" > $O/$name.txt 2>&1 < /dev/null )
  note "parity $name rc=$? shim=$(grep -c m3_5060_shim $O/$name.txt)"
  rm -rf "${O:?}/cache_${name:?}"
}
note "tree $(git -C $TB rev-parse --short HEAD) torch $(python -c 'import torch;print(torch.__version__)')"
one bf16_free 0
one bf16_imposed 1 --force-hf-blocks --force-hf-routing &
one fp32_free 2 --dtype float32 --attn_backend flex &
one fp32_imposed 3 --dtype float32 --attn_backend flex --force-hf-blocks --force-hf-routing &
wait
note PARITY_DONE
