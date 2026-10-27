#!/bin/bash
# The 4D body's HF parity (bf16, free and HF-imposed) on the old and the rebased 5D tree, sharing one fresh compile cache.
. /workspace/kit_m3_5060/local5060/env_h100.sh
OLD=/workspace/wt_m3_5d_pre_e111; NEW=/workspace/wt_m3_5d
O=/workspace/kit_m3_5060/runs5060/rebase_e111; P=$O/progress.txt
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
. $V/bin/activate
note "parity5d old $(git -C $OLD rev-parse --short HEAD) new $(git -C $NEW rev-parse --short HEAD)"
M='^(pixel values|vision features|sparse attention key blocks|router choices|last-token logits|all-position logits)'
PC=$O/cache_parity5d; rm -rf "${PC:?}"
for mode in free imposed; do
  args=""; [ $mode = imposed ] && args="--force-hf-blocks --force-hf-routing"
  for t in old new; do
    T=$OLD; [ $t = new ] && T=$NEW
    ( cd $T && M3_FA_OFF=1 PYTHONPATH=$SHIM:$T CUDA_VISIBLE_DEVICES=4 TORCHINDUCTOR_CACHE_DIR=$PC/ic TRITON_CACHE_DIR=$PC/tc \
        timeout 3000 python -m scripts.checkpoint_conversion.numerical_tests_minimax_m3 $args > $O/parity5d_${mode}_$t.txt 2>&1 < /dev/null )
    note "parity5d $mode $t rc=$?"
    grep -E "$M" $O/parity5d_${mode}_$t.txt > $O/parity5d_${mode}_$t.metrics
  done
  cmp -s $O/parity5d_${mode}_old.metrics $O/parity5d_${mode}_new.metrics && note "parity5d $mode: all $(wc -l < $O/parity5d_${mode}_new.metrics) metric lines identical" || note "parity5d $mode: metric lines DIFFER"
done
rm -rf "${PC:?}"
note PARITY5D_REBASE_DONE
