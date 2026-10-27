#!/bin/bash
# After rebasing both M3 branches onto e111f10d4 (#5583 tied RoPE cache): numerics before and after on this box, then the
# GPU tests. HF parity (bf16 default) on the old and the new baseline tree, sharing one fresh compile cache; fsdp4 for 3
# steps on c4_test, each tree from its own single-GPU seed, the new tree on a copy of the cache the old tree filled; then
# the M3 GPU unit tests and the two models-suite cells on the new trees.
. /workspace/kit_m3_5060/local5060/env_h100.sh
OLD=/workspace/wt_m3_pre_e111; NEW=/workspace/wt_m3; NEW5=/workspace/wt_m3_5d
O=/workspace/kit_m3_5060/runs5060/rebase_e111; mkdir -p "${O:?}"; P=$O/progress.txt
export M3_DATASET=c4_test
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
. $V/bin/activate
note "old $(git -C $OLD rev-parse --short HEAD) new $(git -C $NEW rev-parse --short HEAD) new5d $(git -C $NEW5 rev-parse --short HEAD) dirty=$(( $(git -C $OLD status --short | wc -l) + $(git -C $NEW status --short | wc -l) + $(git -C $NEW5 status --short | wc -l) ))"

M='^(pixel values|vision features|sparse attention key blocks|router choices|last-token logits|all-position logits)'
PC=$O/cache_parity; rm -rf "${PC:?}"
for t in old new; do
  T=$OLD; [ $t = new ] && T=$NEW
  ( cd $T && M3_FA_OFF=1 PYTHONPATH=$SHIM:$T CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=$PC/ic TRITON_CACHE_DIR=$PC/tc \
      timeout 3000 python -m scripts.checkpoint_conversion.numerical_tests_minimax_m3 > $O/parity_$t.txt 2>&1 < /dev/null )
  note "parity $t rc=$? shim=$(grep -a -c m3_5060_shim $O/parity_$t.txt)"
  grep -E "$M" $O/parity_$t.txt > $O/parity_$t.metrics
done
rm -rf "${PC:?}"
cmp -s $O/parity_old.metrics $O/parity_new.metrics && note "parity: all $(wc -l < $O/parity_new.metrics) metric lines identical" || note "parity: metric lines DIFFER"

seed() {  # <tag> <tree>
  local tag=$1 T=$2 S=$O/seed_$1
  rm -rf "${S:?}" "${O:?}/seed_run_$tag" "${O:?}/cache_seed_$tag"
  ( cd $T && M3_SEED_OUT=$S TORCHINDUCTOR_CACHE_DIR=$O/cache_seed_$tag/ic TRITON_CACHE_DIR=$O/cache_seed_$tag/tc PYTHONPATH=$SHIM:$K:$T \
      CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
      -m torchtitan.train --module m3_text18_probe --config seed --output-dir $O/seed_run_$tag > $O/seed_$tag.log 2>&1 < /dev/null )
  rm -rf "${O:?}/seed_run_$tag" "${O:?}/cache_seed_$tag"
  note "seed $tag: $([ -d $S/step-0 ] && echo written || echo MISSING)"
}
run() {  # <tag> <tree> <cache dir>
  local tag=$1 T=$2 C=$3 D=$O/fsdp4_$1
  rm -rf "${D:?}"; mkdir -p $D/tmp
  ( cd $T && M3_DUMP_GRADS=1 M3_CELL="dp=4" M3_STEPS=3 M3_SEED=$O/seed_$tag/step-0 M3_PROBE_DIR=$D PYTHONPATH=$SHIM:$K:$T \
      TORCHINDUCTOR_CACHE_DIR=$C/ic TRITON_CACHE_DIR=$C/tc TMPDIR=$D/tmp CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 3000 \
      torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
      $K/step1_probe.py --module m3_text18_probe --config cell --output-dir $D/out > $D/run.log 2>&1 < /dev/null )
  local rc=$?; rm -rf "${D:?}/out" "${D:?}/tmp"
  note "fsdp4 $tag rc=$rc shim=$(grep -a -c m3_5060_shim $D/run.log) seed_loaded=$(grep -a -c 'Loading the checkpoint' $D/run.log) step1=[$(head -1 $D/loss_rank0.txt 2>/dev/null)] step3=[$(sed -n 3p $D/loss_rank0.txt 2>/dev/null)]"
}
seed old $OLD
seed new $NEW
C=$O/cache_fsdp4; rm -rf "${C:?}" "${C:?}_new"
run old $OLD $C
cp -r $C ${C}_new
run new $NEW ${C}_new
rm -rf "${C:?}" "${C:?}_new"
same=0; for f in $O/fsdp4_old/loss_rank*.txt; do cmp -s $f $O/fsdp4_new/$(basename $f) && same=$((same+1)); done
note "fsdp4 loss files identical: $same/$(ls $O/fsdp4_old/loss_rank*.txt | wc -l)"
( cd $NEW && python $K/compare_grads.py $O/fsdp4_old/grads.pt $O/fsdp4_new/grads.pt > $O/fsdp4_grads_old_vs_new.txt 2>&1 )
note "fsdp4 step-1 grads old vs new: $(head -1 $O/fsdp4_grads_old_vs_new.txt)"

( cd $NEW && CUDA_VISIBLE_DEVICES=0 timeout 1800 python -m pytest tests/unit_tests/gpu/models/minimax_m3 -q -p no:cacheprovider > $O/gpu_unit_baseline.txt 2>&1 < /dev/null )
note "gpu_unit baseline rc=$? $(tail -1 $O/gpu_unit_baseline.txt)"
( cd $NEW5 && CUDA_VISIBLE_DEVICES=0 timeout 1800 python -m pytest tests/unit_tests/gpu/models/minimax_m3 -q -p no:cacheprovider > $O/gpu_unit_5d.txt 2>&1 < /dev/null )
note "gpu_unit 5d rc=$? $(tail -1 $O/gpu_unit_5d.txt)"
rm -rf "${O:?}/ci_base" "${O:?}/ci_5d"
( cd $NEW && M3_FA_OFF=1 PYTHONPATH=$SHIM:$NEW CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 3000 python -m tests.integration_tests.run_tests $O/ci_base \
    --test_suite models --test_name minimax_m3_mm_fsdp --ngpu 4 --no-parallel > $O/ci_base.txt 2>&1 < /dev/null; echo $? > $O/ci_base.rc ) &
( cd $NEW5 && M3_FA_OFF=1 PYTHONPATH=$SHIM:$NEW5 CUDA_VISIBLE_DEVICES=4,5,6,7 timeout 3000 python -m tests.integration_tests.run_tests $O/ci_5d \
    --test_suite models --test_name "minimax_m3_mm_fsdp+tp+sp+ep" --ngpu 4 --no-parallel > $O/ci_5d.txt 2>&1 < /dev/null; echo $? > $O/ci_5d.rc ) &
wait
steps() { sed "s/\x1b\[[0-9;]*m//g" $1 | grep -a -o "step: *[0-9]* *loss: *[0-9.]*" | sed -n '1p;$p' | tr -s ' ' | tr '\n' ';'; }
note "ci minimax_m3_mm_fsdp rc=$(cat $O/ci_base.rc) [$(steps $O/ci_base.txt)]"
note "ci minimax_m3_mm_fsdp+tp+sp+ep rc=$(cat $O/ci_5d.rc) [$(steps $O/ci_5d.txt)]"
note GPU_REBASE_DONE
