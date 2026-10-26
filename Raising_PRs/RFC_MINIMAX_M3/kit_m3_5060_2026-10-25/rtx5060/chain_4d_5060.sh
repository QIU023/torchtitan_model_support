#!/bin/bash
# Everything the 4D draft body cites, on the local 8 x RTX 5060 Ti (user 10-26: the 4D numbers come from the 5060 only):
# the FSDP-only matrix at 100 steps on c4_test (all 29 layouts fit on 8 GPUs), its 4312-format tables and summary, the EP
# gradient check, the TP forced-selection attribution, the models-suite cell of the 4D branch and HF parity of its debug
# flavor; text results under $R.
. "$(dirname "$0")/env_h100.sh"
P=$R/chain_4d.txt; mkdir -p "${R:?}"; note() { echo "$(date +%H:%M:%S) $*" >> $P; }
note "chain start, tree $(git -C $T5 rev-parse --short HEAD) dirty=$(git -C $T5 status --short | wc -l)"
RUN_TAG=s100 MEAS_STEPS=100 M3_DATASET=c4_test bash $K/run_m3_4d_h100.sh
O=$R/m4d_s100
. $V/bin/activate
( cd $T5 && python3 $K/tables_4312.py $O auto > $O/table_4312.md 2>&1
  M4D_STEPS=100 M4D_COLS=1,10,50,100 python $K/summarize_m4d.py $O > $O/summary.md 2>&1
  python $K/ep_param_diff.py $O fsdp2:ep2_fsdp2 fsdp4:ep4_fsdp4 fsdp8:ep8_fsdp8 > $O/ep_param_diff.txt 2>&1 )
note "matrix: $(tail -1 $O/progress.txt)"
bash $K/run_tp_force_s100.sh
T=$R/tests5d; mkdir -p "${T:?}"; rm -rf "${T:?}/ci_out"
( cd $T5 && M3_FA_OFF=1 PYTHONPATH=$SHIM:$T5 CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 3000 python -m tests.integration_tests.run_tests $T/ci_out \
    --test_suite models --test_name "minimax_m3_mm_fsdp+tp+sp+ep" --ngpu 4 --no-parallel > $T/ci_cell.txt 2>&1 < /dev/null )
note "ci_cell minimax_m3_mm_fsdp+tp+sp+ep rc=$?"
( cd $T5 && CUDA_VISIBLE_DEVICES=0 timeout 1800 python -m pytest tests/unit_tests/gpu/models/minimax_m3 -q -p no:cacheprovider > $T/gpu_unit.txt 2>&1 < /dev/null )
note "gpu_unit rc=$? $(tail -1 $T/gpu_unit.txt)"
Q=$R/parity5d; mkdir -p "${Q:?}"
par() {  # <name> <gpu> [args...]
  local name=$1 gpu=$2; shift 2
  ( cd $T5 && M3_FA_OFF=1 PYTHONPATH=$SHIM:$T5 CUDA_VISIBLE_DEVICES=$gpu TORCHINDUCTOR_CACHE_DIR=$Q/cache_$name/ic TRITON_CACHE_DIR=$Q/cache_$name/tc \
      timeout 3000 python -m scripts.checkpoint_conversion.numerical_tests_minimax_m3 "$@" > $Q/$name.txt 2>&1 < /dev/null )
  note "parity5d $name rc=$?"
  rm -rf "${Q:?}/cache_${name:?}"
}
par bf16_free 0
par bf16_imposed 1 --force-hf-blocks --force-hf-routing
note CHAIN_4D_DONE
