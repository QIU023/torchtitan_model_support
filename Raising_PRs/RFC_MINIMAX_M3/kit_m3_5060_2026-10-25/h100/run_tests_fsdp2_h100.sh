#!/bin/bash
# The baseline's GPU unit tests and its models-suite cell on the H100 box (the cell with --no-parallel on GPUs 0-3, FA off).
. "$(dirname "$0")/env_h100.sh"
O=$R/tests; mkdir -p "${O:?}"; P=$O/progress.txt; note() { echo "$(date +%H:%M:%S) $*" >> $P; }
. $V/bin/activate
( cd $TB && CUDA_VISIBLE_DEVICES=0 timeout 1800 python -m pytest tests/unit_tests/gpu/models/minimax_m3 -q -p no:cacheprovider > $O/gpu_unit.txt 2>&1 < /dev/null )
note "gpu_unit rc=$? $(tail -1 $O/gpu_unit.txt)"
rm -rf "${O:?}/ci_out"
( cd $TB && M3_FA_OFF=1 PYTHONPATH=$SHIM:$TB CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 3000 python -m tests.integration_tests.run_tests $O/ci_out \
    --test_suite models --test_name minimax_m3_mm_fsdp --ngpu 4 --no-parallel > $O/ci_cell.txt 2>&1 < /dev/null )
note "ci_cell minimax_m3_mm_fsdp rc=$?"
note TESTS_DONE
