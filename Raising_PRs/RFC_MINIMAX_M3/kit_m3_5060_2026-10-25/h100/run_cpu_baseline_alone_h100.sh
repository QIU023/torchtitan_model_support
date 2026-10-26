#!/bin/bash
# The baseline's CPU suite again, alone: test_parallelism_context.py binds a hard-coded port 12356, so the first pass,
# which ran next to main's suite, lost 8 tests to EADDRINUSE; then the test-by-test comparison with main's pass.
. "$(dirname "$0")/env_h100.sh"
O=$R/cpu; P=$O/progress.txt; note() { echo "$(date +%H:%M:%S) $*" >> $P; }
. $V/bin/activate
mv $O/baseline.log $O/baseline_concurrent.log; mv $O/baseline.xml $O/baseline_concurrent.xml
( cd $TB && CUDA_VISIBLE_DEVICES="" PYTHONPATH=$TB timeout 10800 python -m pytest tests/unit_tests/cpu -p no:cacheprovider -q -rfE \
    --junitxml=$O/baseline.xml > $O/baseline.log 2>&1 < /dev/null )
note "baseline alone rc=$? $(tail -1 $O/baseline.log)"
python $K/junit_compare.py $O/main.xml $O/baseline.xml tests.unit_tests.cpu.models.minimax_m3 MiniMaxM3 > $O/compare.txt 2>&1
note "CPU_ALONE_DONE $(sed -n 2p $O/compare.txt)"
