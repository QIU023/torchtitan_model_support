#!/bin/bash
# CI's CPU suite on main and the baseline at once (GPUs hidden), each with its own MASTER_PORT so the fixed gloo port
# cannot collide; junit XML compared test by test.
. "$(dirname "$0")/env_h100.sh"
O=$R/cpu; mkdir -p "${O:?}"; P=$O/progress.txt; note() { echo "$(date +%H:%M:%S) $*" >> $P; }
. $V/bin/activate
one() {  # <tree> <name> <port>
  ( cd $1 && CUDA_VISIBLE_DEVICES="" MASTER_PORT=$3 PYTHONPATH=$1 timeout 10800 python -m pytest tests/unit_tests/cpu -p no:cacheprovider -q -rfE \
      --junitxml=$O/$2.xml > $O/$2.log 2>&1 < /dev/null )
  note "$2 rc=$? $(tail -1 $O/$2.log)"
}
one $B/tt_main main 29611 &
one $TB baseline 29711 &
wait
python $K/junit_compare.py $O/main.xml $O/baseline.xml tests.unit_tests.cpu.models.minimax_m3 MiniMaxM3 > $O/compare.txt 2>&1
note "CPU_DONE"
