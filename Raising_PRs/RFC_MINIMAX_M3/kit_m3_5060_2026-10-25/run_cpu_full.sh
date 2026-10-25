#!/bin/bash
# CI's CPU command (`pytest tests/unit_tests/cpu`) on one tree, GPUs hidden, venv_1025; junit XML for a per-test comparison.
# run_cpu_full.sh <tree> <name>
T=$1 N=$2 O=/workspace/kit_m3_5060/cpu_full; mkdir -p $O
cd $T && CUDA_VISIBLE_DEVICES="" PYTHONPATH=$T timeout 10800 /workspace/venv_1025/bin/python -m pytest tests/unit_tests/cpu \
  -p no:cacheprovider -q -rfE --junitxml=$O/$N.xml > $O/$N.log 2>&1
echo "rc=$? $(tail -1 $O/$N.log)" > $O/$N.done
