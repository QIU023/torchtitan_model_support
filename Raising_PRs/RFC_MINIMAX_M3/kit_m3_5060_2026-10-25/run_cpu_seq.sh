#!/bin/bash
# The first pass ran three suites at once and two of them lost the fixed gloo port 29511 (EADDRINUSE); rerun those two
# one after the other, alone on the box's CPU-suite side.
/workspace/kit_m3_5060/run_cpu_full.sh /workspace/tt_main_3ed5 main_3ed5aa976
/workspace/kit_m3_5060/run_cpu_full.sh /workspace/wt_m3 minimax_m3_9a5ab9ebe
