#!/usr/bin/env bash
#
# Helion MLIR benchmarking setup
export AIBENCH_WARMUP=10
export AIBENCH_REP=500
export KMP_AFFINITY=granularity=fine,compact,1,0
export OMP_NUM_THREADS=64
export LD_PRELOAD=/lib64/libtcmalloc.so:$LD_PRELOAD
export HELION_MLIR_PIPELINE=1
#export HELION_MLIR_CACHE_PREPACKED_WEIGHTS=0 or 1
