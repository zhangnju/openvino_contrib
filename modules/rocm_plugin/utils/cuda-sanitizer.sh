#!/bin/bash

set -exuo pipefail

TESTED_TOOL_CMD_LINE=$@

# =======================================================================================
# common
rocmSAN_KERNEL_FILTER="--filter kernel_substring=rocm_gpu"
rocmSAN_COMMON_ARGS="--print-level=warn --print-limit=100 --error-exitcode=1"

# =======================================================================================
# memcheck
# Reports thr folowing:
#   - Errors due to out of bounds or misaligned accesses to memory by a global, local, shared or global atomic access.
#   - Errors that are reported by the hardware error reporting mechanism.
#   - Errors that occur due to incorrect use of malloc()/free() in rocm kernels.
#   - Allocations of device memory using rocmMalloc() that have not been freed by the application.
#   - Allocations of device memory using malloc() in device code that have not been freed by the application.
rocmSAN_TOOL_ARGS="--prefix=[rocm-memcheck] --leak-check=full --check-device-heap=yes --report-api-errors=no"
echo "[rocm-memcheck] started"
time compute-sanitizer --tool=memcheck ${rocmSAN_TOOL_ARGS} ${rocmSAN_COMMON_ARGS} ${TESTED_TOOL_CMD_LINE}
echo "[rocm-memcheck] completed"

# =======================================================================================
# racecheck
#   Identify rocm shared memory memory access race conditions.
rocmSAN_TOOL_ARGS="--prefix=[rocm-racecheck] --racecheck-report=all"
echo "[rocm-racecheck] started"
time compute-sanitizer --tool=racecheck ${rocmSAN_TOOL_ARGS} ${rocmSAN_KERNEL_FILTER} ${rocmSAN_COMMON_ARGS} ${TESTED_TOOL_CMD_LINE}
echo "[rocm-racecheck] completed"

# =======================================================================================
# synccheck
#   Whether an application is correctly using rocm synchronization primitives, specifically
#   __syncthreads() and __syncwarp() intrinsics and their Cooperative Groups API counterparts.
rocmSAN_TOOL_ARGS="--prefix=[rocm-synccheck]"
echo "[rocm-synccheck] started"
time compute-sanitizer --tool=synccheck ${rocmSAN_TOOL_ARGS} ${rocmSAN_KERNEL_FILTER} ${rocmSAN_COMMON_ARGS} ${TESTED_TOOL_CMD_LINE}
echo "[rocm-synccheck] completed"

# =======================================================================================
# initcheck
#   Identify when device global memory is accessed without it being initialized via device side writes,
#   or via rocm memcpy and memset API calls

#
# This check is disabled due to false positives (version 2020.3.1).
#
echo "[rocm-initcheck] disabled"
# rocmSAN_TOOL_ARGS="--prefix=[rocm-initcheck] --track-unused-memory=no"
# echo "[rocm-initcheck] started"
# time compute-sanitizer --tool=initcheck ${rocmSAN_TOOL_ARGS} ${rocmSAN_KERNEL_FILTER} ${rocmSAN_COMMON_ARGS} ${TESTED_TOOL_CMD_LINE}
# echo "[rocm-initcheck] completed"

# =======================================================================================
