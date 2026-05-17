#!/bin/bash
#
# Usage: ./scripts/sweep_workers.sh <model_size> [steps] [nodes]
#
# Sweeps DATA_NUM_WORKERS from 1 to 32 in powers of 2:
#   DATA_NUM_WORKERS = 1, 2, 4, 8, 16, 32
# TP=1, PP=1 throughout.

set -euo pipefail

MODEL_SIZE=${1:?Usage: ./scripts/sweep_workers.sh <model_size> [steps] [nodes]}
STEPS=${2:-50}
NODES=${3:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH="${SCRIPT_DIR}/../launch.sh"

for WORKERS in 1 2 4 8 16 32; do
    echo "=== Launching DATA_NUM_WORKERS=${WORKERS} ==="
    GPUS_PER_NODE="${GPUS_PER_NODE}" DATA_NUM_WORKERS="${WORKERS}" "${LAUNCH}" throughput "${MODEL_SIZE}" "${STEPS}" "${NODES}" 1 1
done
