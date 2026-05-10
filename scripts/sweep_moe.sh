#!/bin/bash
#
# Usage: ./scripts/sweep_moe.sh <model_size> [steps] [nodes]
#
# Sweeps MoE configurations: num_experts x EP size.
# Each config launches a throughput job via launch.sh.
#
# Example:
#   ./scripts/sweep_moe.sh 760m
#   ./scripts/sweep_moe.sh 3b 50 4

set -euo pipefail

MODEL_SIZE=${1:?Usage: ./scripts/sweep_moe.sh <model_size> [steps] [nodes]}
STEPS=${2:-50}
NODES=${3:-4}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH="${SCRIPT_DIR}/../launch.sh"

# Each entry: "NUM_EXPERTS MOE_TOPK EP_SIZE"
CONFIGS=(
    "8  2 1"
    "8  2 4"
    "16 2 1"
    "16 2 4"
    "64 2 4"
)

for cfg in "${CONFIGS[@]}"; do
    read -r NE TOPK EP <<< "$cfg"
    echo "=== NUM_EXPERTS=${NE}  MOE_TOPK=${TOPK}  EP_SIZE=${EP} ==="
    NUM_EXPERTS="${NE}" MOE_TOPK="${TOPK}" EP_SIZE="${EP}" \
        "${LAUNCH}" throughput "${MODEL_SIZE}" "${STEPS}" "${NODES}"
done
