#!/bin/bash
#
# Usage: ./scripts/sweep_dp.sh <model_size> [steps]
#
# Sweeps pure data parallelism over DP = 1, 2, 4, 8, 16.
# Sub-node DP values (1, 2) use GPUS_PER_NODE < 4 on a single node.
# TP=1, PP=1 throughout.
#
#   DP=1   -> 1 node,  1 GPU/node
#   DP=2   -> 1 node,  2 GPUs/node
#   DP=4   -> 1 node,  4 GPUs/node
#   DP=8   -> 2 nodes, 4 GPUs/node
#   DP=16  -> 4 nodes, 4 GPUs/node

set -euo pipefail

MODEL_SIZE=${1:?Usage: ./scripts/sweep_dp.sh <model_size> [steps]}
STEPS=${2:-50}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH="${SCRIPT_DIR}/../launch.sh"

# Each entry: "DP NODES GPUS_PER_NODE"
case "$MODEL_SIZE" in
    125m)
        CONFIGS=(
            "1  1 1"
            "2  1 2"
            "4  1 4"
            "8  2 4"
            "16 4 4"
        )
        ;;
    350m)
        CONFIGS=(
            "1  1 1"
            "2  1 2"
            "4  1 4"
            "8  2 4"
            "16 4 4"
        )
        ;;
    760m)
        CONFIGS=(
            "1  1 1"
            "2  1 2"
            "4  1 4"
            "8  2 4"
            "16 4 4"
        )
        ;;
    1.5b)
        CONFIGS=(
            "1  1 1"
            "2  1 2"
            "4  1 4"
            "8  2 4"
            "16 4 4"
        )
        ;;
    3b)
        CONFIGS=(
            # "1  1 1"  # OOMs, due to not sharding optimizer states
            "2  1 2"
            "4  1 4"
            "8  2 4"
            "16 4 4"
        )
        ;;
    8b)
        CONFIGS=(
            # "1  1 1"  # OOMs, due to not sharding optimizer states
            # "2  1 2"  # OOMs, due to not sharding optimizer states
            "4  1 4"
            "8  2 4"
            "16 4 4"
        )
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE"
        exit 1
        ;;
esac

for cfg in "${CONFIGS[@]}"; do
    read -r DP NODES GPN <<< "$cfg"
    echo "=== Launching DP=${DP}  (NODES=${NODES}, GPUS_PER_NODE=${GPN}) ==="
    GPUS_PER_NODE="${GPN}" "${LAUNCH}" throughput "${MODEL_SIZE}" "${STEPS}" "${NODES}" 1 1
done
