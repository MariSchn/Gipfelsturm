#!/bin/bash
#
# Usage: ./scripts/sweep_moe.sh <model_size> [steps] [nodes]
#
# Sweeps MoE configurations across three axes:
#   A) Expert count    — how does routing overhead scale with more experts?     (K=2, EP=1)
#   B) Expert parallel — what is the All-to-All cost over NVLink?               (E=8, K=2)
#   C) Top-K routing   — top-1 (Switch) vs top-2 (Mixtral) at same EP?         (E=8, EP=4)
#
# Includes a dense baseline for direct comparison.
# All configs keep active parameters constant relative to the base model.
#
# Constraints: EP must divide total GPUs (nodes*4), EP <= num_experts.
#
# Example:
#   ./scripts/sweep_moe.sh 125m        # 50 steps, 1 node
#   ./scripts/sweep_moe.sh 760m 50 1

set -euo pipefail

MODEL_SIZE=${1:?Usage: ./scripts/sweep_moe.sh <model_size> [steps] [nodes]}
STEPS=${2:-50}
NODES=${3:-1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH="${SCRIPT_DIR}/../launch.sh"

echo "=========================================="
echo " MoE sweep: ${MODEL_SIZE}, ${STEPS} steps, ${NODES} node(s)"
echo "=========================================="

# --- Baseline ---
echo "--- [0] Dense baseline ---"
"${LAUNCH}" throughput "${MODEL_SIZE}" "${STEPS}" "${NODES}"

# --- A: Expert count (K=2, EP=1) ---
# Routing overhead scales with num_experts; active params stay constant.
# EP=1 means experts are replicated across GPUs — no All-to-All, isolates routing cost.
echo "--- [A1] E=4,  K=2, EP=1 ---"
NUM_EXPERTS=4  MOE_TOPK=2 EP_SIZE=1 "${LAUNCH}" throughput "${MODEL_SIZE}" "${STEPS}" "${NODES}"

echo "--- [A2] E=8,  K=2, EP=1 ---"
NUM_EXPERTS=8  MOE_TOPK=2 EP_SIZE=1 "${LAUNCH}" throughput "${MODEL_SIZE}" "${STEPS}" "${NODES}"

echo "--- [A3] E=16, K=2, EP=1 ---"
NUM_EXPERTS=16 MOE_TOPK=2 EP_SIZE=1 "${LAUNCH}" throughput "${MODEL_SIZE}" "${STEPS}" "${NODES}"

# --- B: Expert parallelism (E=8, K=2) ---
# Distributes experts across GPUs; adds All-to-All token dispatch over NVLink.
# EP=2: 2 expert-groups share the node; EP=4: all 4 GPUs each hold E/4 experts.
echo "--- [B1] E=8, K=2, EP=2 ---"
NUM_EXPERTS=8 MOE_TOPK=2 EP_SIZE=2 "${LAUNCH}" throughput "${MODEL_SIZE}" "${STEPS}" "${NODES}"

echo "--- [B2] E=8, K=2, EP=4 ---"
NUM_EXPERTS=8 MOE_TOPK=2 EP_SIZE=4 "${LAUNCH}" throughput "${MODEL_SIZE}" "${STEPS}" "${NODES}"

# --- C: Top-K routing (E=8, EP=4) ---
# K=1: each token activates 1 expert (Switch Transformer) — half the FFN compute of K=2.
# K=2: each token activates 2 experts (Mixtral-style) — better utilisation, more compute.
echo "--- [C1] E=8, K=1, EP=4 ---"
NUM_EXPERTS=8 MOE_TOPK=1 EP_SIZE=4 "${LAUNCH}" throughput "${MODEL_SIZE}" "${STEPS}" "${NODES}"

echo "=========================================="
echo " Submitted 7 jobs. Check: squeue --me"
echo "=========================================="
