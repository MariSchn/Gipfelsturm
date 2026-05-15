#!/bin/bash
#
# Usage: ./scripts/sweep_moe.sh <model_size> [steps]
#
# Sweeps MoE throughput configs for a given model size.
#
# For each model size, active parameters per token = dense base (same FLOPs/token).
# Total parameters grow with num_experts, giving more capacity at the same compute cost.
# The question: does routing + All-to-All overhead outweigh the capacity benefit?
#
# Active/total param ratios with E=8, K=2:
#   125m → 0.52B total  (4.2x)   EP=1 OK on 1 node
#   350m → 1.80B total  (5.2x)   EP=1 OK on 1 node
#   760m → 3.93B total  (5.2x)   EP=1 OK on 1 node
#   1.5b → 8.52B total  (5.7x)   EP=1 OK on 1 node (~47GB)
#   3b   → 19.9B total  (6.6x)   EP=1 OOMs — needs EP≥4 + multi-node
#   8b   → 47.5B total  (5.9x)   EP=1 OOMs — needs EP=8 + 4 nodes (DP=2)
#
# Constraints: EP must divide total GPUs (nodes×4), EP ≤ num_experts.
# Memory note: uses distributed optimizer; optimizer states sharded over DP ranks.
#
# Fine-grained MoE (Qwen3-style, E=128, K=8): set FINE_GRAINED=1 before the run() call,
# or use MOE_FFN_HIDDEN_SIZE=N directly. Not included in the standard sweep — run separately:
#   NUM_EXPERTS=128 MOE_TOPK=8 EP_SIZE=16 FINE_GRAINED=1 ./launch.sh throughput 3b 50 4
#
# Example:
#   ./scripts/sweep_moe.sh 125m
#   ./scripts/sweep_moe.sh 8b 50

set -uo pipefail

MODEL_SIZE=${1:?Usage: ./scripts/sweep_moe.sh <model_size> [steps]}
STEPS=${2:-50}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH="${SCRIPT_DIR}/../launch.sh"

# run NE K EP NODES
# NE=0 → dense baseline (no MoE args)
run() {
    local ne=$1 k=$2 ep=$3 nodes=$4
    if [ "$ne" -eq 0 ]; then
        "${LAUNCH}" throughput "${MODEL_SIZE}" "${STEPS}" "${nodes}"
    else
        NUM_EXPERTS="${ne}" MOE_TOPK="${k}" EP_SIZE="${ep}" \
            "${LAUNCH}" throughput "${MODEL_SIZE}" "${STEPS}" "${nodes}"
    fi
}

echo "=========================================="
echo " MoE sweep: ${MODEL_SIZE}, ${STEPS} steps"
echo "=========================================="

case "$MODEL_SIZE" in

    # ── 125m ─────────────────────────────────────────────────────────────────
    # Active: 125m. Tiny model; memory is trivial for all configs.
    # Focus: isolate routing overhead (EP=1) vs intra-node All-to-All (EP=4)
    #        vs cross-node All-to-All (EP=8), and Switch (K=1) vs Mixtral (K=2).
    125m)
        echo "--- [0]  Dense baseline,                   1 node  ---"; run 0  0 0 1
        echo "--- [A1] E=8,  K=2, EP=1, no All-to-All,  1 node  ---"; run 8  2 1 1
        echo "--- [A2] E=8,  K=2, EP=4, NVLink A2A,     1 node  ---"; run 8  2 4 1
        echo "--- [A3] E=16, K=2, EP=4, 2x capacity,    1 node  ---"; run 16 2 4 1
        echo "--- [A4] E=8,  K=1, EP=4, Switch-style,   1 node  ---"; run 8  1 4 1
        echo "--- [A5] E=8,  K=2, EP=8, Slingshot A2A,  2 nodes ---"; run 8  2 8 2
        ;;

    # ── 350m ─────────────────────────────────────────────────────────────────
    # Active: 350m → 1.8B total (5.2x). Memory fine for all EP on 1 node.
    # Focus: how does routing overhead change as base model grows 3x vs 125m?
    350m)
        echo "--- [0]  Dense baseline,                   1 node  ---"; run 0 0 0 1
        echo "--- [A1] E=8, K=2, EP=1, no All-to-All,   1 node  ---"; run 8 2 1 1
        echo "--- [A2] E=8, K=2, EP=4, NVLink A2A,      1 node  ---"; run 8 2 4 1
        echo "--- [A3] E=8, K=2, EP=8, Slingshot A2A,   2 nodes ---"; run 8 2 8 2
        ;;

    # ── 760m ─────────────────────────────────────────────────────────────────
    # Active: 760m → 3.9B total (5.2x). Memory fine for all EP on 1 node.
    # Focus: does All-to-All cost (NVLink vs Slingshot) scale with model size?
    #        Also tests K=1 (Switch) at a more meaningful scale than 125m.
    760m)
        echo "--- [0]  Dense baseline,                   1 node  ---"; run 0 0 0 1
        echo "--- [A1] E=8, K=2, EP=1, no All-to-All,   1 node  ---"; run 8 2 1 1
        echo "--- [A2] E=8, K=2, EP=4, NVLink A2A,      1 node  ---"; run 8 2 4 1
        echo "--- [A3] E=8, K=1, EP=4, Switch-style,    1 node  ---"; run 8 1 4 1
        echo "--- [A4] E=8, K=2, EP=8, Slingshot A2A,   2 nodes ---"; run 8 2 8 2
        ;;

    # ── 1.5b ─────────────────────────────────────────────────────────────────
    # Active: 1.5b → 8.5B total (5.7x). EP=1 uses ~47GB — still fits on 1 node.
    # EP=8 on 2 nodes: DP=1, each GPU holds 1 expert/layer → ~24GB.
    1.5b)
        echo "--- [0]  Dense baseline,                   1 node  ---"; run 0 0 0 1
        echo "--- [A1] E=8, K=2, EP=1, no All-to-All,   1 node  ---"; run 8 2 1 1
        echo "--- [A2] E=8, K=2, EP=4, NVLink A2A,      1 node  ---"; run 8 2 4 1
        echo "--- [A3] E=8, K=2, EP=8, Slingshot A2A,   2 nodes ---"; run 8 2 8 2
        ;;

    # ── 3b ───────────────────────────────────────────────────────────────────
    # Active: 3b → 19.9B total (6.6x). Highest total/active ratio of all sizes.
    # EP=1 OOMs (~110GB). EP=4 on 1 node borderline (~87GB); use 2 nodes for safety.
    # EP=8 on 2 nodes: DP=1, ~48GB per GPU — comfortable.
    # EP=8 on 4 nodes: DP=2, ~27GB per GPU — includes cross-node at larger DP.
    3b)
        echo "--- [0]  Dense baseline,                   1 node  ---"; run 0 0 0 1
        echo "--- [A1] E=8, K=2, EP=4, NVLink A2A,      2 nodes ---"; run 8 2 4 2
        echo "--- [A2] E=8, K=2, EP=8, Slingshot A2A,   2 nodes ---"; run 8 2 8 2
        echo "--- [A3] E=8, K=2, EP=8, Slingshot A2A,   4 nodes ---"; run 8 2 8 4
        ;;

    # ── 8b ───────────────────────────────────────────────────────────────────
    # Active: 8b → 47.5B total (5.9x). This is the single-GPU throughput challenge target.
    # EP=1 OOMs (~261GB). EP=4 on 1-2 nodes OOMs. Minimum viable: EP=4 + 4 nodes (DP=4, ~83GB).
    # EP=8 on 4 nodes: DP=2, ~72GB per GPU — slightly more comfortable.
    # Both configs involve cross-node All-to-All over Slingshot-11.
    8b)
        echo "--- [0]  Dense baseline,                   1 node  ---"; run 0  0 0 1
        echo "--- [A1] E=8, K=2, EP=4, cross-node A2A,  4 nodes ---"; run 8  2 4 4
        echo "--- [A2] E=8, K=2, EP=8, cross-node A2A,  4 nodes ---"; run 8  2 8 4
        ;;

    *)
        echo "Unknown model size: ${MODEL_SIZE}. Choose: 125m, 350m, 760m, 1.5b, 3b, 8b"
        exit 1
        ;;
esac

echo "=========================================="
echo "Done submitting. Check: squeue --me"
echo "=========================================="
