#!/bin/bash
#
# MoE throughput sweep — Challenge 2 (architecture / topology co-design study)
#
# Usage:  STEPS=100 ./scripts/sweep_throughput_moe.sh
#
# Each run is one line. Comment out any line to skip it.
# Sections are separated by echo banners — those can also be commented out.
#
# 3B compute-matched FFN (dense=8192):  top-2→4096  top-4→2048  top-8→1024
# 8B compute-matched FFN (dense=14336): top-2→7168  top-4→3584  top-8→1792
#
# Memory notes (weights + distributed-optimizer, activations excluded):
#   3B  64E EP=4  4n DP=4 : ~48 GB/GPU  ✓    3B  64E EP=2  4n DP=8 : ~66 GB/GPU  ✓
#   3B  64E EP=8  4n DP=2 : ~41 GB/GPU  ✓    3B  64E EP=16 4n DP=1 : ~42 GB/GPU  ✓
#   3B 128E EP=8  4n DP=2 : ~75 GB/GPU  ✓    3B  64E EP=4  1n DP=1 : ~130GB/GPU  ✗ OOM
#   8B  16E EP=4  4n DP=4 : ~36 GB/GPU  ✓    8B  64E EP=8  8n DP=4 : ~61 GB/GPU  ✓
#   8B  64E EP=4  4n DP=4 : ~112GB/GPU  ✗ OOM
#
# EP topology note:
#   EP≤4 : A2A stays intra-node (NVLink ~200 GB/s) at any node count (4 GPUs/node)
#   EP≥8 : A2A crosses nodes (Slingshot-11 ~93 GB/s)

set -uo pipefail

STEPS=${STEPS:-50}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH="${SCRIPT_DIR}/../launch.sh"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
run_dense() {
    # run_dense PROJECT SIZE NODES STEPS [PARTITION]
    local proj=$1 size=$2 nodes=$3 steps=$4 partition=${5:-}
    SLURM_PARTITION="${partition}" PROJECT_NAME="${proj}" \
        "${LAUNCH}" throughput "${size}" "${steps}" "${nodes}"
}

run_moe() {
    # run_moe PROJECT NE K EP NODES FFN LFREQ SIZE STEPS [PARTITION]
    local proj=$1 ne=$2 k=$3 ep=$4 nodes=$5 ffn=$6 lfreq=$7 size=$8 steps=$9 partition=${10:-}
    SLURM_PARTITION="${partition}" PROJECT_NAME="${proj}" \
    NUM_EXPERTS="${ne}" MOE_TOPK="${k}" EP_SIZE="${ep}" \
    MOE_FFN="${ffn}" MOE_LAYER_FREQ="${lfreq}" \
        "${LAUNCH}" throughput "${size}" "${steps}" "${nodes}"
}

# ===========================================================================
# A. Smoke tests  —  debug partition, 30 steps, 760m
#    Not part of the main story. Run these first to catch config bugs cheaply.
# ===========================================================================
echo "=== A. Smoke ==="
run_dense  moe-smoke   760m  1  30  debug                                          # [1]  760m dense
run_moe    moe-smoke   64 4  4  1  1024  1   760m  30  debug                       # [2]  760m 64E top-4 EP=4 FFN=1024

# ===========================================================================
# B. 3B architecture  —  4 nodes throughout, EP=4 unless noted
#    Question: how do expert count, top-k, FFN size, and layer frequency
#    affect throughput at fixed topology?
#    Run #6 is the main modern candidate and the anchor for section C.
#    Run #9 uses EP=8 (not EP=4) because 128E EP=4 4n exceeds memory (~120 GB/GPU).
# ===========================================================================
echo "=== B. 3B architecture ==="
run_dense  moe-3b-arch  3b    4  "${STEPS}"                                        # [3]  3B dense baseline
run_moe    moe-3b-arch   8 2  4  4  4096  1   3b  "${STEPS}"                       # [4]  3B  8E top-2 EP=4 FFN=4096 (Mixtral-style)
run_moe    moe-3b-arch  64 2  4  4  4096  1   3b  "${STEPS}"                       # [5]  3B 64E top-2 EP=4 FFN=4096 (more experts, same top-k)
run_moe    moe-3b-arch  64 4  4  4  2048  1   3b  "${STEPS}"                       # [6]  3B 64E top-4 EP=4 FFN=2048 (main modern candidate) ← topology anchor
run_moe    moe-3b-arch  64 8  4  4  1024  1   3b  "${STEPS}"                       # [7]  3B 64E top-8 EP=4 FFN=1024 (high routing pressure)
run_moe    moe-3b-arch  64 4  4  4  2048  2   3b  "${STEPS}"                       # [8]  3B 64E top-4 EP=4 FFN=2048 layerfreq=2
run_moe    moe-3b-arch 128 4  8  4  2048  1   3b  "${STEPS}"                       # [9]  3B 128E top-4 EP=8 FFN=2048 (EP=8 for memory)

# ===========================================================================
# C. 3B topology  —  architecture fixed: 64E top-4 FFN=2048
#
# Story 1 — EP sweep (4 nodes fixed):
#   EP=2 → EP=4 [=#6] → EP=8 → EP=16
#   EP≤4: A2A intra-node (NVLink).  EP≥8: A2A cross-node (Slingshot-11).
#
# Story 2 — Node scaling (EP=4 fixed):
#   dense 1n [=#13] → MoE 2n → MoE 4n [=#6] → MoE 8n
#   EP=4 always fits its expert group on one node, so A2A stays on NVLink
#   regardless of total node count. Dense 1n replaces MoE 1n (which OOMs).
# ===========================================================================
echo "=== C. 3B topology ==="
# EP sweep — 4 nodes fixed
# EP=4 anchor re-submitted here so moe-3b-topo W&B project has a complete EP comparison.
run_moe    moe-3b-topo  64 4   4  4  2048  1   3b  "${STEPS}"                      # [6'] 3B 64E top-4 EP=4  4n (anchor — same arch as #6, topo project)
run_moe    moe-3b-topo  64 4   2  4  2048  1   3b  "${STEPS}"                      # [10] 3B 64E top-4 EP=2  4n (intra-node, DP=8)
run_moe    moe-3b-topo  64 4   8  4  2048  1   3b  "${STEPS}"                      # [11] 3B 64E top-4 EP=8  4n (cross-node,  DP=2)
run_moe    moe-3b-topo  64 4  16  4  2048  1   3b  "${STEPS}"                      # [12] 3B 64E top-4 EP=16 4n (cross-node,  DP=1)
# Node scaling — EP=4 fixed  (#6 = 4n anchor; MoE 1n OOMs ~130 GB/GPU)
run_dense  moe-3b-topo  3b    1  "${STEPS}"                                        # [13] 3B dense 1n (DP-scaling reference)
run_moe    moe-3b-topo  64 4   4  2  2048  1   3b  "${STEPS}"                      # [14] 3B 64E top-4 EP=4 2n (EP group=1 node, DP=2)
run_moe    moe-3b-topo  64 4   4  8  2048  1   3b  "${STEPS}"                      # [15] 3B 64E top-4 EP=4 8n (EP group=1 node, DP=8)

# ===========================================================================
# D. 8B validation  —  checks whether 3B conclusions hold at larger scale
#    8B + 64E needs EP=8 on 8 nodes (DP=4) to fit (~61 GB/GPU).
#    8B + 16E at EP=4 on 4 nodes is comfortable (~36 GB/GPU).
# ===========================================================================
echo "=== D. 8B validation ==="
run_dense  moe-8b-val   8b    4  "${STEPS}"                                        # [16] 8B dense 4n
run_moe    moe-8b-val    8 2  4  4  7168  1   8b  "${STEPS}"                       # [17] 8B  8E top-2 EP=4 FFN=7168 4n (coarse)
run_moe    moe-8b-val   16 4  4  4  3584  1   8b  "${STEPS}"                       # [18] 8B 16E top-4 EP=4 FFN=3584 4n (fine-grained at EP=4)
run_moe    moe-8b-val   16 4  4  4  3584  2   8b  "${STEPS}"                       # [19] 8B 16E top-4 EP=4 FFN=3584 layerfreq=2 4n
run_moe    moe-8b-val   64 4  8  8  3584  1   8b  "${STEPS}"                       # [20] 8B 64E top-4 EP=8 FFN=3584 8n (EP=4 4n OOMs)

echo ""
echo "Done submitting. Check: squeue --me"
