# Gipfelsturm — Results & Journal

Cluster: CSCS Clariden, GH200 GPUs (95 GB HBM), 4 GPUs/node  
Metric: **tokens/sec/GPU** (median steps 30–45), seq_len=4096, GBS=256

---

## Main Results (Report)

Three scales from the challenge spec:

### Scale 1 — 8B, TP=1, PP=1 (no model parallelism), 1 node / 4 GPUs

| Attention | MBS | Partition | tok/s/GPU | W&B run |
|-----------|-----|-----------|-----------|---------|
| Softmax (baseline, TE) | 2 | debug | **10,800** (steps 37–47 stable ~24.2s; steps 48–50 Triton recompile outliers) | throughput-8b-1n-2255003 |
| Linear attention (local, TP=4) | 1 | debug | **6,600** (steps 30–45 stable 6,562–6,687; steps 46–50 Triton outliers) | throughput-8b-1n-linear-2291911 |
| Mamba | 1 | debug | **6,050** (steps 36–48 stable 5,961–6,056; spikes at 31/33/35 Triton recompile) | throughput-8b-1n-2292147 |
| xLSTM | 1 | debug | **6,150** (steps 26–43 stable 6,126–6,186; outlier spikes at 31/40/44/45) | throughput-8b-1n-2293280 |

### Scale 2 — 32B, TP=4

> **Note on linear attention OOM (1 node):** With TP=4 on a single node, DP=1 — the distributed optimizer cannot shard, pushing memory to 90.5 GB / 95 GB and causing OOM. Running on 2 nodes gives DP=2, halving per-GPU optimizer memory. Linear attention results are therefore from 2-node runs.

| Attention | Nodes | MBS | Partition | tok/s/GPU | W&B run |
|-----------|-------|-----|-----------|-----------|---------|
| Softmax (baseline, TE) | 1 | 1 | debug | **3,200** (steps 31–50 stable ~81s/step, 438–444 TFLOP/s/GPU) | throughput-32b-1n-2255615 |
| Linear attention (local, TP=4) | 2 | 1 | normal | **3,200** (steps 31–37 stable 3,154–3,217; steps 38–50 Triton recompile) | throughput-32b-2n-2259732 |
| Mamba | — | — | — | TODO | — |
| xLSTM | — | — | — | TODO | — |

### Scale 3 — 140B, TP=4, PP=4, 8 nodes / 32 GPUs

| Attention | MBS | Partition | tok/s/GPU | W&B run |
|-----------|-----|-----------|-----------|---------|
| Softmax (baseline, TE) | 1 | normal | TODO | — |
| Linear attention | 1 | normal | TODO | — |
| Mamba | 1 | normal | TODO | — |
| xLSTM | 1 | normal | TODO | — |

---

## Journal (All Runs)

Exploratory runs, smoke tests, debugging — not in main report.

| Date | Job | Model | Attention | Nodes | Partition | tok/s/GPU | Notes |
|------|-----|-------|-----------|-------|-----------|-----------|-------|
| 2026-05-13 | 2174965 | 125m | Softmax (TE) | 1 | debug | ~38,500 | First baseline smoke test |
| 2026-05-13 | 2175443 | 125m | Linear (local) | 1 | debug | — | Crashed: persist_layer_norm error |
| 2026-05-13 | 2175862 | 125m | Linear (local) | 1 | debug | — | OOM: patch not applied correctly |
| 2026-05-13 | 2176795 | 125m | Linear (local) | 1 | debug | ~32,000 | First successful linear attention run |
| 2026-05-13 | 2177043 | 8b | Linear (local) | 1 | debug | — | OOM: 8B too large for local impl |
| 2026-05-13 | 2177088 | 8b | Linear (local) MBS=1 | 1 | debug | — | OOM: still too large |
| 2026-05-13 | 2178279 | 3b | Linear (local) MBS=2 | 1 | debug | 20,690 | 3B linear attention working |
| 2026-05-14 | 2236096 | 3b | Softmax (TE) MBS=4 | 1 | debug | 12,207 | 3B baseline — linear is 69% faster |
| 2026-05-15 | 2255003 | 8b | Softmax (TE) MBS=2 | 1 | debug | 10,800 | Steps 37–47 stable; steps 48–50 Triton recompile outliers |
| 2026-05-15 | 2255123 | 8b | Linear (local, TP=4) MBS=1 | 1 | debug | ~6,200 | Killed at step 38 by time limit; JIT still settling |
| 2026-05-15 | 2255615 | 32b | Softmax (TE, TP=4) MBS=1 | 1 | debug | 3,200 | Steps 31–50 stable; ~81s/step, 438–444 TFLOP/s/GPU |
| 2026-05-15 | 2255826 | 32b | Linear (local, TP=4) MBS=1 | 1 | debug | OOM | 90.5 GB / 95 GB; DP=1 can't shard optimizer; needs 2 nodes |
| 2026-05-17 | 2259732 | 32b | Linear (local, TP=4) MBS=1 | 2 | normal | 3,200 | Steps 31–37 stable; matches baseline — FFN dominates at 32B |
| 2026-05-18 | 2291911 | 8b | Linear (local, TP=4) MBS=1 | 1 | debug | 6,600 | Clean 50-step run; steps 30–45 stable 6,562–6,687 |
| 2026-05-18 | 2292147 | 8b | Mamba MBS=1 | 1 | debug | 6,050 | Steps 36–48 stable 5,961–6,056 |
| 2026-05-18 | 2293280 | 8b | xLSTM MBS=1 | 1 | debug | 6,150 | Steps 26–43 stable 6,126–6,186 |
| 2026-05-18 | 2292236 | 140b | Softmax (TE, TP=4, PP=4) MBS=1 | 8 | normal | TODO | Baseline, queued ~23:00 |
| 2026-05-18 | 2296208 | 3b | Mamba MBS=1 | 1 | debug | 16,714 | Steps 30–45 stable 16,707–16,717; grad norm nan (expected at init) |
