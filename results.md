# Gipfelsturm — Results & Journal

Cluster: CSCS Clariden, GH200 GPUs (95 GB HBM), 4 GPUs/node  
Metric: **tokens/sec/GPU** (median steps 10–50), seq_len=4096, GBS=256

---

## Main Results (Report)

Three scales from the challenge spec:

### Scale 1 — 8B, TP=1, PP=1 (no model parallelism), 1 node / 4 GPUs

| Attention | MBS | Partition | tok/s/GPU | W&B run |
|-----------|-----|-----------|-----------|---------|
| Softmax (baseline, TE) | 2 | debug | **10,800** (steps 37–47 stable at ~24.2s; steps 48–50 outliers due to Triton recompile) | throughput-8b-1n-2255003 |
| Linear attention (local, TP=4) | 1 | debug | ~6,200 (job killed at step 38 by 30-min limit; JIT still warming up, high variance 4,471–6,514) | throughput-8b-1n-2255123 |
| Mamba | — | — | TODO | — |
| xLSTM | — | — | TODO | — |

### Scale 2 — 32B, TP=4

> **Note on linear attention OOM (1 node):** With TP=4 on a single node, DP=1 — there is only one data-parallel worker so the distributed optimizer cannot shard optimizer states. Each GPU must hold the full optimizer state for its TP shard, pushing memory to 90.5 GB / 95 GB and causing OOM. Running on 2 nodes gives TP=4, DP=2: the optimizer shards across the two DP workers, halving per-GPU optimizer memory and fitting within 95 GB. Linear attention results below are therefore from 2-node runs.

| Attention | Nodes | MBS | Partition | tok/s/GPU | W&B run |
|-----------|-------|-----|-----------|-----------|---------|
| Softmax (baseline, TE) | 1 | 1 | debug | **3,200** (steps 31–50 stable ~81s/step, 438–444 TFLOP/s/GPU) | throughput-32b-1n-2255615 |
| Linear attention (local, TP=4) | 2 | 1 | normal | **3,200** (steps 31–37 stable 3,154–3,217; steps 38–50 high variance 2,560–2,962 due to Triton recompile) | throughput-32b-2n-2259732 |
| Mamba | — | — | — | TODO | — |
| xLSTM | — | — | — | TODO | — |
| Mamba | — | — | TODO | — |
| xLSTM | — | — | TODO | — |

### Scale 3 — 140B, TP=4, PP=4, multi-node

| Attention | Nodes | GPUs | Partition | tok/s/GPU | W&B run |
|-----------|-------|------|-----------|-----------|---------|
| Softmax (baseline, TE) | — | — | normal | TODO | — |
| Linear attention | — | — | normal | TODO | — |
| Mamba | — | — | normal | TODO | — |
| xLSTM | — | — | normal | TODO | — |

---

## Scalability Plot (Report)

Winner (TBD after Scale 1 results). Test winner at 1→2→4 nodes.

| Model | Attention | Nodes | GPUs | Partition | tok/s/GPU |
|-------|-----------|-------|------|-----------|-----------|
| TBD   | TBD       | 1     | 4    | debug     | — |
| TBD   | TBD       | 2     | 8    | debug     | — |
| TBD   | TBD       | 4     | 16   | normal    | — |

---

## Loss Curves (Report)

Compare baseline vs winner at same model size, ~2000 steps.  
Needs non-debug partition (time limit >30 min).

| Model | Attention | Steps | Partition | Status |
|-------|-----------|-------|-----------|--------|
| TBD | Softmax | 2000 | normal | TODO |
| TBD | Winner  | 2000 | normal | TODO |

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
| 2026-05-15 | 2255003 | 8b | Softmax (TE) MBS=2 | 1 | debug | 10,800 | Steps 37–47 stable; steps 48–50 outliers (Triton recompile spike at step 48) |
| 2026-05-15 | 2255123 | 8b | Linear (local, TP=4) MBS=1 | 1 | debug | ~6,200 | Killed at step 38 by 30-min limit; JIT still settling, range 4,471–6,514 |
| 2026-05-15 | 2255615 | 32b | Softmax (TE, TP=4) MBS=1 | 1 | debug | 3,200 | Steps 31–50 stable; ~81s/step, 438–444 TFLOP/s/GPU |
| 2026-05-15 | 2255826 | 32b | Linear (local, TP=4) MBS=1 | 1 | debug | OOM | 90.5 GB / 95 GB; distributed optimizer can't shard with DP=1; needs 2 nodes |
| 2026-05-17 | 2259732 | 32b | Linear (local, TP=4) MBS=1 | 2 | normal | 3,200 | Steps 31–37 stable; matches baseline — at 32B scale FFN dominates, attention type has minimal throughput impact |
