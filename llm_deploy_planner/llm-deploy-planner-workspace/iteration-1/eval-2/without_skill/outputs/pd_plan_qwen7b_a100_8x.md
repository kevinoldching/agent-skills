# LLM PD Deployment Plan

## 1. Input Summary

| Category | Field | Value |
|----------|-------|-------|
| **Model** | model_name | Qwen-7B |
| | param_count | 7B |
| | hidden_size | 4096 |
| | num_layers | 32 |
| | vocab_size | 151936 |
| | seq_length | 8192 |
| **Hardware** | hardware_type | A100-80GB |
| | cards_per_node | 8 |
| | total_cards | 8 |
| | memory_per_card | 80 GB |
| | interconnect | NVLink |
| **I/O Features** | avg_input_len | 512 tokens |
| | avg_output_len | 128 tokens |
| | peak_batch_size | (to be determined) |
| | input_len_std | Not specified |
| **Scenario** | serving_type | Not specified (default: online) |
| **Performance Target** | target_throughput | Not specified |
| | target_latency | Not specified |
| | target_utilization | Not specified |

---

## 2. Memory Analysis

### Model Memory Footprint

| Component | Calculation | Size |
|-----------|-------------|------|
| **Weights (FP16)** | 7B x 2 bytes | **14 GB** |
| **KV Cache per token** | 2 x 4096 x 32 x 2 bytes | ~0.5 MB per token |
| **KV Cache per token (GQA)** | 2 x 8 KV heads x 128 head_dim x 32 layers x 2 bytes | ~128 KB per token |
| **Activation estimate** | 7B x 2 bytes x 3 (forward pass) | ~42 GB (full) |

### Hardware Constraints

| Constraint | Value | Analysis |
|------------|-------|----------|
| Memory per card | 80 GB | Model fits in single card (14 GB) |
| Model weights vs card memory | 14 GB < 80 GB | TP=1 feasible for model storage |
| **Binding constraint** | Activation memory during prefill | Limits maximum batch |

### Memory Breakdown by Phase

**During Prefill Phase (input=512 tokens):**
- Prefill is **compute-bound** (attention O(n^2) complexity)
- Activation memory per card: ~32 GB at batch 32
- KV cache: ~2 GB at batch 32
- Model weights: 14 GB
- **Total per card at batch 32: ~48 GB** - fits within 80 GB

**During Decode Phase (output=128 tokens):**
- Decode is **memory-bandwidth bound** (逐token生成)
- KV cache per token: ~128 KB
- At batch 32 with 640 total tokens (512+128): ~2.5 GB
- Activation memory is minimal (no O(n^2) attention over full sequence)
- **Total per card at batch 32: ~18 GB** - fits easily

### Maximum Batch Analysis

| Batch Size | Prefill Activation | KV Cache (640 tok) | Model Weights | Total | Feasible? |
|-----------|-------------------|-------------------|---------------|-------|-----------|
| 32 | ~32 GB | ~2.5 GB | 14 GB | ~49 GB | Yes |
| 48 | ~48 GB | ~4 GB | 14 GB | ~66 GB | Yes (tight) |
| 64 | ~64 GB | ~5 GB | 14 GB | ~83 GB | No (exceeds 80 GB) |
| 96 | ~96 GB | ~8 GB | 14 GB | ~118 GB | No |

**Maximum Single-Card Batch (TP=1): ~32-48**

With Tensor Parallel across 8 cards (TP=8), each card handles batch/8 samples:
- Activation per card at batch 64: 64/8 = 8 samples -> ~8 GB activation
- KV cache per card: ~0.6 GB
- Model weights per card (sharded): 14/8 = 1.75 GB
- **Total: ~10.5 GB per card** - very comfortable

**Maximum Concurrent Batch (TP=8 on 8 cards): ~128-256**
(limited by compute throughput, not memory)

---

## 3. PD Strategy Decision

### Decision Criteria Analysis

| Criterion | Value | Threshold | Indicates |
|-----------|-------|-----------|-----------|
| Input length variance | Not specified | - | Cannot use high variance rule |
| avg_input_len vs threshold | 512 < 2048 | 2048 | Mixed signal |
| avg_output_len vs threshold | 128 < 256 | 256 | Mixed signal |
| Prefill/Decode ratio | 512:128 = 4:1 | - | Prefill dominates |
| Input length | 512 (moderate) | - | Not extremely long |
| Output length | 128 (short) | - | Not long decode |

### Prefill vs Decode Workload Analysis

```
Total tokens per request = 512 (input) + 128 (output) = 640 tokens
Prefill portion: 512/640 = 80%
Decode portion: 128/640 = 20%

Prefill is compute-bound (attention quadratic cost)
Decode is memory-bandwidth bound (sequential access)
```

### Recommendation: **PD Mixed (Hybrid)**

**Rationale:**
1. **Output is short (128 tokens)**: Decode phase is only 20% of total work; separation overhead may not be justified
2. **Moderate input (512 tokens)**: Not long enough to warrant dedicated prefill scaling
3. **Prefill:Decode ratio = 4:1**: Prefill dominates but decode is still non-trivial
4. **Simplicity**: PD Mixed with continuous batching achieves good throughput with lower deployment complexity
5. **Memory efficiency**: Hybrid allows dynamic batch composition without maintaining separate model copies

**When PD Separation would be better:**
- Input length > 2048 AND output > 512
- High variance in input length (std > 30% of mean)
- Strict latency SLO requiring independent scaling
- Very long output (decode dominates)

**For this workload (512 in, 128 out), PD Mixed is optimal.**

---

## 4. Deployment Configuration

### PD Mixed Deployment (Recommended)

**Configuration: TP=8, All 8 Cards as Single Instance**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Total GPUs | 8 | Full NVLink topology |
| Tensor Parallel | 8 | Optimal for A100 NVLink |
| Expert Parallel | 1 | Not needed for dense model |
| Data Parallel | 1 | Single instance |
| **Max Concurrent Batch** | **64-128** | Limited by compute, not memory |
| **Recommended Batch** | **32-64** | Good throughput with headroom |
| Continuous Batching | Yes | Dynamic request scheduling |

### Memory Efficiency Analysis (TP=8)

| Component | Per Card (batch=64) | Total (8 cards) |
|-----------|---------------------|-----------------|
| Model Weights (sharded) | 1.75 GB | 14 GB |
| KV Cache | 0.6 GB | 5 GB |
| Activations | 8 GB | 64 GB |
| **Total** | **~10.5 GB** | **~83 GB aggregate** |

With TP=8, memory per card is only ~10.5 GB at batch 64, leaving substantial headroom.

### Alternative: PD Separation (1P1D)

If strict latency SLO requires PD separation:

**Prefill Instance:**
| Parameter | Value |
|-----------|-------|
| Instances | 1 |
| Cards | 8 (TP=8, DP=1) |
| Total Cards Used | 8 |
| Tensor Parallel | 8 |
| Expert Parallel | 1 |
| Data Parallel | 1 |
| Batch Size | 32-64 (compute-bound, can use high batch) |
| Focus | Maximize prefill throughput |

**Decode Instance:**
| Parameter | Value |
|-----------|-------|
| Instances | 1 |
| Cards | 8 (TP=1, DP=8) |
| Total Cards Used | 8 (separate model copy) |
| Tensor Parallel | 1 |
| Expert Parallel | 1 |
| Data Parallel | 8 |
| Batch Size | 32-64 (memory-bandwidth bound) |
| Focus | Memory-efficient decode |

**Total Cards Required: 16** (exceeds 8 available - requires model offloading or reduced config)

**Note:** PD Separation with 1P1D requires double the memory (two model copies) unless using model offloading. With only 8 cards, **PD Separation is NOT recommended** unless:
1. Model offloading is acceptable (higher latency)
2. Prefill and Decode run sequentially (not concurrent)

---

## 5. Throughput & Latency Estimates

### PD Mixed (TP=8, batch=64)

| Metric | Estimate | Notes |
|--------|----------|-------|
| **Prefill Throughput** | ~2000-4000 tok/s | Per instance, 8 cards |
| **Decode Throughput** | ~500-1000 tok/s per card | Memory-bandwidth limited |
| **Total Throughput** | ~8000-16000 tok/s | Depends on input length |
| **TTFT (Time to First Token)** | ~50-100 ms | For 512 input tokens |
| **TPOT (Time Per Output Token)** | ~10-20 ms | For 128 output tokens |
| **GPU Utilization** | 85-95% | Compute-bound phases |

### Concurrent Request Capacity

| Configuration | Max Concurrent Requests | Notes |
|--------------|-------------------------|-------|
| TP=8, batch=32 | 32 requests | ~20 GB per card |
| TP=8, batch=64 | 64 requests | ~10.5 GB per card |
| TP=8, batch=128 | 128 requests | ~20 GB per card (activation spike) |

---

## 6. Summary & Recommendations

### Question 1: Maximum Concurrent Batch?

**Answer: 64-128 concurrent requests**

| Factor | Value |
|--------|-------|
| With TP=8 on 8x A100-80GB | ~64-128 concurrent requests |
| Memory-limited single-card batch | ~32-48 |
| Compute-limited batch (TP=8) | Up to 128+ |

### Question 2: PD Mixed or Separation?

**Answer: PD Mixed (Hybrid)**

| Consideration | Analysis |
|--------------|----------|
| Input/Output ratio | 512:128 = 4:1 (prefill dominant) |
| Input length | 512 (moderate, not extremely long) |
| Output length | 128 (short, decode is 20% of work) |
| Deployment complexity | PD Mixed is simpler |
| Memory efficiency | Better with mixed (single model copy) |

### Final Recommendation

```
Configuration: PD Mixed
Parallel Strategy: TP=8 (all 8 cards)
Batch Size: 32-64 (recommended for production)
Continuous Batching: Enabled
Expected Throughput: 8000-16000 tokens/s
Max Concurrent Requests: 64-128

If strict latency SLO required:
- Consider PD Separation only with model offloading
- Or reserve 4 cards for Prefill, 4 cards for Decode (reduced throughput)
```

---

## Appendix: Key Formulas Used

| Formula | Calculation |
|---------|-------------|
| Model weights (FP16) | 7B x 2 = 14 GB |
| KV cache per token (GQA, 8 KV heads) | 2 x 8 x 128 x 32 x 2 = 128 KB |
| Activation memory (estimate) | param_count x 2 x batch_factor |
| TP=8 sharded weights | 14 GB / 8 = 1.75 GB per card |
| Max batch (memory-bound) | (80 - 14 - 2) / 0.5 = ~128 tokens capacity |
| Max batch (activation-bound) | Limited to ~64 before exceeding 80 GB at prefill |

---

*Generated by LLM PD Deployment Planner Skill*
*Date: 2026-03-21*
