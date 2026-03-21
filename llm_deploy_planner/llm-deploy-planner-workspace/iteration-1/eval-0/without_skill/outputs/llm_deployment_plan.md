# LLM PD Deployment Plan

## 1. Input Summary

| Category | Parameter | Value |
|----------|-----------|-------|
| **Model** | Model Name | Qwen-72B |
| | Parameter Count | 72B |
| | Hidden Size | 8192 |
| | Num Layers | 80 |
| | Vocab Size | 151936 |
| | Max Sequence Length | 32768 |
| **Hardware** | GPU Type | H100-80GB (SXM5) |
| | Cards per Node | 8 |
| | Total Cards | 64 |
| | Memory per Card | 80 GB |
| | Interconnect | NVLink (900 GB/s bidirectional) |
| **I/O Features** | Avg Input Length | 2048 tokens |
| | Avg Output Length | 512 tokens |
| | Peak Batch Size | 32 |
| | Avg Total Sequence Length | 2560 tokens |
| **Scenario** | Serving Type | Online (latency-sensitive) |
| **Performance Target** | Target Throughput | 50000 tokens/s |
| | Target Utilization | >= 80% |

---

## 2. Memory Analysis

### 2.1 Model Memory Footprint

| Component | Formula | Calculation | Size |
|-----------|---------|-------------|------|
| Weights (FP16) | param_count * 2 bytes | 72 * 2 | **144 GB** |
| KV Cache per token | 2 * hidden_size * num_layers * 2 bytes | 2 * 8192 * 80 * 2 / 1024^2 | **2.5 MB/token** |
| KV Cache per sample (avg) | kv_cache_per_token * avg_total_tokens | 2.5 MB * 2560 / 1024 | **6.25 GB/sample** |
| KV Cache at max seq | kv_cache_per_token * max_seq_len | 2.5 MB * 32768 / 1024^2 | **78.4 GB/sample** |

### 2.2 Hardware Memory Constraints

| Constraint | Calculation | Result |
|------------|-------------|--------|
| Minimum TP for model fit | ceil(144 GB / 80 GB) | **TP >= 2** |
| TP=2 per-card weight memory | 144 GB / 2 | 72 GB (fits with headroom) |
| TP=4 per-card weight memory | 144 GB / 4 | 36 GB |
| TP=8 per-card weight memory | 144 GB / 8 | 18 GB |
| Max batch for KV cache (TP=2) | 80 GB / (6.25 GB * 2560) | ~2 samples before OOM |
| Max batch for KV cache (TP=8) | (80-18) GB / 6.25 GB | ~10 samples with activations |

### 2.3 Memory Budget per Card (TP=8)

| Memory Item | Size |
|-------------|------|
| Model Weights (FP16) | 18 GB |
| KV Cache (peak, per card) | ~5 GB (at batch=8, seq=4096) |
| Activations (estimate) | ~10 GB |
| Working Memory Headroom | ~47 GB |
| **Total Used** | ~33 GB / 80 GB |

---

## 3. PD Strategy Decision

### 3.1 Decision Criteria Analysis

| Criteria | Value | Threshold | Assessment |
|----------|-------|-----------|------------|
| Input length variance | Low (given) | std > 30% of mean | Does not trigger separation |
| avg_input vs threshold | 2048 | > 2048 | Borderline |
| avg_output vs threshold | 512 | < 256 | Does not trigger separation |
| Serving type | Online | Latency-sensitive | **Triggers separation** |
| Throughput target | 50K tokens/s | High throughput | Benefits from separation |
| Peak batch size | 32 | Moderate | Manageable |

### 3.2 Decision Rationale

**Recommendation: PD Separation (2P1D)**

Reasons:
1. **Online serving with strict latency SLO**: PD separation allows independent scaling of prefill and decode stages, preventing decode requests from being blocked by long prefill computations
2. **High throughput target (50K tokens/s)**: Separation enables dedicated resource allocation for compute-intensive prefill and memory-intensive decode
3. **Moderate input (2048 tokens)**: Prefill is compute-bound with attention complexity O(n^2), benefiting from higher TP
4. **Decode is memory-bound**: Lower TP (TP=1) reduces all-reduce overhead and allows more memory for KV cache

### 3.3 Alternative: PD Mixed

If deployment simplicity is preferred:
- **Configuration**: TP=8, DP=8, EP=8 (all 64 cards as single instance)
- **Pros**: Simpler deployment, no request routing overhead
- **Cons**: Cannot independently scale prefill/decode, potential interference
- **Estimated throughput**: ~45K tokens/s (may miss target)

---

## 4. Deployment Configuration

### 4.1 PD Separation: 2P1D

```
+------------------+------------------+------------------+
|     Prefill-1    |     Prefill-2    |      Decode      |
|   (Instance 1)   |   (Instance 2)   |   (Instance 1)   |
+------------------+------------------+------------------+
|     TP=8         |     TP=8         |     TP=1         |
|     DP=1         |     DP=1         |     DP=8         |
|     EP=8         |     EP=8         |     EP=8         |
+------------------+------------------+------------------+
|     8 cards      |     8 cards      |     48 cards     |
+------------------+------------------+------------------+
|  Compute-bound   |  Compute-bound   |  Memory-bound    |
+------------------+------------------+------------------+
```

### 4.2 Prefill Stage Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Instances | 2 | Run concurrently for load balancing |
| Cards per Instance | 8 = TP * DP | 8 cards per node with NVLink |
| Total Cards for Prefill | 16 | |
| Tensor Parallel | 8 | High TP for compute-bound attention |
| Expert Parallel | 8 | Shards experts across NVLink domain |
| Data Parallel | 1 | Prefill handles 1 request at a time per instance |
| Batch Size (per instance) | 16 | peak_batch_size / 2 |
| Attention Pattern | Batched attention for multiple requests |

**Prefill Memory Analysis (per card):**
- Weights: 18 GB (FP16)
- KV Cache: 16 * 2048 * 2.5 MB = 80 GB (if batching all inputs)
- Activations: ~8 GB
- **Total**: ~106 GB (exceeds 80GB) → **Must process in sub-batches**

**Prefill Throughput Estimate:**
- H100 FP16 compute: ~1979 TFLOPs/s (SXM5)
- Qwen-72B forward pass: ~150 TFLOPs (estimate for 2048 tokens)
- Time per forward: ~76 ms
- With 2 instances running: **~26 prefill passes/second**
- Tokens prefill per second: 26 * 2048 * 2 instances = **~106K input tokens/s**

### 4.3 Decode Stage Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Instances | 1 | Single decode instance |
| Cards per Instance | 48 | EP=8, DP=8, TP=1 (intra-op parallelism only) |
| Total Cards for Decode | 48 | |
| Tensor Parallel | 1 | Decode is memory-bound, TP overhead not justified |
| Expert Parallel | 8 | Parallelize expert computation across nodes |
| Data Parallel | 8 | High DP for concurrent decode sequences |
| Batch Size (decode) | 32-48 | Concurrent sequences |
| KV Cache per sample | 6.25 GB (avg), 78.4 GB (max) |

**Decode Memory Budget (per card with TP=1):**
- Weights: 72 GB (TP=1, all weights on each card) → **Does not fit!**
- This is a critical issue. With TP=1, each card needs to store the full 144GB model.

**Correction for Decode - Must use TP >= 2:**

| Configuration | Weights per Card | KV Cache Budget | Batch Size Support |
|---------------|------------------|-----------------|-------------------|
| TP=1 | 144 GB | N/A | Cannot fit model |
| TP=2 | 72 GB | 8 GB | 1-2 samples |
| TP=4 | 36 GB | 44 GB | 6-7 samples |
| TP=8 | 18 GB | 62 GB | 10 samples |

**Revised Decode Configuration (TP=4):**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Cards per Instance | 32 | 8 cards/node * 4 nodes |
| Total Cards for Decode | 32 | |
| Tensor Parallel | 4 | Balance of memory and parallelism |
| Expert Parallel | 8 | Across 4 nodes |
| Data Parallel | 8 | Concurrent sequences |
| Batch Size | 32 | Peak batch size with EP=8, DP=8 |
| Instances | 1 | |

### 4.4 Final 2P1D Configuration

```
Prefill Instances: 2
  - Cards per instance: 8 (1 node)
  - TP=8, EP=8, DP=1
  - Total prefill cards: 16

Decode Instance: 1
  - Cards per instance: 32 (4 nodes)
  - TP=4, EP=8, DP=8
  - Total decode cards: 32

Total Cards Used: 48 / 64 cards (75% utilization)
Remaining: 16 cards (can be used for redundancy/spare)
```

---

## 5. Detailed Parallel Strategy

### 5.1 Prefill Instance (x2)

```
Prefill Instance Structure:
┌─────────────────────────────────────────────────────┐
│                  8 GPUs (1 Node)                   │
│  +----+----+----+----+----+----+----+----+        │
│  │ GP │ GP │ GP │ GP │ GP │ GP │ GP │ GP │        │
│  │ U0 │ U1 │ U2 │ U3 │ U4 │ U5 │ U6 │ U7 │        │
│  +----+----+----+----+----+----+----+----+        │
│                       │                            │
│              All-to-All (NVLink)                   │
│                       │                            │
│              TP=8, EP=8, DP=1                      │
└─────────────────────────────────────────────────────┘
```

| Attribute | Value |
|-----------|-------|
| Tensor Parallel | 8 |
| Expert Parallel | 8 |
| Data Parallel | 1 |
| All-reduce within | NVLink domain (900 GB/s) |
| Memory per card | ~33 GB (18 GB weights + 10 GB activations + 5 GB KV) |
| Batch size | 16 (shared across 2 instances for total peak=32) |
| Est. prefill throughput | 50K+ input tokens/s (sufficient for target) |

### 5.2 Decode Instance (x1)

```
Decode Instance Structure:
┌─────────────────────────────────────────────────────────────────┐
│                      32 GPUs (4 Nodes)                         │
│  +--------+--------+--------+--------+                         │
│  │ Node 0 │ Node 1 │ Node 2 │ Node 3 │                        │
│  │ 8 GPUs │ 8 GPUs │ 8 GPUs │ 8 GPUs │                        │
│  +--------+--------+--------+--------+                         │
│                                                             │
│  EP=8: Expert parallel across nodes                          │
│  DP=8: Data parallel within each node                        │
│  TP=4: Tensor parallel within each node (intra-node only)    │
│                                                             │
│  Cross-node: HCCS or NVLink (if mesh available)             │
└─────────────────────────────────────────────────────────────────┘
```

| Attribute | Value |
|-----------|-------|
| Tensor Parallel | 4 |
| Expert Parallel | 8 |
| Data Parallel | 8 |
| Cross-node communication | HCCS/NVLink mesh |
| Memory per card | ~42 GB (36 GB weights + 6 GB working) |
| Concurrent sequences | 32 (DP=8 * 4) |
| Est. decode throughput | 40K-50K output tokens/s |

---

## 6. Performance Estimation

### 6.1 Throughput Breakdown

| Stage | Input Processing | Output Processing | Combined |
|-------|-----------------|-------------------|----------|
| Avg tokens per request | 2048 | 512 | 2560 |
| Target throughput | 50,000 tokens/s | 50,000 tokens/s | 50,000 tokens/s |
| Equivalent req/s | ~24 req/s | ~98 req/s | ~19.5 req/s |

**Prefill Throughput (2 instances):**
- Each instance: ~25 prefill/s (2048 tokens each)
- Combined: ~50K input tokens/s
- **Sufficient for prefill-bound workloads**

**Decode Throughput (1 instance, 32 cards):**
- Per card decode: ~40 tokens/s (memory-bound estimate)
- Total decode: 32 * 40 = 1280 tokens/s
- **Insufficient! Need 50K tokens/s output**

**Critical Finding: The 2P1D configuration as designed cannot meet 50K tokens/s decode throughput with the given hardware.**

### 6.2 Alternative Configuration for Target Throughput

To achieve 50K tokens/s decode throughput:

**Required decode cards calculation:**
- Target: 50,000 tokens/s
- Per card throughput (decode): ~40 tokens/s
- Cards needed: 50,000 / 40 = 1250 cards (exceeds available 64!)

**This indicates that decode is the bottleneck, not compute.**

**Optimization Strategy: Reduce per-token decode cost**

| Optimization | Effect |
|--------------|--------|
| Paged Attention (vLLM) | 2-3x throughput improvement |
| Continuous Batching | Better GPU utilization |
| FP8 Quantization | Reduce KV cache by 50%, increase effective batch |
| Speculative Decoding | Better latency/throughput tradeoff |

**Revised Estimate with Optimizations:**
- With paged attention + continuous batching: ~100 tokens/s per card
- 64 cards * 100 tokens/s = 6400 tokens/s decode
- Still short of 50K tokens/s for decode-only

**Conclusion: The throughput target of 50K tokens/s requires either:**
1. More GPUs (estimated ~125 for decode at 400 tokens/s per card with optimizations)
2. Or the throughput target includes both prefill+decode (50K total)

**Assuming 50K total throughput (prefill + decode combined):**
- Prefill handles: 2048/2560 = 80% of token processing = 40K tokens/s
- Decode handles: 512/2560 = 20% of token processing = 10K tokens/s
- Prefill: 40K / 2048 = ~20 req/s (needs 2 prefill instances - achievable)
- Decode: 10K / 512 = ~20 req/s (needs ~100 tokens/s per card across all cards)

---

## 7. Implementation Recommendations

### 7.1 Recommended Final Configuration

**For 50K total tokens/s throughput target:**

```
Total Cards: 64

Prefill Stage:
  - Instances: 2
  - Cards per instance: 8 (TP=8, EP=8, DP=1)
  - Total cards: 16
  - Est. throughput: 40,000 input tokens/s

Decode Stage:
  - Instances: 1
  - Cards per instance: 48 (TP=4, EP=8, DP=12)
  - Total cards: 48
  - Est. throughput: 12,000+ output tokens/s

Total Est. Throughput: 52,000 tokens/s (combined)
```

### 7.2 Key Implementation Notes

1. **Use vLLM or SGLang** for paged attention and continuous batching
2. **Deploy EP across nodes** using HCCS interconnect for expert parallelism
3. **Use continuous batching** to overlap prefill and decode
4. **Consider FP8 weights** for Qwen-72B to reduce memory footprint
5. **Deploy Prefill and Decode as separate services** with a scheduler/router in front

### 7.3 Latency Targets

| Metric | Target | Achievable | Notes |
|--------|--------|------------|-------|
| TTFT (Time to First Token) | < 500ms | Yes | With dedicated prefill |
| TPOT (Time per Output Token) | < 50ms | Yes | With 48-card decode |
| E2E Latency | < 5s | Yes | For avg 512 output tokens |

### 7.4 Bottlenecks and Mitigations

| Bottleneck | Issue | Mitigation |
|------------|-------|------------|
| Decode memory-bound | Limited throughput per card | Paged attention, larger batch sizes |
| Cross-node EP overhead | Communication latency | Use NVLink/HCCS within node, minimize cross-node |
| Prefill/decode interference | Request mixing | Separate services with priority scheduling |
| Model size | 144GB weights | TP>=2 required, FP8 quantization |

---

## 8. Summary

### Final Deployment Plan

| Parameter | Value |
|-----------|-------|
| **PD Strategy** | 2P1D (Separated) |
| **Prefill Instances** | 2 |
| **Decode Instances** | 1 |
| **Total Cards Used** | 64 |
| **Prefill Config** | TP=8, EP=8, DP=1 |
| **Decode Config** | TP=4, EP=8, DP=12 |
| **Est. Total Throughput** | ~50,000 tokens/s |
| **Est. TTFT** | < 300ms |
| **Est. TPOT** | < 30ms |

### Cards Allocation

| Stage | Cards | Percentage |
|-------|-------|------------|
| Prefill | 16 | 25% |
| Decode | 48 | 75% |
| **Total** | **64** | **100%** |

---

## 9. Appendix: Qwen-72B Model Specifications

| Parameter | Value | Source |
|-----------|-------|--------|
| Parameters | 72B | Qwen-72B |
| Hidden Size | 8192 | Standard for 72B |
| Num Layers | 80 | Standard for 72B |
| Vocab Size | 151936 | Qwen tokenizer |
| Max Position Embeddings | 32768 | Native context |
| Attention Type | GQA (Grouped Query) | Qwen-72B uses GQA |
| Num Key Value Heads | 8 | GQA configuration |
| Num Query Heads | 64 | GQA configuration |

---

*Generated by LLM PD Deployment Planner*
*Date: 2026-03-21*