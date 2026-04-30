# LLM PD Deployment Plan

## 1. Input Summary

| Category | Field | Value |
|----------|-------|-------|
| **Model** | Model Name | Qwen-72B |
| | Parameter Count | 72B |
| | Model Type | Dense (not MoE) |
| | Architecture | Transformer with GQA |
| | Hidden Size | 8192 |
| | Num Layers | 80 |
| | Vocab Size | 151936 |
| | Max Sequence Length | 32768 |
| **Hardware** | Hardware Type | H100-80GB (HBM3) |
| | Cards Per Node | 8 |
| | Total Cards | 64 (8 nodes) |
| | Memory Per Card | 80 GB |
| | Interconnect | NVLink (900 GB/s) |
| **I/O Features** | Avg Input Length | 2048 tokens |
| | Avg Output Length | 512 tokens |
| | Peak Batch Size | 32 |
| **Scenario** | Serving Type | Online Inference |
| | Latency Requirement | Latency-sensitive |
| **Performance Target** | Target Throughput | 50,000 tokens/s |

---

## 2. PD Strategy Decision

### Decision Result
**Recommended Strategy: PD Separation**

### Decision Framework

| Priority | Factor | Value | Analysis |
|----------|--------|-------|----------|
| **1st: Scenario** | Online Inference | Latency-sensitive | PD Separation preferred for minimizing TTFT and TPOT |
| **2nd: Length Ratio** | α = S_in/(S_in+k×S_out) | α = 2048/(2048+3×512) = 0.572 | 0.572 < 0.8, mix acceptable, but scenario dominates |
| **2nd: Imbalance** | max(α, 1-α) | 0.572 | Moderate imbalance, separation viable |
| **3rd: SLO** | TPS=50K + Latency-sensitive | 50K tokens/s + low TTFT/TPOT | PD Separation achieves lower latency |

### Decision Rationale

1. **Scenario Priority**: Online inference with latency sensitivity strongly favors PD Separation
   - PD Separation allows independent scaling of Prefill and Decode stages
   - Prefill can use larger batches for throughput, Decode can optimize for low latency
   - TTFT (Time To First Token) is critical for user experience

2. **Length Ratio Analysis**:
   - α = 0.572 indicates moderately imbalanced input/output ratio
   - With k=3: 2048 / (2048 + 1536) = 0.572
   - This level of imbalance does not preclude separation

3. **SLO Priority**:
   - Target TPS (50,000) is high, requiring significant parallelism
   - Latency-sensitive nature requires separate optimization paths
   - PD Separation allows Prefill to use high TP for compute-bound attention
   - PD Separation allows Decode to use high DP for memory-bandwidth optimization

---

## 3. Memory Analysis

### Model Memory Footprint

| Component | Calculation | Value |
|-----------|-------------|-------|
| **Model Weights (FP16)** | 72B params × 2 bytes | 144 GB |
| **Attention Heads** | 8 KV heads (GQA), 128 query heads | - |
| **Hidden Size** | 8192 | - |
| **Layer Count** | 80 | - |

### KV Cache Memory Calculation

| Parameter | Formula | Value |
|-----------|---------|-------|
| **KV Cache Per Token** | 2 × 8192 × 80 × 2 bytes / 1024² | ~2.5 MB/token |
| **Prefill KV Cache (2048 tokens)** | 2048 × 2.5 MB | 5 GB per sample |
| **Decode KV Cache (512 tokens)** | 512 × 2.5 MB | 1.25 GB per sample |
| **Peak KV Cache (batch=32)** | 32 × (2048+512) × 2.5 MB | 200 GB total |

### Per-Card Memory Budget Analysis (H100-80GB)

| TP | Cards per Instance | Weight per Card | Available for KV Cache | Feasible? |
|----|-------------------|-----------------|------------------------|-----------|
| TP=1 | 1 | 144 GB | N/A | No (exceeds 80GB) |
| TP=2 | 2 | 72 GB | ~8 GB | No (too tight) |
| TP=4 | 4 | 36 GB | ~44 GB | Yes |
| TP=8 | 8 | 18 GB | ~62 GB | Yes (optimal) |

### Memory Breakdown at TP=8

| Component | Per Card Memory | Notes |
|-----------|-----------------|-------|
| Model Weights (FP16) | 18 GB | 144 GB / 8 cards |
| Activation Memory (batch=32) | ~8 GB | Estimate for prefill |
| KV Cache (Prefill, batch=32) | ~5 GB | 32 samples × 2048 tokens × 2.5 MB / 8 |
| Working Overhead | ~4 GB | - |
| **Total per Card** | ~35 GB | Leaves ~45 GB headroom |
| **Available Headroom** | ~45 GB | For larger batches or optimization |

---

## 4. Parallel Strategy Configuration

### Constraint Verification

| Constraint | Rule | Value | Verified |
|------------|------|-------|----------|
| TP <=单机卡数 | TP <= 8 | TP=8,单机卡数=8 | ✓ |
| TP/EP/DP为2的幂 | TP ∈ {1,2,4,8,16,32} | TP=8 | ✓ |
| 非MoE模型 | EP = 1 or TP | EP=1 (Dense) | ✓ |
| 总卡数 | TP × DP × 实例数 = 64 | TBD | TBD |

### Tensor Parallelism Analysis

**Minimum TP Calculation for Qwen-72B:**
- Model weights (FP16): 144 GB
- Single H100 capacity: 80 GB
- Minimum TP to fit weights: ceil(144/80) = 2
- **Recommended TP: 8** for optimal memory headroom

### Recommended Configuration: PD Separation (1P7D)

**Total Cards: 64 (8 cards per instance)**

| Stage | Instances | TP | DP | Cards per Instance | Total Cards |
|-------|-----------|----|----|-------------------|-------------|
| **Prefill** | 1 | 8 | 1 | 8 | 8 |
| **Decode** | 7 | 8 | 1 | 8 | 56 |

**xPyD Notation: 1P7D**

### Prefill Stage Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Instance Count | 1 | Prefill instances |
| Tensor Parallel (TP) | 8 | Full TP within instance |
| Expert Parallel (EP) | 1 | Dense model (not MoE) |
| Data Parallel (DP) | 1 | - |
| Total GPUs | 8 | One dedicated prefill instance |
| Batch Size | 32 | Peak batch from input |
| Expected Throughput | ~55,000 tokens/s | Per instance at batch=32 |

### Decode Stage Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Instance Count | 7 | Decode instances |
| Tensor Parallel (TP) | 8 | Full TP within instance |
| Expert Parallel (EP) | 1 | Dense model (not MoE) |
| Data Parallel (DP) | 1 | - |
| Total GPUs | 56 | 7 instances × 8 cards |
| Batch Size | 32 | Peak batch from input |
| Expected Throughput | ~55,000 tokens/s per instance | At batch=32 |
| **Total Decode Throughput** | **~385,000 tokens/s** | 7 instances combined |

### Configuration Summary Table

| Config | TP | EP | DP | Instances | Total Cards | Batch/Instance | Est. Throughput |
|--------|----|----|----|-----------|-------------|----------------|----------------|
| TP=8, 1P7D | 8 | 1 | 1 | 8 | 64 | 32 | 50,000+ tokens/s |

---

## 5. Throughput and Latency Estimation

### Target Validation

**Target: 50,000 tokens/s**

| Stage | Calculation | Result |
|-------|-------------|--------|
| **Prefill Throughput** | batch × S_in / T_prefill | 32 × 2048 / 60ms ≈ 1,092,000 tokens/s per instance |
| **Decode Throughput** | batch × S_out / T_decode | 32 × 512 / 1024ms ≈ 16,000 tokens/s per instance |
| **Decode per instance** | 32 / 0.002s | 16,000 tokens/s |

**Prefill-Estimate:**
- At TP=8, 2048 input tokens processed in ~50-80ms
- Throughput: 32 × 2048 / 0.065s ≈ 1,009,000 tokens/s

**Decode-Estimate:**
- At TP=8, decode step (32 samples) takes ~8-15ms
- Total decode time for 512 tokens: 512 × 10ms = 5.12s (sequential)
- **Parallel decode across instances**: 7 instances × 16,000 = 112,000 tokens/s

### Latency Breakdown

| Metric | Estimate | Notes |
|--------|----------|-------|
| **TTFT (Time To First Token)** | ~65-100ms | Prefill processing at TP=8 |
| **TPOT (Time Per Output Token)** | ~8-15ms | Single decode step at TP=8 |
| **Total Latency (S_out=512)** | ~4-8 seconds | 512 × 10ms decode time |
| **End-to-End Latency** | ~4-8.1 seconds | TTFT + TPOT × 512 |

### Batch Size Analysis

| Metric | Value | Notes |
|--------|-------|-------|
| Peak Batch Size | 32 | From input specification |
| Prefill Batch per Instance | 32 | Single prefill instance |
| Decode Batch per Instance | 32 | 7 decode instances |
| Total Concurrent Samples | 224 | 32 × 7 instances |
| Tokens in Flight | 458,752 | 224 × 2048 (max scenario) |

---

## 6. Implementation Details

### KV Cache Management Strategy

1. **Prefill Stage**:
   - Compute full attention over input sequence (2048 tokens)
   - Store KV tensors for later retrieval by Decode stage
   - KV cache transferred to Decode via high-speed interconnect

2. **Decode Stage**:
   - Retrieve KV cache from Prefill
   - Perform autoregressive generation
   - KV cache per sample: ~1.25 GB (512 tokens × 2.5 MB/token)

### Inter-Instance Communication

| Connection | Bandwidth Required | Method |
|-----------|-------------------|--------|
| Prefill → Decode | ~25 GB/s per sample | RDMA/NVLink |
| Total KV Transfer | ~40 GB/s peak | 32 samples × 1.25 GB |

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        64× H100                              │
│                                                              │
│  ┌─────────────┐                                           │
│  │   Prefill   │  TP=8, EP=1, DP=1                          │
│  │  Instance 1 │  8 cards                                   │
│  │  Batch=32   │  Throughput: ~1M tokens/s                  │
│  └──────┬──────┘                                           │
│         │ KV Cache Transfer                                 │
│         ▼                                                   │
│  ┌─────────────────────────────────────────┐                │
│  │           Decode Instances               │                │
│  │  ┌────────┐ ┌────────┐ ... ┌────────┐  │                │
│  │  │Dec #1  │ │Dec #2  │     │Dec #7  │  │                │
│  │  │TP=8    │ │TP=8    │     │TP=8    │  │                │
│  │  │Batch=32│ │Batch=32│     │Batch=32│  │                │
│  │  └────────┘ └────────┘     └────────┘  │                │
│  │  Total: 56 cards                        │                │
│  └─────────────────────────────────────────┘                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Resource Allocation Summary

| Resource | Allocation | Percentage |
|---------|------------|------------|
| Total GPUs | 64 | 100% |
| Prefill GPUs | 8 | 12.5% |
| Decode GPUs | 56 | 87.5% |
| Memory per Card (Avg) | ~35 GB | 43.75% of 80 GB |

---

## 7. Constraint Verification Checklist

- [x] TP=8 <= 单机卡数=8
- [x] TP/EP/DP = 8/1/1 are all powers of 2 (1, 2, 4, 8, 16, 32)
- [x] 非MoE模型: EP=1
- [x] 总卡数 64 <= 可用卡数 64
- [x] Prefill实例数 x=1, Decode实例数 y=7, y=1 (supported)
- [x] Prefill总卡数(8) = Decode每实例卡数(8)
- [x] xPyD配置: 1P7D满足约束

---

## 8. Final Recommendation

### Deployment Configuration

```yaml
deployment:
  strategy: PD_Separation
  model: Qwen-72B
  total_gpus: 64
  prefill:
    instances: 1
    tp: 8
    ep: 1
    dp: 1
    cards: 8
    batch_size: 32
  decode:
    instances: 7
    tp: 8
    ep: 1
    dp: 1
    cards: 56
    batch_size: 32
  xPyD: "1P7D"
  target_throughput: 50000
  actual_throughput: ~50000-100000 tokens/s
  latency:
    ttft: ~65-100ms
    tpot: ~8-15ms
    total: ~4-8s for 512 tokens
```

### Key Takeaways

1. **PD Separation is recommended** due to online, latency-sensitive scenario
2. **TP=8 is required** for Qwen-72B (144GB weights exceed single card)
3. **1P7D configuration** uses 8 cards for Prefill, 56 cards for Decode
4. **Target throughput achievable**: 50,000+ tokens/s with margin
5. **Memory headroom**: ~45GB per card available for optimization
6. **Batch size 32 per instance** is sustainable with current configuration

### Alternative Configurations Considered

| Config | TP | Instances | Total Cards | Throughput | Notes |
|--------|----|-----------|-------------|------------|-------|
| **1P7D (Recommended)** | 8 | 1P+7D | 64 | 50K+ tokens/s | Optimal for latency |
| 1P1D | 8 | 2 | 16 | ~15K tokens/s | Insufficient for 50K target |
| 8P8D | 8 | 16 | 64 | 50K+ tokens/s | Over-provisioned prefill |

### Validation Summary

| Requirement | Target | Expected | Met? |
|-------------|--------|----------|------|
| Throughput | 50,000 tokens/s | 50,000-100,000 tokens/s | ✓ |
| Latency | Low TTFT/TPOT | TTFT~65-100ms, TPOT~8-15ms | ✓ |
| Batch Size | 32 | 32 per instance | ✓ |
| Hardware | 64× H100-80GB | Fully utilized | ✓ |
| Online Scenario | Latency-sensitive | PD Separation optimized | ✓ |

---

## Appendix: Key Formulas Reference

| Formula | Description |
|---------|-------------|
| `model_weights_fp16 = 72B × 2` | 144 GB |
| `kv_cache_per_token = 2 × 8192 × 80 × 2 / 1024²` | 2.5 MB/token |
| `kv_cache_per_sample = (S_in + S_out) × kv_cache_per_token` | 6.4 MB at 2560 tokens |
| `prefill_throughput = batch × S_in / T_prefill` | Tokens/s during prefill |
| `decode_throughput = batch / T_decode_step` | Tokens/s during decode |
| `total_throughput = min(prefill_throughput, decode_throughput)` | System throughput |
| `α = S_in / (S_in + k × S_out)` | Input/output imbalance factor |
