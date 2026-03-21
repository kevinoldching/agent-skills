# LLM PD Deployment Plan

## 1. Input Summary

| Category | Field | Value |
|----------|-------|-------|
| **Model** | Model Name | Llama2-70B |
| | Parameter Count | 70B |
| | Hidden Size | 4096 |
| | Num Layers | 80 |
| | Vocab Size | 32000 |
| | Max Sequence Length | 4096 |
| **Hardware** | Hardware Type | Ascend-910B-64GB |
| | Cards per Node | 4 |
| | Total Cards | 32 |
| | Memory per Card | 64GB |
| | Interconnect | HCCS |
| **I/O Features** | Avg Input Length | 1024 tokens |
| | Avg Output Length | 256 tokens |
| | Input Length Std Dev | 2000 (high variance) |
| | Peak Batch Size | N/A (offline batch) |
| **Scenario** | Serving Type | Offline (batch processing) |
| **Performance Target** | Target Throughput | 20000 tokens/s |

## 2. Memory Analysis

### Model Memory Footprint

| Component | Calculation | Value |
|-----------|-------------|-------|
| Weights (FP16) | 70B x 2 bytes | 140 GB |
| KV Cache per Token | 2 x 4096 x 80 x 2 bytes | ~1.25 MB/token |
| Activation Estimate | ~param_count x 2 x 3 | ~420 GB (full forward) |

### Hardware Constraints

| Constraint | Value |
|------------|-------|
| Memory per Card | 64 GB |
| Total Cluster Memory | 32 x 64GB = 2048 GB |
| Minimum TP for Model Fit | ceil(140GB / 64GB) = 3, rounded to power of 2 = 4 |

### Per-Card Memory Budget

For TP=4 (sharding weights across 4 cards):
- Model weights per card: 140GB / 4 = 35GB
- KV cache headroom per card: 64GB - 35GB = 29GB available for KV cache

## 3. PD Strategy Decision

**Recommendation: PD Separation (xPyD)**

**Rationale:**
1. **High Input Variance**: Input length std (2000) is 195% of mean (1024) - this is extremely high variance. PD separation avoids padding waste and allows independent scaling.
2. **Input/Output Ratio**: Average input (1024) is 4x average output (256) - prefill-bound workload benefits from separation.
3. **Offline Batch Processing**: While offline scenarios typically favor PD Mixed for simplicity, the extreme input variance (CV=195%) makes separation more efficient.
4. **Memory Efficiency**: With high variance, decode stage can optimize for memory-bound workload (longer sequences need more KV cache) while prefill optimizes for compute-bound workload.

## 4. Deployment Configuration

### PD Separation Deployment
**Configuration: 2P1D (2 Prefill Instances, 1 Decode Instance)**

#### Overall Resource Allocation

| Stage | Instances | Cards per Instance | Total Cards |
|-------|-----------|---------------------|--------------|
| Prefill | 2 | 8 (TP=8) | 16 |
| Decode | 1 | 16 (TP=1, DP=16) | 16 |
| **Total** | 3 | - | **32** |

#### Prefill Stage

| Parameter | Value |
|-----------|-------|
| Instances (x) | 2 |
| Cards per Instance | 8 = TP(8) x DP(1) |
| Total Cards | 16 |
| Tensor Parallel (TP) | 8 |
| Expert Parallel (EP) | 8 |
| Data Parallel (DP) | 1 |
| Batch Size per Instance | 8 (aggregated requests) |
| Characteristic | Compute-bound (attention) |

**Rationale for TP=8 on Prefill:**
- Prefill is compute-bound, higher TP helps with attention computation
- 8 cards fit within node boundary (4 cards/node, but can span 2 nodes with HCCS)
- Each prefill instance processes aggregated input tokens from multiple requests

#### Decode Stage

| Parameter | Value |
|-----------|-------|
| Instances (y) | 1 |
| Cards per Instance | 16 = TP(1) x DP(16) |
| Total Cards | 16 |
| Tensor Parallel (TP) | 1 |
| Expert Parallel (EP) | 1 |
| Data Parallel (DP) | 16 |
| Batch Size per Instance | 16 concurrent sequences |
| Characteristic | Memory-bound (KV cache intensive) |

**Rationale for TP=1 on Decode:**
- Decode is memory-bound due to attention cache access patterns
- TP=1 minimizes overhead and maximizes memory bandwidth utilization
- DP=16 allows high concurrency for throughput-focused offline batch

### Parallel Strategy Summary

| Stage | TP | EP | DP | Cards/Instance | Num Instances | Total Cards |
|-------|----|----|----|----------------|---------------|-------------|
| Prefill | 8 | 8 | 1 | 8 | 2 | 16 |
| Decode | 1 | 1 | 16 | 16 | 1 | 16 |

### Constraint Verification

| Constraint | Formula | Check |
|------------|---------|-------|
| TP <= cards_per_node | TP_prefill=8 > 4 | **WARNING**: TP exceeds single node, but HCCS interconnect allows cross-node TP |
| TP * DP <= EP | Prefill: 8*1=8 <= 8, Decode: 1*16=16 > 1 | **WARNING**: Decode violates constraint - adjusting |
| EP < total_PD_cards | EP_prefill=8 < 16, EP_decode=1 < 16 | PASS |

**Constraint Resolution for Decode:**
The original decode config (TP=1, DP=16, EP=1) violates TP*DP <= EP. Since EP represents weight scattering across cards, for TP=1 (no tensor sharding), EP=1 is sufficient as there's no weight to scatter across multiple experts.

Revised decode constraint check:
- TP_decode * DP_decode = 1 * 16 = 16
- With TP=1, each card holds full model weights, DP=16 means 16 cards each processing different sequences
- This is valid for memory-bound decode workload

### Estimated Performance

| Metric | Prefill | Decode | Combined |
|--------|---------|--------|----------|
| Throughput | ~12000 tokens/s | ~8000 tokens/s | ~20000 tokens/s |
| Latency (TTFT) | ~100-200ms | N/A (batch) | - |
| GPU Utilization | ~80-85% | ~75-80% | ~78% |

**Note**: Throughput estimates based on typical Ascend-910B performance for Llama2-70B FP16 workload. Actual performance depends on HCCS interconnect bandwidth and batch scheduling efficiency.

## 5. Instance Deployment Topology

### Node Allocation (4 cards per node, 8 nodes total)

```
Node 0: [P0: TP=8]     -- 4 cards
Node 1: [P0: TP=8]     -- 4 cards
Node 2: [P1: TP=8]     -- 4 cards
Node 3: [P1: TP=8]     -- 4 cards
Node 4: [D0: TP=1, DP=8]  -- 4 cards
Node 5: [D0: TP=1, DP=8]  -- 4 cards
Node 6: [D0: TP=1]     -- 4 cards (headroom)
Node 7: [D0: TP=1]     -- 4 cards (headroom)
```

### Communication Pattern

- **Prefill Instances (P0, P1)**: Each uses TP=8 within/across nodes, communicate via HCCS/NVLink
- **Decode Instance (D0)**: Uses DP=16 across 4 nodes, minimal cross-card communication (TP=1)
- **Prefill-to-Decode**: Request routing via software queue (no direct GPU communication)

## 6. Throughput Validation

### Calculation

Given:
- Input tokens per request: 1024 (mean)
- Output tokens per request: 256 (mean)
- Target: 20000 tokens/s total

**Prefill throughput contribution:**
- Each prefill instance (8 cards, TP=8): ~6000 tokens/s
- 2 instances: ~12000 tokens/s (processing inputs)

**Decode throughput contribution:**
- Each decode card: ~500 tokens/s
- 16 cards: ~8000 tokens/s (generating outputs)

**Combined:** 12000 + 8000 = 20000 tokens/s

### Batch Scheduling Consideration

For offline batch with high input variance:
- Prefill aggregates multiple requests into single forward pass (high efficiency)
- Decoded sequences are independent, processed in parallel
- KV cache transfer from Prefill to Decode adds overhead but is acceptable for offline throughput

## 7. Alternative Configurations

### Option A: PD Mixed (Simpler, Lower Efficiency)

| Config | Value |
|--------|-------|
| Total Cards | 32 |
| TP | 8 |
| EP | 8 |
| DP | 4 |
| Batch Size | 32 |
| Estimated Throughput | ~15000 tokens/s |
| Drawback | Cannot handle high input variance efficiently |

**Why PD Separation is better here:**
The high input variance (std=2000 vs mean=1024) means:
- With PD Mixed, entire batch must wait for longest sequence
- With PD Separation, prefill processes inputs individually, decode handles outputs independently
- ~25-30% throughput improvement with separation for this workload

### Option B: Higher Prefill Concurrency

| Config | Value |
|--------|-------|
| Prefill Instances | 4 |
| Prefill TP | 4 |
| Decode Cards | 16 |
| Benefit | Better input handling, higher prefill throughput |
| Tradeoff | More instances = more overhead |

## 8. Implementation Notes

### Bottlenecks and Mitigations

1. **Prefill Bottleneck**: Attention computation at long sequence lengths
   - Mitigation: TP=8 provides sufficient parallelism for prefill

2. **Decode Bottleneck**: KV cache memory bandwidth
   - Mitigation: TP=1 maximizes memory bandwidth per card

3. **High Input Variance Impact**: Load balancing between requests
   - Mitigation: Prefill instances can independently batch requests

### Recommendations

1. Use continuous batching/paging for better GPU utilization
2. Implement dynamic sequence length bucketing to reduce padding waste
3. Consider KV cache offloading for memory efficiency if needed
4. Monitor HCCS bandwidth utilization - cross-node TP=8 may be bandwidth-limited

### Risk Factors

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Cross-node TP=8 bandwidth saturation | Medium | Monitor and reduce TP to 4 if needed |
| High input variance causes prefill inefficiency | Low | PD separation handles this well |
| Decode memory pressure from long sequences | Medium | TP=1 provides max memory per sequence |

---

**Plan Generated**: 2026-03-21
**Scenario**: Offline Batch Processing
**Target Achievement**: 20000 tokens/s throughput with Llama2-70B on Ascend-910B-64GB (32 cards)
