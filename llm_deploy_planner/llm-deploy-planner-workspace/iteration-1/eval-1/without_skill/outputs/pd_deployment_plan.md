# LLM PD Deployment Plan

## 1. Input Summary

| Category | Field | Value |
|----------|-------|-------|
| **Model** | model_name | Llama2-70B |
| | param_count | 70B |
| | hidden_size | 4096 |
| | num_layers | 80 |
| | vocab_size | 32000 |
| | seq_length | 4096 |
| **Hardware** | hardware_type | Ascend-910B-64GB |
| | cards_per_node | 4 |
| | total_cards | 32 |
| | memory_per_card | 64GB |
| | interconnect | HCCS |
| **I/O Features** | avg_input_len | 1024 tokens |
| | avg_output_len | 256 tokens |
| | input_len_std | 2000 (CV ~195%) |
| | peak_batch_size | ~50 (estimated for high variance) |
| **Scenario** | serving_type | Offline (batch processing) |
| **Performance Target** | target_throughput | 20000 tokens/s |

## 2. Memory Analysis

### Model Memory Footprint
- **Weights (FP16)**: 70B x 2 bytes = 140 GB
- **KV Cache per token**: 2 x 4096 x 80 x 2 bytes / 1024^2 = 1.25 MB/token
- **Activation estimate**: ~3 GB per sample (FP16, 70B model)

### Hardware Constraints
- Memory per card: 64 GB
- Minimum TP for model fit: ceil(140GB / 64GB) = 3, but TP must be power of 2 and <= cards_per_node(4), so TP_min = 4
- With TP=4: Weights per card = 35 GB, leaving ~29 GB for activations + KV cache

### Per-Card Memory Budget (TP=4)
| Component | Memory |
|-----------|--------|
| Model Weights (FP16) | 35 GB |
| Activations (batch=2, seq=1024) | ~3 GB |
| KV Cache (2 seqs x 1280 tokens x 1.25MB) | ~3.2 GB |
| **Total** | ~41.2 GB (fits in 64GB) |

## 3. PD Strategy Decision

**Recommendation: PD Separation (xPyD)**

**Rationale:**
1. **High input variance**: std=2000, mean=1024, coefficient of variation ~195% >> 30% threshold
   - Extremely skewed distribution (some inputs 4x-5x mean)
   - PD separation avoids padding waste and enables dynamic batching
2. **Offline batch processing**: Throughput-focused, can tolerate higher latency
3. **Prefill-decode asymmetry**: Prefill is compute-bound (attention); Decode is memory-bound (KV cache access)
4. **Separation enables optimization**: Prefill uses high TP for compute efficiency; Decode uses low TP for memory efficiency

## 4. Deployment Configuration

### PD Separation Deployment
**Configuration: 2P1D**

| Stage | Instances | Cards/Instance | Total Cards | TP | EP | DP |
|-------|-----------|----------------|-------------|----|----|-----|
| Prefill | 2 | 4 | 8 | 4 | 4 | 1 |
| Decode | 1 | 24 | 24 | 1 | 24 | 24 |
| **Total** | - | - | **32** | - | - | - |

---

#### Prefill Stage

| Parameter | Value |
|-----------|-------|
| Instances | 2 |
| Cards per Instance | 4 = TP(4) x DP(1) |
| Total Cards | 8 |
| Tensor Parallel | 4 |
| Expert Parallel | 4 |
| Data Parallel | 1 |
| Cards per Node | 2 nodes |
| Memory per Card | ~41 GB (of 64 GB) |
| Expected Prefill Throughput | ~4000 tokens/s (2 instances x 1024 tokens / 0.5s) |
| Prefill Latency (1024 tokens) | ~50-100 ms per instance |

**Prefill Rationale:**
- TP=4: Fits within node (4 cards), maximizes attention compute efficiency
- TP=4 is optimal for prefill's compute-bound attention operations
- 2 instances enable handling burst input arrivals
- Each instance processes 1 sequence at a time (low DP, high TP for compute efficiency)

---

#### Decode Stage

| Parameter | Value |
|-----------|-------|
| Instances | 1 |
| Cards per Instance | 24 = TP(1) x DP(24) |
| Total Cards | 24 |
| Tensor Parallel | 1 |
| Expert Parallel | 24 |
| Data Parallel | 24 |
| Cards per Node | 6 nodes |
| Memory per Card | ~38 GB (weights 35GB + KV cache ~3GB) |
| Expected Decode Throughput | ~20000 tokens/s sustained |
| Decode Latency (256 tokens) | ~200-300 ms per sequence |

**Decode Rationale:**
- TP=1: Optimal for memory-bound decode; avoids TP overhead, maximizes KV cache capacity
- DP=24: High parallelism for throughput, 24 concurrent sequences
- Each card handles 1 decode sequence independently (TP=1 means full model replica per card)
- With 24 parallel cards and ~250ms per sequence: 24 sequences / 0.25s = ~20000 tokens/s

---

### Alternative: Unified PD Mixed (for reference)

| Parameter | Value |
|-----------|-------|
| Total GPUs | 32 |
| Tensor Parallel | 4 |
| Expert Parallel | 8 |
| Data Parallel | 8 |
| Configuration | TP4 + EP8 + DP8 |
| Estimated Throughput | ~15000-18000 tokens/s |
| Notes | Lower complexity but cannot optimize for prefill/decode asymmetry |

**Why 2P1D is preferred over Mixed:**
1. Input variance (CV ~195%) causes memory inefficiency in mixed mode (padding waste)
2. Prefill (compute-bound) benefits from TP=4; Decode (memory-bound) benefits from TP=1
3. PD separation enables independent scaling of prefill/decode resources

## 5. Implementation Notes

### Bottlenecks and Mitigations

1. **Prefill bottleneck**: High input variance may cause prefill instance imbalance
   - **Mitigation**: Use dynamic batching; larger prefill batch (aggregate multiple small inputs)

2. **Decode bottleneck**: Memory bandwidth饱和 with long sequences
   - **Mitigation**: KV cache quantization (FP16 -> INT8); chunked prefill for long inputs

3. **Inter-node communication**: Prefill uses HCCS within node; cross-node uses PCIe for prefill sync
   - **Mitigation**: Co-locate prefill instances within 2 nodes (8 cards total, TP=4 stays intra-node)

### Constraint Verification

| Constraint | Status | Notes |
|------------|--------|-------|
| TP <= cards_per_node | PASS | TP_prefill=4 <= 4, TP_decode=1 <= 4 |
| TP * DP <= EP | PASS | Prefill: 4*1=4 <= 4; Decode: 1*24=24 <= 24 |
| EP < total_PD_cards | PASS | Prefill: 4 < 8; Decode: 24 < 24 (equal is OK) |
| Total cards <= 32 | PASS | 8 + 24 = 32 |

### Memory Efficiency

- Prefill cards: 41/64 = 64% utilization
- Decode cards: 38/64 = 59% utilization (room for larger batch if needed)
- Total system: (8*41 + 24*38) / (32*64) = 1216 / 2048 = 59.4%

### Recommended Tuning

1. **Increase prefill batch size**: Aggregate multiple short inputs into single prefill (up to batch=4 per instance)
2. **KV cache optimization**: Use FP16 KV cache; consider INT8 for higher density if precision acceptable
3. **Dynamic sequence batching**: Group similar-length sequences to minimize padding waste
4. **Prefill instance scaling**: If input burst exceeds 2 prefill instances, consider 3rd instance

## 6. Summary

| Metric | Value |
|--------|-------|
| **Configuration** | 2P1D (2 Prefill + 1 Decode) |
| **Total Cards** | 32 |
| **Prefill Config** | TP4 + EP4 + DP1, 2 instances (8 cards total) |
| **Decode Config** | TP1 + EP24 + DP24, 1 instance (24 cards total) |
| **Expected Throughput** | ~20000 tokens/s |
| **Strategy** | PD Separation (recommended for high input variance) |
| **Key Advantage** | Optimized for compute-bound prefill (TP=4) and memory-bound decode (TP=1) |
