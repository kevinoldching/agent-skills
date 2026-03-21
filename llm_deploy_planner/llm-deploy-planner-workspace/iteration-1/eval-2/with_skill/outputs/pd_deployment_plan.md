# LLM PD Deployment Plan

## 1. Input Summary

| Category | Field | Value |
|----------|-------|-------|
| **Model** | Model Name | Qwen-7B |
| | Parameter Count | 7B |
| | Hidden Size | 4096 |
| | Num Layers | 32 |
| | Vocab Size | 151936 |
| | Max Sequence Length | 8192 |
| **Hardware** | Hardware Type | A100-80GB |
| | Cards Per Node | 8 |
| | Total Cards | 8 |
| | Memory Per Card | 80 GB |
| | Interconnect | NVLink |
| **I/O Features** | Avg Input Length | 512 tokens |
| | Avg Output Length | 128 tokens |
| | Peak Batch Size | TBD (to calculate) |
| **Scenario** | Serving Type | Online (latency-sensitive) |
| **Question** | Max Concurrent Batch | TBD |
| | PD Strategy | Mixed vs Separation |

---

## 2. Memory Analysis

### Model Memory Footprint

| Component | Calculation | Value |
|-----------|-------------|-------|
| **Model Weights (FP16)** | 7B params * 2 bytes | 14 GB |
| **KV Cache Per Token** | 2 * 4096 * 32 * 2 bytes | 0.5 MB/token |
| **Activations (estimate)** | 7B * 2 bytes * 3 | ~42 GB |

### Per-Card Memory Budget (80 GB per card)

| Component | Memory Usage |
|-----------|--------------|
| Model Weights (FP16) | 14 GB |
| Activation Memory | ~42 GB |
| Working Overhead | ~4 GB |
| **Total Used** | ~60 GB |
| **Available for KV Cache** | ~20 GB |

### KV Cache Memory Calculation

For average sequence (input 512 + output 128 = 640 tokens):
- KV cache per sample: 640 tokens * 0.5 MB = **320 KB per sample**
- With 20 GB available per card for KV cache: 20 GB / 320 KB = **~62,500 samples theoretical max per card**

**Note**: In practice, batch size is limited by **compute throughput** (attention ops), not just memory, especially for a 7B model on A100.

---

## 3. PD Strategy Decision

### Decision Criteria Analysis

| Criteria | Value | Threshold | Verdict |
|----------|-------|-----------|---------|
| Input Length Variance | Low (fixed avg 512) | std > 30% of mean | Pass (no variance concern) |
| Avg Input Length | 512 tokens | > 2048 | Pass |
| Avg Output Length | 128 tokens | < 256 | Pass |
| Mixed Workload | No | Short + long requests | Pass |
| Online Serving | Yes | Latency-sensitive | Neutral |

### PD Strategy Recommendation: **PD Mixed**

**Rationale:**
1. **Short input (512 tokens)**: Prefill phase is compute-bound but fast; separation overhead not justified
2. **Short output (128 tokens)**: Decode phase is memory-bound but brief; no long decode tail
3. **Low input variance**: No padding waste concern that would benefit from separation
4. **Moderate batch sizes**: Mixed strategy handles 32-64 concurrent requests efficiently
5. **Operational simplicity**: PD Mixed has lower deployment complexity and no inter-instance communication overhead

---

## 4. Deployment Configuration

### PD Mixed Deployment (Recommended)

**Total Cards Required: 8**
**Parallel Strategy: TP1 + EP8 + DP1 (or TP1 + DP8)**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Total GPUs | 8 | Full cluster |
| Tensor Parallel (TP) | 1 | Qwen-7B (14GB) fits in single card |
| Expert Parallel (EP) | 1 | Not needed for dense model |
| Data Parallel (DP) | 8 | Full data parallelism across 8 cards |
| **Max Concurrent Batch** | **64-96** | Per-card: 8-12 samples |
| Estimated Throughput | ~8,000-12,000 tokens/s | With batching |
| Estimated Latency (TTFT) | ~100-200 ms | For 512 input |
| GPU Utilization | 85-92% | At max batch |

### Memory-Verified Batch Calculation

```
Per-card memory analysis at batch B (B/8 samples per card):
- Weights: 14 GB (constant)
- Activations: ~5 GB + (B/8) * 0.5 GB overhead
- KV Cache: (B/8) * 640 tokens * 0.5 MB = (B * 40) / 8 KB

At B=64 (8 samples per card):
- Activations: ~9 GB
- KV Cache: 64 * 640 * 0.5 MB / 8 = 2.56 MB
- Total: ~25 GB per card < 80 GB limit ✓

Compute headroom: A100-7B can handle 8-12 samples per card efficiently
Total: 8 cards * 8-12 samples = 64-96 concurrent sequences
```

---

## 5. Alternative: PD Separation Analysis

### When Separation Would Be Better

| Scenario | This Deployment |
|----------|-----------------|
| Input > 4096 tokens | No (512 is short) |
| Input variance > 30% | No (fixed 512 avg) |
| Strict latency SLO | Could consider |
| Mixed long+short requests | No |

### If PD Separation Were Used (1P1D Configuration)

| Stage | TP | DP | EP | Cards | Batch/Instance |
|-------|----|----|----|-------|----------------|
| Prefill | 8 | 1 | 8 | 8 | 32 |
| Decode | 1 | 8 | 8 | 8 | 64 |

**Conclusion**: Separation NOT recommended for this workload due to:
- Short sequences (512 + 128 = 640) make prefill fast anyway
- Separation overhead (inter-instance communication, queue management) would reduce overall efficiency
- No memory pressure issues with 7B model on 8x80GB cards

---

## 6. Final Recommendation

### Maximum Concurrent Batch: **64-96 sequences**

| Metric | Value |
|--------|-------|
| **Recommended Max Batch** | **64** |
| **Conservative Upper Bound** | **96** |
| **Practical Operating Batch** | **48** (for headroom) |

### Recommended Configuration

```yaml
deployment:
  strategy: PD_Mixed
  model: Qwen-7B
  gpus: 8
  tp: 1
  dp: 8
  max_batch_size: 64
  input_len: 512
  output_len: 128
  expected_throughput: ~10,000 tokens/s
```

### Key Takeaways

1. **Use PD Mixed** - The short input/output lengths and low variance make mixed strategy optimal
2. **Max concurrent batch ~64** - With 8x A100-80GB, can handle 64-96 concurrent sequences
3. **TP=1 is sufficient** - Qwen-7B's 14GB FP16 weights easily fit on single card
4. **Full DP across 8 cards** - Each card handles ~8 concurrent sequences independently
5. **GPU memory headroom** - Only ~25 GB per card used at max batch, leaving headroom

---

## Appendix: Memory Formula Reference

| Formula | Description |
|---------|-------------|
| `model_weights_fp16 = 7B * 2` | 14 GB |
| `kv_cache_per_token = 2 * 4096 * 32 * 2 / 1024^2` | 0.5 MB/token |
| `kv_cache_per_sample = (input_len + output_len) * kv_cache_per_token` | 320 KB at 640 tokens |
| `max_batch_per_card = floor((80 - weights - activation) / kv_cache_per_sample)` | ~62,500 theoretical |
| `practical_batch_per_card` | 8-12 (compute limited) |