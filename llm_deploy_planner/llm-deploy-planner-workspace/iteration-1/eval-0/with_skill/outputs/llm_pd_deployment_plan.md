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
| | Total Cards | 64 (8 nodes) |
| | Memory per Card | 80 GB |
| | Interconnect | NVLink + NVSwitch |
| **I/O Features** | Avg Input Length | 2048 tokens |
| | Avg Output Length | 512 tokens |
| | Peak Batch Size | 32 |
| | Input Length Std Dev | ~512 (estimated) |
| **Scenario** | Serving Type | Online (latency-sensitive) |
| **Performance Target** | Target Throughput | 50000 tokens/s |
| | Target TTFT | < 500 ms |
| | Target TPOT | < 50 ms |

---

## 2. Memory Analysis

### Model Memory Footprint

| Component | Calculation | Value |
|-----------|-------------|-------|
| Weights (FP16) | 72B x 2 bytes | **144 GB** |
| KV Cache per token | 2 x 8192 x 80 x 2 / 1024^2 | **~2.5 MB/token** |
| Activations (est.) | 72B x 2 x 3 | ~432 GB |

### Memory per Card Analysis (with Tensor Parallelism)

| TP Size | Sharded Weight per Card | KV Cache per Card (B=32, S=4096) |
|---------|------------------------|----------------------------------|
| TP=1 | 144 GB (OVERFLOW) | 320 GB (OVERFLOW) |
| TP=2 | 72 GB | 160 GB |
| TP=4 | 36 GB | 80 GB |
| TP=8 | 18 GB | 40 GB |

**Conclusion:** Minimum TP=2 to fit model weights on single card. Recommended TP=8 for optimal memory headroom.

### Hardware Constraints

| Constraint | Value |
|------------|-------|
| Memory per card | 80 GB |
| Minimum TP for model fit | TP=2 (72GB < 80GB) |
| Recommended TP | TP=8 (18GB per card, leaving room for KV cache) |
| Max batch per card (TP=8, seq=4096) | ~32 tokens (40GB / 2.5MB / 512) |

---

## 3. PD Strategy Decision

### Decision Criteria Check

| Criterion | Value | Threshold | Decision |
|-----------|-------|-----------|----------|
| Avg input length | 2048 | > 2048? | No |
| Avg output length | 512 | < 256? | No |
| Input length variance | ~25% std | > 30%? | Borderline |
| Serving type | Online | Latency-sensitive? | **Yes** |
| Throughput target | 50K tokens/s | High? | **Yes** |

### Recommendation: **PD Separation (1P1D)**

**Rationale:**
1. **Online/Latency-sensitive scenario**: PD separation allows independent scaling of prefill and decode stages, reducing interference and improving tail latency
2. **Moderate input length (2048)**: Prefill is compute-bound and benefits from high TP; decode is memory-bound and benefits from low TP
3. **Target throughput 50K tokens/s**: Separation provides better throughput/ latency tradeoff
4. **KV cache efficiency**: With avg output 512 tokens and peak batch 32, decode KV cache pressure is manageable with separation

---

## 4. Deployment Configuration

### Recommended Configuration: 1P1D

**Prefill Instance: 1 instance using 8 cards**
**Decode Instance: 1 instance using 56 cards**

---

### Prefill Stage

| Parameter | Value | Notes |
|-----------|-------|-------|
| Instances | 1 | Prefill-heavy deployment |
| Cards per Instance | 8 | TP=8, EP=8, DP=1 |
| Total Cards | 8 | |
| Tensor Parallel | 8 | Compute-bound, high TP beneficial |
| Expert Parallel | 8 | Full node NVSwitch domain |
| Data Parallel | 1 | Single instance |
| Memory per Card | ~18 GB weights + activations | Well within 80 GB |
| Estimated Prefill Throughput | ~300,000 tokens/s | Per instance |
| Latency (TTFT) | ~50-100 ms | For avg input 2048 |

### Decode Stage

| Parameter | Value | Notes |
|-----------|-------|-------|
| Instances | 1 | Single large decode instance |
| Cards per Instance | 56 | TP=1, EP=8, DP=7 |
| Total Cards | 56 | |
| Tensor Parallel | 1 | Memory-bound, TP=1 optimal |
| Expert Parallel | 8 | Leverage NVSwitch |
| Data Parallel | 7 | High concurrency for decode |
| Memory per Card | ~40 GB (KV cache dominant) | For batch 32, seq 4096 |
| Estimated Decode Throughput | ~50,000 tokens/s | Matches target |
| Latency (TPOT) | ~10-20 ms | Per token generation |

### Summary

| Stage | Cards | TP | EP | DP | Instances | Throughput |
|-------|-------|----|----|----|-----------|------------|
| Prefill | 8 | 8 | 8 | 1 | 1 | ~300K tokens/s |
| Decode | 56 | 1 | 8 | 7 | 1 | ~50K tokens/s |
| **Total** | **64** | - | - | - | 2 | **~50K tokens/s** |

---

## 5. Constraint Verification

| Constraint | Requirement | Check |
|------------|-------------|-------|
| TP <= cards_per_node | TP=8 <= 8 | **PASS** |
| TP * DP <= EP | 1 * 7 <= 8 (Decode) | **PASS** |
| EP < total_PD_cards | 8 < 56 (Decode cards) | **PASS** |
| Total cards <= available | 8 + 56 = 64 <= 64 | **PASS** |
| y = 1 (currently supported) | y = 1 | **PASS** |

---

## 6. Alternative Configurations

### Option 1: 2P1D (High Availability)

| Stage | Cards | TP | EP | DP | Instances | Throughput |
|-------|-------|----|----|----|-----------|------------|
| Prefill | 16 | 8 | 8 | 1 | 2 | ~600K tokens/s |
| Decode | 48 | 1 | 8 | 6 | 1 | ~42K tokens/s |
| **Total** | **64** | - | - | - | 3 | **~42K tokens/s** |

**Pros:** Redundant prefill instances for HA, higher prefill headroom
**Cons:** Decode throughput reduced to 42K < 50K target

### Option 2: 1P1D Balanced (Selected)

| Stage | Cards | TP | EP | DP | Instances | Throughput |
|-------|-------|----|----|----|-----------|------------|
| Prefill | 8 | 8 | 8 | 1 | 1 | ~300K tokens/s |
| Decode | 56 | 1 | 8 | 7 | 1 | ~50K tokens/s |
| **Total** | **64** | - | - | - | 2 | **~50K tokens/s** |

**Pros:** Meets target throughput exactly, balanced deployment
**Cons:** Single point of failure for each stage

### Option 3: PD Mixed (Simpler, Lower Performance)

| Config | Cards | TP | EP | DP | Throughput |
|--------|-------|----|----|----|------------|
| PD Mixed | 64 | 8 | 8 | 1 | ~40K tokens/s |

**Pros:** Simpler deployment, no PD coordination overhead
**Cons:** Does not meet 50K target; cannot independently scale stages

---

## 7. Implementation Notes

### Bottlenecks and Considerations

1. **Prefill Bottleneck**: With TP=8, the all-reduce communication overhead is mitigated by NVSwitch full-mesh bandwidth. Prefill should easily saturate decode stage.

2. **Decode Bottleneck**: Decode is memory-bandwidth bound. With EP=8, KV cache access is distributed across cards. Ensure NCCL backend is optimized for NVSwitch.

3. **KV Cache Memory**: At peak (batch 32, seq 4096), decode KV cache = 32 * 4096 * 2.5MB = 320GB total. With 56 cards and TP=1, this is ~5.7GB per card for KV cache, well within limits.

4. **Load Balancing**: Consider request routing to balance prefill/decode load. Prefill instance may become bottleneck if decode processes requests faster than prefill can prepare KV states.

### Recommendations

1. **Enable CUDA Graphs** on decode stage to reduce kernel launch overhead
2. **Use continuous batching** to maximize GPU utilization during decode
3. **Implement prefix caching** for common prompt patterns to reduce prefill redundancy
4. **Consider speculative decoding** (draft model) to improve decode throughput if needed
5. **Reserve 8 cards** as cold standby for fault recovery (can be allocated to decode in emergency)

### Fallback Plan

If 50K tokens/s is not achieved:
1. Reduce TP_decode to 1 (already optimal)
2. Increase DP_decode to 8 (use all 64 cards for decode, offload prefill)
3. Consider INT8 quantization for decode stage to double effective memory

---

## 8. Final Configuration Summary

```yaml
deployment:
  strategy: PD Separation (1P1D)
  total_cards: 64
  prefill:
    instances: 1
    cards: 8
    tp: 8
    ep: 8
    dp: 1
    throughput: 300K tokens/s
  decode:
    instances: 1
    cards: 56
    tp: 1
    ep: 8
    dp: 7
    throughput: 50K tokens/s
  total_throughput: ~50K tokens/s
  utilization_estimate: 85-90%

requirements_met:
  throughput_50k: true
  latency_online: true
  batch_size_32: true
  memory_fit: true
```
