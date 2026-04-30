# LLM PD Deployment Plan: Qwen-7B on 8x A100-80GB

## 1. Input Summary

| Category | Field | Value |
|----------|-------|-------|
| **Model** | model_name | Qwen-7B |
| | param_count | 7B |
| | hidden_size | 4096 |
| | num_layers | 32 |
| | vocab_size | 151936 |
| | seq_length | 8192 |
| | num_attention_heads | 32 |
| | num_kv_heads | 8 (GQA) |
| | head_dim | 128 |
| **Hardware** | hardware_type | NVIDIA A100-80GB |
| | cards_per_node | 8 |
| | total_cards | 8 |
| | memory_per_card | 80 GB |
| | interconnect | NVLink (900 GB/s) |
| **I/O Features** | avg_input_len | 512 tokens |
| | avg_output_len | 128 tokens |
| | peak_batch_size | TBD |
| | input_len_std | Not specified |
| **Scenario** | serving_type | Online serving |
| **Performance Target** | target_throughput | Not specified |
| | target_latency | Not specified |
| | target_utilization | Not specified |

---

## 2. Memory Analysis

### 2.1 Model Memory Footprint

| Component | Calculation | Size |
|-----------|-------------|------|
| **Weights (FP16)** | 7B x 2 bytes | **14 GB** |
| **Weights (INT8)** | 7B x 1 byte | 7 GB (if quantized) |
| **KV Cache per token** | 2 (K+V) x num_kv_heads(8) x head_dim(128) x layers(32) x 2 bytes | **128 KB per token** |
| **Activation memory (full forward)** | param_count x 2 bytes x activation_factor | ~42 GB (theoretical max) |

### 2.2 KV Cache Capacity Per Card

```
KV Cache per token = 2 x 8 x 128 x 32 x 2 = 128 KB/token

With 80 GB per card:
- After model weights (14 GB): 66 GB available
- For KV cache only: 66 GB / 128 KB = ~515,000 tokens capacity
- For decode phase (128 tokens/output): ~4,000 concurrent decode tokens
```

### 2.3 Activation Memory During Prefill (Critical Constraint)

Prefill phase is compute-bound with O(n^2) attention cost. Activation memory scales with:
- Batch size
- Sequence length (512 input tokens)

**Activation estimate for prefill (batch=B, seq=512):**
```
Activation_memory ≈ batch_size x seq_len x hidden_size x layers x constant_factor
                  ≈ B x 512 x 4096 x 32 x 4 bytes (attention + mlp activations)
                  ≈ B x 256 MB
```

| Batch Size | Activation Memory (Prefill) | KV Cache (640 tokens) | Model Weights | Total | Status |
|-----------|----------------------------|-----------------------|---------------|-------|--------|
| 32 | ~8 GB | 4 MB x 32 = 128 KB | 14 GB | ~22 GB | OK |
| 64 | ~16 GB | 8 MB | 14 GB | ~30 GB | OK |
| 128 | ~32 GB | 16 MB | 14 GB | ~46 GB | OK |
| 256 | ~64 GB | 32 MB | 14 GB | ~78 GB | TIGHT |
| 320 | ~80 GB | 40 MB | 14 GB | ~94 GB | EXCEEDS |

### 2.4 Decode Phase Memory (Less Constrained)

Decode phase processes one token at a time, no O(n^2) attention:
- Activation memory is much smaller
- KV cache accumulates as sequence grows

```
At batch=128, output_len=128, total_seq=640:
KV cache per sample: 640 x 128 KB = 80 MB
KV cache total: 128 x 80 MB = 10 GB
Activation: ~2 GB (no attention over full sequence)
Model weights: 14 GB
Total: ~26 GB per card
```

---

## 3. Parallel Strategy Analysis

### 3.1 Tensor Parallel (TP) Options

**TP=1 (Single Card, No Sharding):**
- Memory per card: 14 GB (weights) + activations + KV cache
- Bottleneck: Cannot fit large batches due to activation memory
- Max batch: ~32-48 (activation limited)

**TP=8 (Full Sharding Across 8 Cards):**
- Weights per card: 14 GB / 8 = 1.75 GB
- Activations per card: divided by 8
- KV cache distributed across cards
- Optimal for A100 NVLink topology
- Max batch: limited by compute, not memory (~128-256)

**TP=4 (Partial Sharding):**
- Weights per card: 14 GB / 4 = 3.5 GB
- Requires 2 instances for 8 cards (DP=2)
- Less efficient NVLink communication

### 3.2 Recommended: TP=8 Configuration

```
Memory breakdown per card at TP=8, batch=128:
- Model weights (sharded): 1.75 GB
- Activation (Prefill): 256/8 = 32 GB
- KV cache (640 tokens x 128 samples): 10 GB / 8 = 1.25 GB
- Total per card: ~35 GB (within 80 GB)
```

---

## 4. Maximum Concurrent Batch Analysis

### 4.1 Memory-Limited Single-Card Batch (TP=1)

| Batch | Prefill Activation | KV Cache (640 tok) | Model Weights | Total | Feasible |
|-------|-------------------|-------------------|---------------|-------|----------|
| 32 | ~8 GB | 4 MB x 32 = 128 MB | 14 GB | ~22 GB | Yes |
| 48 | ~12 GB | 6 MB x 48 = 288 MB | 14 GB | ~26 GB | Yes |
| 64 | ~16 GB | 8 MB x 64 = 512 MB | 14 GB | ~31 GB | Yes |
| 96 | ~24 GB | 12 MB x 96 = 1.2 GB | 14 GB | ~39 GB | Yes |
| 128 | ~32 GB | 16 MB x 128 = 2 GB | 14 GB | ~48 GB | Yes |
| 192 | ~48 GB | 24 MB x 192 = 4.6 GB | 14 GB | ~67 GB | Yes (tight) |
| 256 | ~64 GB | 32 MB x 256 = 8 GB | 14 GB | ~86 GB | No |

**TP=1 Max Batch: ~192 (limited by 80GB memory)**

### 4.2 Compute-Limited TP=8 Batch

With TP=8, memory is no longer the bottleneck. Batch is limited by GPU compute throughput.

```
A100-80GB SXM GPU specs:
- FP16 Tensor Core: 312 TFLOPS
- Memory Bandwidth: 2 TB/s

Prefill compute requirement (512 tokens, batch=B):
- Attention: O(B x 512^2 x 32) = O(B x 8.4M) ops per layer
- MLP: O(B x 512 x 4096 x 32) = O(B x 67M) ops per layer
- Total per layer: ~75M x B ops
- Total for 32 layers: ~2.4B x B ops

Decode compute requirement (per token, batch=B):
- Attention: O(B x 512 x 32) = O(B x 16K) ops per layer
- MLP: O(B x 4096 x 32) = O(B x 131K) ops per layer
- Total per layer: ~147K x B ops
- Total for 32 layers: ~4.7M x B ops
```

### 4.3 Maximum Concurrent Requests Summary

| Configuration | Max Concurrent Requests | Limiting Factor |
|---------------|------------------------|-----------------|
| TP=1, single card | ~192 | Memory (80GB) |
| TP=8, 8 cards | **~256-512** | Compute throughput |
| TP=4 + DP=2 | ~256 | Compute + memory |

**Answer: Maximum concurrent batch = 256-512 requests**

---

## 5. PD Strategy Decision

### 5.1 Workload Characterization

```
Total tokens per request = 512 (input) + 128 (output) = 640 tokens
Prefill portion: 512/640 = 80%
Decode portion: 128/640 = 20%

Prefill:Decode ratio = 4:1
```

### 5.2 Decision Criteria

| Criterion | Value | Threshold | Indicates |
|-----------|-------|-----------|-----------|
| Input length (512) | < 2048 | 2048 | Mixed (not long enough for auto-separation) |
| Output length (128) | < 256 | 256 | Mixed (not short enough for auto-separation) |
| Input/Output ratio | 4:1 | > 4:1 for prefill-separation | Prefill dominant but not extreme |
| Output/Input ratio | 1:4 | > 1:2 for decode-separation | Decode is minority |
| Prefill compute fraction | 80% | > 90% for separation consideration | Should use mixed |

### 5.3 PD Separation Candidates (when to separate)

PD Separation is recommended when:
1. Input length > 2048 AND output > 512
2. High variance in input length (std > 30% of mean)
3. Strict latency SLO requiring independent scaling
4. Decode-heavy workload (output >> input)

### 5.4 PD Mixed Candidates (this case)

PD Mixed is recommended when:
1. Moderate input lengths (512 < 2048)
2. Short outputs (128 < 256)
3. Prefill:Decode ratio < 4:1
4. Simpler deployment is preferred
5. Memory efficiency needed (single model copy)

### 5.5 Recommendation: **PD Mixed (Hybrid)**

| Factor | Analysis | Recommendation |
|--------|----------|----------------|
| Input length (512) | Moderate, not extreme | Mixed OK |
| Output length (128) | Short, 20% of work | Mixed OK |
| Prefill fraction (80%) | Dominant but not extreme | Mixed OK |
| Memory efficiency | Single model copy | Mixed preferred |
| Deployment complexity | Lower with mixed | Mixed preferred |
| Throughput | Similar for both | Mixed preferred |

**PD Separation is NOT recommended because:**
- Requires 2x model copies (16 cards for same throughput)
- Overhead likely exceeds benefits for this workload
- Decode is only 20% of total work

---

## 6. Deployment Configuration

### 6.1 Recommended Configuration

```
Configuration: PD Mixed (Continuous Batching)
Parallel Strategy: TP=8 (all 8 cards as single instance)
Batch Size: 64-128 (recommended for production)
Max Concurrent Requests: 256-512
Continuous Batching: Enabled
```

### 6.2 Detailed Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Total GPUs | 8 | Full NVLink topology |
| Tensor Parallel | 8 | Optimal for A100 NVLink |
| Pipeline Parallel | 1 | Not needed |
| Expert Parallel | 1 | Not needed for dense model |
| Data Parallel | 1 | Single instance |
| **Max Concurrent Batch** | **256-512** | Compute-limited |
| **Recommended Batch** | **64-128** | Production recommendation |
| Continuous Batching | Yes | Dynamic request scheduling |

### 6.3 Memory Layout (TP=8, batch=128)

| Component | Per Card | Total (8 cards) |
|-----------|----------|-----------------|
| Model Weights (sharded) | 1.75 GB | 14 GB |
| KV Cache (640 tokens x 128 samples) | 1.25 GB | 10 GB |
| Activation (Prefill phase) | 32 GB | 256 GB total |
| **Total Per Card** | **~35 GB** | 280 GB |

### 6.4 Alternative Configurations

**Configuration A: High Throughput (batch=256)**
- Memory per card: ~50 GB
- Achieves max throughput
- Good for offline/batch processing

**Configuration B: Low Latency (batch=64)**
- Memory per card: ~22 GB
- Lower latency per request
- Good for online serving with latency SLO

---

## 7. Throughput & Latency Estimates

### 7.1 Throughput Estimates

**Prefill Phase (512 tokens input):**
```
Throughput = batch_size x tokens_per_second_per_gpu x num_gpus x gpu_utilization

At batch=128, TP=8:
- Estimated prefill throughput: 50,000-100,000 tok/s
- Time for 512 tokens: ~5-10 ms per sample
```

**Decode Phase (128 tokens output):**
```
At batch=128:
- Decode throughput per card: ~500-800 tok/s (memory-bandwidth bound)
- Total decode throughput: ~4,000-6,400 tok/s
- Time for 128 tokens: ~20-32 ms per token
```

### 7.2 End-to-End Latency

| Metric | Estimate | Notes |
|--------|----------|-------|
| **TTFT (Time to First Token)** | 50-100 ms | Prefill for 512 tokens at batch 128 |
| **TPOT (Time Per Output Token)** | 15-25 ms | Decode at batch 128 |
| **Total Latency (512+128)** | 2.5-4.3 sec | 512/15 + 128x25ms |

### 7.3 Concurrent Request Capacity

| Batch Size | Concurrent Requests | Per-Card Memory | Use Case |
|------------|--------------------|--------------------|----------|
| 64 | 64 | ~22 GB | Low latency, latency-critical |
| 128 | 128 | ~35 GB | Balanced |
| 256 | 256 | ~50 GB | High throughput |
| 512 | 512 | ~70 GB | Max capacity |

---

## 8. Summary & Recommendations

### 8.1 Question 1: Maximum Concurrent Batch?

**Answer: 256-512 concurrent requests**

| Configuration | Max Concurrent | Limiting Factor |
|---------------|----------------|-----------------|
| Single card (TP=1) | ~192 | Memory (80GB) |
| 8 cards (TP=8) | 256-512 | Compute throughput |
| Recommended | 64-128 | Balanced production |

### 8.2 Question 2: PD Mixed or Separation?

**Answer: PD Mixed (Hybrid) is recommended**

| Consideration | Analysis | Decision |
|---------------|---------|----------|
| Input/Output ratio | 512:128 = 4:1 | Prefill dominant but decode non-trivial |
| Input length | 512 (moderate) | Not extreme enough for separation |
| Output length | 128 (short) | Decode is only 20% of work |
| Memory efficiency | Single model copy | PD Mixed saves 50% memory |
| Deployment complexity | Lower with mixed | PD Mixed simpler |
| Throughput efficiency | Similar | No advantage to separation |

### 8.3 Final Configuration

```
Model: Qwen-7B
Hardware: 8x NVIDIA A100-80GB
Parallel Strategy: TP=8 (all cards)
PD Strategy: PD Mixed (Continuous Batching)
Max Concurrent Batch: 256-512 (compute limited)
Recommended Batch: 64-128 (production)
Expected Throughput: 50,000-100,000 tokens/s (prefill)
                     4,000-6,400 tokens/s (decode)
TTFT: 50-100 ms
TPOT: 15-25 ms
```

---

## Appendix: Key Formulas

| Formula | Calculation |
|---------|-------------|
| Model weights (FP16) | 7B x 2 = 14 GB |
| Model weights (INT8) | 7B x 1 = 7 GB |
| KV cache per token (GQA) | 2 x 8 x 128 x 32 x 2 = 128 KB |
| TP=8 sharded weights | 14 GB / 8 = 1.75 GB per card |
| Activation memory (prefill) | batch x 256 MB (approx) |
| Max batch (memory-bound, TP=1) | (80 - 14) / 0.5 = ~132 |
| Max batch (compute-bound, TP=8) | 256-512 |

---

*Generated by LLM PD Deployment Planner*
*Date: 2026-03-22*