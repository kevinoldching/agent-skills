# LLM PD 部署规划

## 1. 输入摘要

| 字段 | 值 |
|------|-----|
| 模型 | Qwen-7B |
| 硬件 | A100-80GB × 8 卡 |
| 输入长度 | 512 tokens（平均） |
| 输出长度 | 128 tokens（平均） |
| 场景 | 通用推理（默认在线推理/chat） |
| 性能目标 | 最大并发 batch（吞吐量优先） |

---

## 2. PD 策略决策

### 决策结果
**推荐策略: PD 混部（Mixed Prefill/Decode）**

### 决策依据

| 优先级 | 因素 | 分析 |
|--------|------|------|
| 场景 | 在线推理/通用chat | 吞吐量优先，PD混部可最大化批量处理效率 |
| 长度比 | α = 0.57 | α ∈ [0.3, 0.7]，基本平衡，imbalance = 0.43 < 0.8，支持混部 |
| SLO | 最大并发batch | 混部无 KV transfer 开销，所有卡可用于 decode，吞吐量更高 |

### 长度比计算
```
α = S_in / (S_in + k × S_out)，k = 3 (default)
α = 512 / (512 + 3 × 128) = 512 / 896 = 0.571
imbalance = max(α, 1-α) = max(0.571, 0.429) = 0.571
判定: 0.571 < 0.8 → 混部策略可行
```

### PD 分离 vs 混部对比

| 指标 | PD 混部 | PD 分离 |
|------|---------|---------|
| KV Cache 传输开销 | 无 | Prefill → Decode 传输有开销 |
| 延迟 | 适中 | 可更低（分离优化） |
| 吞吐量 | 更高（无传输开销） | 较低 |
| 部署复杂度 | 较低 | 较高 |
| 适用场景 | 吞吐量优先 | 延迟敏感 |

**结论：追求最大并发 batch 场景下，PD 混部是更优选择。**

---

## 3. 显存分析

### 模型显存占用（Qwen-7B FP16）

| 项目 | 显存占用 |
|------|----------|
| 模型权重（FP16） | ~14 GB |
| Embedding/LM Head | ~1 GB |
| Activations（估算） | ~1-2 GB |
| 单卡总开销（无 KV Cache） | ~16 GB |

### KV Cache 显存估算

Qwen-7B 架构参数：
- Hidden size: 4096
- Num attention heads: 32
- Head dim: 128
- Num layers: 32

KV Cache 每 token 每层计算：
```
KV per token per layer = 2 × num_heads × head_dim × 2(bytes)
                       = 2 × 32 × 128 × 2 = 16,384 bytes ≈ 16 KB
```
- **每 token KV Cache: ~16 KB**
- **单序列（512 + 128 = 640 tokens）KV Cache: 640 × 16 KB ≈ 10 MB**

### 单卡最大 batch 估算（A100-80GB）

场景：TP=1，8 独立实例，每实例独享 80GB

| batch_size | 权重 | KV Cache (640 tokens) | Activation | 总占用 | 可用余量 |
|------------|------|------------------------|------------|--------|----------|
| 16 | 14 GB | 160 MB | 1 GB | ~15.2 GB | ~64 GB |
| 32 | 14 GB | 320 MB | 2 GB | ~16.3 GB | ~63 GB |
| 48 | 14 GB | 480 MB | 3 GB | ~17.5 GB | ~62 GB |
| 64 | 14 GB | 640 MB | 4 GB | ~18.6 GB | ~61 GB |

单实例单卡可支持 batch_size ≈ 48-64（保守估算 48）

### 并行配置显存验证

**约束条件**：
- TP <= 单机卡数（8）或 TP = 单机卡数 × 机器数量
- TP × DP <= EP（仅 MoE 模型）
- EP < 实例总卡数
- TP/EP/DP 为 2 的幂（1, 2, 4, 8, 16, 32）
- Qwen-7B 为 Dense 模型，EP = 1

| 配置 | 权重/卡 | KV Cache/卡 | 总占用/卡 | 可用 |
|------|---------|-------------|----------|------|
| TP=1, EP=1, DP=1 (×8实例) | 14 GB | 随 batch | ~16-20 GB | 60-64 GB |
| TP=4, EP=1, DP=2 | 3.5 GB | 随 batch | ~5-8 GB | 72-75 GB |
| TP=8, EP=1, DP=1 | 1.75 GB | 随 batch | ~3-6 GB | 74-77 GB |

---

## 4. 并行策略配置

### PD 混部部署

#### 推荐配置: TP=4, EP=1, DP=2

| 参数 | 值 |
|------|-----|
| 总 GPU 数 | 8 |
| Tensor Parallel | 4 |
| Expert Parallel | 1（Dense 模型，无 MoE） |
| Data Parallel | 2 |
| 模型副本数 | 2 |
| 每副本卡数 | 4 |

#### 备选配置: TP=8, EP=1, DP=1

| 参数 | 值 |
|------|-----|
| 总 GPU 数 | 8 |
| Tensor Parallel | 8 |
| Expert Parallel | 1（Dense 模型） |
| Data Parallel | 1 |
| 模型副本数 | 1 |
| 每副本卡数 | 8 |

### 配置对比分析

| 指标 | TP=4, DP=2 | TP=8, DP=1 |
|------|-------------|------------|
| 最大 batch/replica | 48-64 | 32-40 |
| 总最大 batch | 96-128 | 32-40 |
| 通信开销 | 中等（4卡内 TP） | 较低（8卡内 TP） |
| 容错性 | 较好（2副本） | 一般（1副本） |
| 推荐场景 | 吞吐量优先 | 延迟敏感 |

**若追求最大并发 batch，强烈推荐 TP=4, DP=2 配置**

### 预估性能（TP=4, DP=2, batch=48/replica）

基于 Qwen-7B 单卡推理性能估算（A100 BF16 算力 312 TFLOPS）：

**Prefill 阶段**：
- Prefill 算量：7B × 512 tokens × 2 FLOPs/param ≈ 7.2 TFLOPs
- Prefill 时延（TP=4）：~50-100 ms（取决于 batch）
- 吞吐量：512 tokens / 0.075s ≈ 6,800 tokens/s

**Decode 阶段**：
- Decode 算量：7B × 1 token × 2 FLOPs/param ≈ 14 GFLOPs/token
- Decode 时延（TP=4）：~10-20 ms/token
- 单 replica TPS：48 tokens / 0.015s ≈ 3,200 tokens/s
- **总 TPS（2副本）：~6,400 tokens/s**

| 指标 | 预估 |
|------|------|
| 最大并发 batch | **48-64 每副本，总计 96-128** |
| TTFT（batch=48） | 50-100 ms |
| TPOT | 10-20 ms |
| 总吞吐量 | ~6,000-8,000 tokens/s |

---

## 5. 约束验证

| 约束 | 配置 TP=4, DP=2 | 状态 |
|------|-----------------|------|
| TP <= 单机卡数 | 4 <= 8 | 通过 |
| TP × DP <= EP | 4 × 2 = 8 <= 1 | N/A（Dense 模型） |
| EP < 实例总卡数 | EP=1 < 8 | 通过 |
| EP < expert 数量 | N/A（Dense） | N/A |
| TP/EP/DP 为 2 的幂 | 4, 1, 2 | 通过 |
| 总卡数 <= 可用卡数 | 8 <= 8 | 通过 |
| 总卡数为单机整数倍 | 8 % 8 = 0 | 通过 |

---

## 6. 最大并发 Batch 总结

### 结论

对于 **8 张 A100-80GB 卡部署 Qwen-7B**：

| 配置 | 最大并发 batch |
|------|---------------|
| **TP=4, DP=2（推荐）** | **96-128** |
| TP=8, DP=1 | **32-40** |

**如果追求最大并发 batch，建议使用 TP=4, DP=2，总最大并发 batch 可达 96-128。**

### PD 策略建议

**使用 PD 混部策略**，理由：
1. 长度比 α=0.57 属于基本平衡范围（0.3-0.7）
2. 混部无 KV transfer 开销，带宽利用率更高
3. 对于 batch 优先场景，混部 TPS 更高
4. Qwen-7B 属于小模型，单卡即可承载，PD 分离收益有限

---

## 7. 实现注意事项

1. **Batch Size 调度**：建议使用 vLLM 或 SGLang 的自适应 batch 策略，根据实际负载动态调整
2. **KV Cache 管理**：使用 PagedAttention 优化显存利用，可提升有效 batch 上限 20-30%
3. **Prefill/Decode 混合**：在混部策略下，注意控制 prefill 和 decode 请求的比例，避免 compute bound
4. **监控指标**：重点关注 GPU 利用率、显存使用率、TPS、TTFT、TPOT
5. **若遇显存不足**：适当降低 batch 或启用 TP=8 降低单卡显存压力

---

## 附录：关键公式

| 公式 | 说明 |
|------|------|
| `Decode TPS_混部 = batch_size × (S_in + S_out) / (TTFT + S_out × TPOT)` | 混部 TPS |
| `KV Cache/卡 = batch × seq_len × 16KB / TP` | 每卡 KV Cache 估算 |
| `batch_max ≈ (80GB - 16GB) / (seq_len × 16KB)` | 单卡最大 batch 估算 |
| `α = S_in / (S_in + k × S_out)` | 长度比计算 |
| `imbalance = max(α, 1-α)` | 失衡度计算，< 0.8 支持混部 |

---

## 附录：Qwen-7B 模型规格

| 参数 | 值 |
|------|-----|
| 参数量 | 7B |
| Hidden size | 4096 |
| Num layers | 32 |
| Num attention heads | 32 |
| Head dim | 128 |
| Vocab size | 151936 |
| Max sequence length | 8192 |
| 权重（FP16） | ~14 GB |