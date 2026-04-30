# LLM PD 部署规划

## 1. 输入摘要

| 字段 | 值 |
|------|-----|
| **模型** | Kimi-K2.5 (MoE) |
| **硬件** | Ascend-910B-64GB × 8卡 (单机8卡配置) |
| **输入长度** | 1024 tokens (平均) |
| **输出长度** | 512 tokens (平均) |
| **场景** | Chat |
| **性能目标** | TPS优先 |

---

## 2. 模型与硬件参数

### 2.1 Kimi-K2.5 模型规格

| 参数 | 值 |
|------|-----|
| 参数量 | 170.74B |
| Hidden Size | 7168 |
| Num Layers | 61 |
| Vocab Size | 163840 |
| Max Sequence Length | 32768 |
| 专家数量 | 384 (MoE) |
| 每Token激活专家数 | 8 |
| MoE Intermediate Size | 2048 |
| Attention Type | MLA (Multi-head Latent Attention) |
| KV LoRA Rank | 512 |
| QK RoPE Head Dim | 64 |
| V Head Dim | 128 |
| 量化方式 | INT8 (MoE experts) / BF16 (attention, embedding) |

### 2.2 Ascend-910B-64GB 硬件规格

| 参数 | 值 |
|------|-----|
| 每卡显存 | 64 GB (HBM) |
| 显存带宽 | 1200 GB/s |
| 互联 | HCCL (华为集合通信库) |
| 典型配置 | 8卡/单机 |

---

## 3. PD 策略决策

### 3.1 三优先级决策分析

**第一优先级: 场景分析**

| 场景 | 本部署 | 推荐策略 |
|------|--------|---------|
| 强化学习/数据生产/Agent轨迹 | 否 | PD混部 |
| **Chat/在线推理** | **是** | **PD混部** |

**决策结果: PD混部**

**分析**: Chat场景为在线推理，对TPS(吞吐量)有较高要求。PD混部可避免KV Cache传输开销，所有卡共同参与计算，在TPS优先目标下是最优选择。

**第二优先级: 输入输出长度比**

```
α = S_in / (S_in + k × S_out), k=3 (default)
α = 1024 / (1024 + 3 × 512) = 1024 / 2560 = 0.4
imbalance = max(α, 1-α) = 0.6
imbalance < 0.8 → 可接受混部
```

| α 值 | 含义 | 本部署 |
|------|------|--------|
| 0.4 | 轻度偏向输入 | 可接受混部 |
| < 0.2 或 > 0.8 | 明显失衡 | 否 |

**第三优先级: SLO指标**

| 指标 | 要求 | 决策 |
|------|------|------|
| 优先级 | TPS优先 | **PD混部** - 无KV传输开销，所有卡参与decode |

### 3.2 策略决策结果

**推荐策略: PD 混部 (Mixed)**

**决策依据:**

| 优先级 | 因素 | 分析结果 |
|--------|------|---------|
| 场景 | Chat在线推理 | **PD混部** - 简化部署，减少阶段间开销 |
| 长度比 | α=0.4, imbalance=0.6 | 基本平衡，混部可接受 |
| SLO | TPS优先 | **PD混部** - 无KV Cache传输开销，最大化TPS |

### 3.3 PD分离 vs PD混部对比

| 维度 | PD混部 (推荐) | PD分离 |
|------|--------------|--------|
| 部署复杂度 | 低 | 中 |
| 阶段间协调开销 | 无 | KV Cache传输开销 |
| 扩展性 | 统一扩展 | 独立扩展 |
| TPS优化 | **最优** | 一般 |
| 卡间负载均衡 | 静态均衡 | 需动态调配 |

---

## 4. 显存分析

### 4.1 模型显存占用

**权重估算 (FP16基准)**:
- MoE Experts: 170.74B × 2 bytes ≈ **341 GB**
- Attention + Embedding + Norms: ~15 GB
- **总计**: ~356 GB (FP16)

**INT8量化后 (推荐)**:
- MoE Experts (INT8): ~170 GB
- Attention + Embedding + Norms (BF16): ~15 GB
- **总计**: ~185 GB

### 4.2 KV Cache 计算

使用MLA (Multi-head Latent Attention) 的KV Cache公式:
```
KV Cache = batch_size × seq_len × (kv_lora_rank + qk_rope_head_dim) × num_layers
KV Cache = batch_size × seq_len × (512 + 64) × 61
KV Cache = batch_size × seq_len × 35136
```

| Batch Size | Seq Length | KV Cache/卡 (TP=8) |
|------------|------------|-------------------|
| 8 | 1536 | 8 × 1536 × 35136 / 8 / 1024³ ≈ 5.1 GB |
| 16 | 1536 | 10.2 GB |
| 32 | 1536 | 20.4 GB |

### 4.3 单卡显存分析 (TP=8, EP=1)

| 组件 | 计算公式 | 估算值 |
|------|---------|--------|
| 权重 (INT8) | 341 GB / 8 | 42.6 GB |
| Attention/Embedding (BF16) | ~15 GB / 8 | 1.9 GB |
| KV Cache (B=16, S=1536) | 16 × 1536 × 35136 / 8 / 1024³ | 10.2 GB |
| 激活值 | batch × seq × hidden × experts_per_tok / TP | ~3 GB |
| **总计** | | **~57.7 GB** |

**结论**: TP=8配置下，单卡总占用约58 GB < 64 GB，**满足要求**。

---

## 5. 并行策略配置

### 5.1 约束条件验证

| 约束 | 要求 | 验证结果 |
|------|------|---------|
| TP <= 单机卡数 | TP=8 <= 8 | **满足** ✓ |
| TP × DP = EP | EP=1, TP=8, DP=0.125 | **不适用** (EP=1固定) |
| MoE模型 | EP可以>1 | EP=1为保守配置 |
| TP/EP/DP 为2的幂 | 1,2,4,8,16,32 | TP=8 ✓ |
| 总卡数为单机整数倍 | 8 % 8 = 0 | **满足** ✓ |

### 5.2 推荐配置: PD混部

**总卡数: 8**
**并行策略: TP=8 + EP=1 + DP=1 (单实例)**

| 参数 | 值 | 说明 |
|------|-----|------|
| 总GPU数 | 8 | 单机8卡 |
| Tensor Parallel | 8 | 权重分片+计算并行 |
| Expert Parallel | 1 | 保守配置(专家全部在一个TP组内) |
| Data Parallel | 1 | 单实例 |
| 每卡显存占用 | ~58 GB | < 64 GB限制 |
| 最大Batch Size | 16 | 预留约6 GB余量 |

### 5.3 替代配置对比

#### 配置1: TP=8, EP=1, DP=1 (推荐-均衡)

| 阶段 | 卡数 | TP | EP | DP | 权重/卡 | KV Cache/卡 | 可用余量 |
|------|------|----|----|----|---------|------------|---------|
| 混部 | 8 | 8 | 1 | 1 | 43 GB | 10 GB (B=16) | ~11 GB |

- 优点: 部署简单，KV Cache空间充足
- 缺点: 未利用MoE的Expert Parallel特性

#### 配置2: TP=4, EP=2, DP=1

| 阶段 | 卡数 | TP | EP | DP | 权重/卡 | KV Cache/卡 | 可用余量 |
|------|------|----|----|----|---------|------------|---------|
| 混部 | 8 | 4 | 2 | 1 | 85 GB | 20 GB (B=16) | **溢出** |

- 优点: 利用EP分担专家存储
- 缺点: 权重+KV Cache > 64 GB，**不可行**

#### 配置3: TP=4, EP=8, DP=1 (8卡) → 需要8卡，EP=8满足

| 阶段 | 卡数 | TP | EP | DP | 权重/卡 | KV Cache/卡 | 可用余量 |
|------|------|----|----|----|---------|------------|---------|
| 混部 | 8 | 4 | 2 | 1 | 85 GB | 20 GB (B=16) | **溢出** |

- 分析: TP=4时每卡权重85GB已超限，即使EP=2也无法解决

**最终推荐: 配置1 (TP=8, EP=1, DP=1)**

---

## 6. 性能估算

### 6.1 吞吐量公式

**混部 TPS**:
```
TPS = batch_size × (S_in + S_out) / (TTFT + S_out × TPOT)
```

### 6.2 性能计算

**参数假设**:
- 层数: 61
- 隐藏维度: 7168
- Ascend-910B FP16 算力: ~256 TFLOPS (估算)
- 并行效率因子: η ≈ 0.7

**Prefill阶段 (TP=8)**:
```
TTFT ≈ S_in / (TP × η × f_layer)
f_layer ≈ 1 layer per ms (估算)
TTFT ≈ 1024 / (8 × 0.7 × 1) ≈ 183 ms
```

**Decode阶段 (TP=8)**:
```
TPOT ≈ 1 / (TP × η × f_per_token)
TPOT ≈ 1 / (8 × 0.7 × 1) ≈ 0.18 ms per token
```

**吞吐量估算**:
```
Decode TPS = batch × (S_in + S_out) / (TTFT + S_out × TPOT)
           = 16 × 1536 / (183ms + 512 × 0.18ms)
           = 24576 / (183 + 92)
           = 24576 / 275
           ≈ 89 tokens/s per batch iteration

若每秒可处理 ~4-5 个batch (考虑生成512 token需512次decode step):
预估总TPS ≈ 89 × 4-5 ≈ 350-450 tokens/s
```

### 6.3 与其他配置对比

| 配置 | TP | EP | DP | Batch | 预估TPS | 可行性 |
|------|----|----|----|-------|---------|--------|
| TP=8, EP=1, DP=1 | 8 | 1 | 1 | 16 | 350-450 | **可行** |
| TP=4, EP=2, DP=1 | 4 | 2 | 1 | 8 | 200-280 | 内存溢出 |
| TP=2, EP=4, DP=1 | 2 | 4 | 1 | 4 | 100-150 | 内存溢出 |

---

## 7. 显存约束验证

### 7.1 TP=8 配置详细验证

```
每卡权重 (INT8 MoE + BF16 Attention):
  - MoE Experts: 341 GB / 8 = 42.6 GB
  - Attention (q_a, q_b, kv_a, kv_b, o_proj): ~4.5 GB
  - Embedding + Norms: ~1.5 GB
  - 总计: ~48.6 GB

每卡KV Cache (batch=16, seq=1536):
  - KV = 16 × 1536 × (512 + 64) × 61 / 8 bytes
  - = 16 × 1536 × 35136 / 8 / 1024³ GB
  - ≈ 10.2 GB

每卡激活值 (batch=16, seq=1024):
  - 激活 ≈ batch × seq × hidden × experts_per_tok × 2 / TP
  - = 16 × 1024 × 7168 × 8 × 2 / 8 / 1024³ GB
  - ≈ 1.8 GB

每卡总占用: 48.6 + 10.2 + 1.8 + 系统预留 ≈ 62 GB < 64 GB ✓
```

### 7.2 EP=8 配置尝试 (TP=4, EP=2)

```
每卡权重 (TP=4, EP=2):
  - MoE Experts: 341 GB / (4×2) = 42.6 GB (分片后)
  - Attention: ~9 GB (TP=4分片)
  - Embedding: ~6 GB (TP分片)
  - 总计: ~57.6 GB

KV Cache (batch=16, seq=1536, TP=4):
  - = 16 × 1536 × 35136 / 4 / 1024³ ≈ 20.4 GB

每卡总占用: 57.6 + 20.4 + 1.8 ≈ 80 GB > 64 GB ✗
```

**结论**: TP=4即使配合EP=2也会导致内存溢出，TP=8是唯一可行配置。

---

## 8. 实现注意事项

### 8.1 关键瓶颈与优化

1. **MoE计算瓶颈**
   - 每次前向需激活8/384个专家
   - 建议启用专家级负载均衡，避免部分专家过热
   - HCCL通信需针对All-to-All优化

2. **MLA注意力优化**
   - MLA使用低秩分解，需确保kv_a_proj计算效率
   - Q和K使用共享LoRA投影，减少计算量

3. **内存带宽优化**
   - Decode阶段内存带宽受限
   - 建议启用Continuous Batching提高GPU利用率

### 8.2 优化建议

| 优化项 | 说明 | 预期收益 |
|--------|------|---------|
| INT8量化 | MoE专家保持INT8 | 权重内存减半 |
| Continuous Batching | 动态batch调度 | 吞吐量提升30-50% |
| Prefix Caching | 复用chat历史KV Cache | TTFT降低40-60% |
| CUDA Graphs | 减少kernel启动开销 | TPOT降低10-15% |
| 异步通信 | HCCL与计算重叠 | 隐藏通信延迟 |

### 8.3 EP扩展考虑

若未来需要更大规模部署:

| 配置 | TP | EP | DP | 总卡数 | 每卡权重 | 每卡KV(B=16) |
|------|----|----|----|-------|---------|-------------|
| 8卡基准 | 8 | 1 | 1 | 8 | 48 GB | 10 GB |
| 16卡扩展 | 8 | 2 | 2 | 16 | 24 GB | 5 GB |
| 32卡扩展 | 8 | 4 | 4 | 32 | 12 GB | 2.5 GB |

**注意**: EP增加可线性降低每卡内存占用，但需权衡通信开销。

---

## 9. 最终配置总结

### 9.1 推荐配置 (PD混部)

```yaml
deployment:
  strategy: PD Mixed (混部)
  total_cards: 8

  parallel_strategy:
    tp: 8
    ep: 1
    dp: 1

  memory_per_card:
    weights: ~48.6 GB
    kv_cache: ~10.2 GB (B=16)
    activation: ~1.8 GB
    total: ~62 GB

  performance:
    max_batch_size: 16
    estimated_tps: 350-450 tokens/s
    estimated_ttft: ~180-200 ms
    estimated_tpot: ~0.15-0.2 ms
```

### 9.2 需求满足度

| 指标 | 目标 | 实际 | 满足 |
|------|------|------|------|
| TPS | 越高越好 | ~350-450 tokens/s | ✓ |
| 显存限制 | < 64 GB/卡 | ~62 GB | ✓ |
| Batch Size | 最大化 | 16 | ✓ |
| 部署复杂度 | 低 | 单实例混部 | ✓ |

### 9.3 约束验证

- [x] TP=8 <= 单机卡数=8
- [x] MoE模型: EP可>1 (当前EP=1为保守配置)
- [x] TP/EP/DP 为2的幂: TP=8 ✓
- [x] 总卡数为单机卡数整数倍: 8 % 8 = 0
- [x] 每卡显存 < 64 GB: ~62 GB ✓

---

## 10. 附录: 关键公式参考

| 公式 | 说明 |
|------|------|
| `MoE权重(FP16) = 170.74B × 2 bytes` | 341 GB |
| `KV Cache = B × S × (kv_lora_rank + qk_rope_head_dim) × L` | MLA格式 |
| `混部TPS = B × (S_in + S_out) / (TTFT + S_out × TPOT)` | 吞吐量公式 |
| `TTFT ≈ S_in / (TP × η × f)` | 首Token延迟 |
| `TPOT ≈ 1 / (TP × η × f_per_token)` | 每Token延迟 |

---

## 11. 配置速查表

| 项目 | 值 |
|------|-----|
| 部署策略 | **PD混部 (Mixed)** |
| 模型 | Kimi-K2.5 (MoE, 170.74B) |
| 总GPU数 | 8 |
| Tensor Parallel | 8 |
| Expert Parallel | 1 |
| Data Parallel | 1 |
| 每卡显存占用 | ~62 GB |
| 最大Batch Size | 16 |
| 预估吞吐量 | 350-450 tokens/s |
| 预估TTFT | 180-200 ms |
| 预估TPOT | 0.15-0.2 ms |
