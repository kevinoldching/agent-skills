# LLM PD 部署规划

## 1. 输入摘要

| 字段 | 值 |
|------|-----|
| 模型 | Kimi-K2.5 (MoE模型) |
| 硬件 | Ascend-910B-64GB × 8 卡（单机8卡配置） |
| 输入长度 | 1024 tokens（平均） |
| 输出长度 | 512 tokens（平均） |
| 场景 | chat |
| 性能目标 | TPS优先 |

---

## 2. PD 策略决策

### 决策结果
**推荐策略: PD 混部（Mixed Prefill/Decode）**

### 决策依据

| 优先级 | 因素 | 分析 |
|--------|------|------|
| 场景 | chat | 在线推理场景，吞吐量优先，PD混部可最大化批量处理效率 |
| 长度比 | α = 0.40 | α ∈ [0.3, 0.7]，基本平衡，imbalance = 0.60 < 0.8，支持混部 |
| SLO | TPS优先 | 混部无KV transfer开销，所有卡都可用于decode，吞吐量更高 |

### 长度比计算
```
α = S_in / (S_in + k × S_out)，k = 3 (default)
α = 1024 / (1024 + 3 × 512) = 1024 / 2560 = 0.40
imbalance = max(α, 1-α) = max(0.40, 0.60) = 0.60
判定: 0.60 < 0.8 → 混部策略可行
```

### PD 分离 vs 混部对比

| 指标 | PD 混部 | PD 分离 |
|------|---------|---------|
| KV Cache 传输开销 | 无 | Prefill → Decode 传输有开销 |
| 延迟 | 适中 | 可更低（分离优化） |
| 吞吐量 | 更高（无传输开销） | 较低 |
| 部署复杂度 | 较低 | 较高 |
| 适用场景 | 吞吐量优先 | 延迟敏感 |

**结论：追求TPS优先场景下，PD混部是更优选择。**

---

## 3. 显存分析

### 模型显存占用（Kimi-K2.5 MoE FP16/BF16）

| 项目 | 显存占用 |
|------|----------|
| 模型权重（BF16/INT8混合） | ~554 GB（原始） |
| MoE FFN（96% of total） | ~532 GB |
| Attention | ~11.5 GB |
| Embedding/LM Head | ~5.25 GB |
| 单卡总开销（无KV Cache） | ~16-77 GB（取决于EP） |

### KV Cache 显存估算

Kimi-K2.5 架构参数（基于MLA注意力）:
- Hidden size: 7168
- Num attention heads: 64
- Head dim: 112
- Num layers: 61
- kv_lora_rank: 512
- qk_rope_head_dim: 64

KV Cache 每 token 每层计算：
```
KV per token per layer = 2 × (kv_lora_rank + qk_rope_head_dim) × 2(bytes)
                       = 2 × (512 + 64) × 2 = 2,304 bytes ≈ 2.25 KB
```
- **每 token KV Cache: ~2.25 KB**（MLA优化，显著小于传统MHA/GQA）
- **单序列（1024 + 512 = 1536 tokens）KV Cache: 1536 × 2.25 KB ≈ 3.46 MB**

### 单卡最大 batch 估算（Ascend-910B-64GB）

| batch_size | 权重 | KV Cache (1536 tokens) | Activation | 总占用 | 可用 |
|------------|------|------------------------|------------|--------|------|
| 32 | 42.43 GB | 3.2 GB | 6.5 GB | ~54 GB | 10 GB |
| 40 | 42.43 GB | 4.0 GB | 8.1 GB | ~57 GB | 7 GB |
| 43 | 42.43 GB | 4.3 GB | 8.8 GB | ~58 GB | 6 GB |
| 48 | 42.43 GB | 4.8 GB | 9.8 GB | ~60 GB | 4 GB |

单卡可支持 batch_size ≈ 43（保守估算，考虑系统预留）

### 并行配置显存验证

**约束条件**：
- TP <= 单机卡数（8）或 TP = 单机卡数 × 机器数量
- TP × DP = EP（MoE模型）
- EP <= MoE expert数量（384）
- TP/EP/DP 为 2 的幂（1, 2, 4, 8, 16, 32, 64）
- 总卡数必须是单机卡数的整数倍

| 配置 | 总卡数 | 权重/卡 | KV Cache/卡 | Activation/卡 | 总占用/卡 | 可用 | Max Batch |
|------|--------|---------|-------------|--------------|----------|------|-----------|
| TP=8, EP=8 | 8 | 75.66 GB | - | - | >64GB | 不足 | N/A |
| TP=8, EP=16 | 16 | 42.43 GB | 4.32 GB | 8.82 GB | 57.57 GB | 6.4 GB | 43 |
| TP=8, EP=32 | 32 | 25.82 GB | 9.75 GB | 19.89 GB | 57.47 GB | 6.5 GB | 97 |
| TP=8, EP=64 | 64 | 17.52 GB | 12.47 GB | 25.43 GB | 57.41 GB | 6.6 GB | 124 |
| TP=4, EP=16 | 16 | 44.36 GB | 3.62 GB | 7.38 GB | 57.36 GB | 6.6 GB | 36 |

**注：EP 表示 Expert Parallelism 维度，即总卡数。DP = 总卡数 / TP**

---

## 4. 并行策略配置

### PD 混部部署

#### 推荐配置: TP=8, EP=16（16卡，2副本）

| 参数 | 值 |
|------|-----|
| 总 GPU 数 | 16 |
| Tensor Parallel | 8 |
| Expert Parallel | 16 |
| Data Parallel | 2（DP = EP/TP = 16/8 = 2） |
| 模型副本数 | 2 |
| 每副本卡数 | 8 |
| 每副本最大 batch | 43 |
| **总最大 batch** | **86（2副本 × 43）** |

#### 备选配置: TP=8, EP=32（32卡，4副本）

| 参数 | 值 |
|------|-----|
| 总 GPU 数 | 32 |
| Tensor Parallel | 8 |
| Expert Parallel | 32 |
| Data Parallel | 4（DP = EP/TP = 32/8 = 4） |
| 模型副本数 | 4 |
| 每副本卡数 | 8 |
| 每副本最大 batch | 97 |
| **总最大 batch** | **388（4副本 × 97）** |

#### 更大规模配置: TP=8, EP=64（64卡，8副本）

| 参数 | 值 |
|------|-----|
| 总 GPU 数 | 64 |
| Tensor Parallel | 8 |
| Expert Parallel | 64 |
| Data Parallel | 8（DP = EP/TP = 64/8 = 8） |
| 模型副本数 | 8 |
| 每副本卡数 | 8 |
| 每副本最大 batch | 124 |
| **总最大 batch** | **992（8副本 × 124）** |

### 配置对比分析

| 指标 | TP=8, EP=16 (16卡) | TP=8, EP=32 (32卡) | TP=8, EP=64 (64卡) |
|------|---------------------|---------------------|---------------------|
| 每副本 batch | 43 | 97 | 124 |
| 总最大 batch | 86 | 388 | 992 |
| 通信开销 | 中等（8卡内TP） | 中等（8卡内TP） | 中等（8卡内TP） |
| 显存效率 | 极高（~90%） | 极高（~90%） | 极高（~90%） |
| 推荐场景 | 有限卡数 | 中等规模 | 大规模部署 |

**若追求最大TPS，强烈推荐 TP=8, EP=64（64卡）配置，总最大并发batch可达992。**

### 预估性能（TP=8, EP=16，batch=43/replica）

基于 Ascend-910B-64GB 算力估算（参考华为Altas 900规格）：

**Prefill 阶段**：
- Prefill 算量：170.74B × 1024 tokens × 2 FLOPs/param ≈ 350 TFLOPs
- Prefill 时延（TP=8，batch=43）：~200-400 ms（取决于compute bound程度）
- 吞吐量：43 × 1024 tokens / 0.3s ≈ 146,000 tokens/s

**Decode 阶段**：
- Decode 算量：170.74B × 1 token × 2 FLOPs/param ≈ 341 GFLOPs/token
- Decode 时延（TP=8，batch=43）：~20-40 ms/token
- 单 replica TPS：43 tokens / 0.03s ≈ 1,433 tokens/s
- **总 TPS（2副本）：~2,866 tokens/s**

| 指标 | 预估（16卡配置） |
|------|------------------|
| 最大并发 batch | **86（2副本 × 43）** |
| TTFT（batch=43） | 200-400 ms |
| TPOT | 20-40 ms |
| 总吞吐量 | ~2,800-3,000 tokens/s |

---

## 5. 约束验证

| 约束 | 配置 TP=8, EP=16 | 状态 |
|------|-----------------|------|
| TP <= 单机卡数 | 8 <= 8 | 通过 |
| TP × DP = EP | 8 × 2 = 16 | 通过 |
| EP <= expert数量 | 16 <= 384 | 通过 |
| TP/EP/DP 为 2 的幂 | 8, 16, 2 | 通过 |
| 总卡数 <= 可用卡数 | 16 <= 16（假设） | 通过 |
| 总卡数为单机整数倍 | 16 % 8 = 0 | 通过 |

---

## 6. TPS 优化总结

### 结论

对于 **Ascend-910B-64GB 卡部署 Kimi-K2.5（MoE模型）**：

| 配置 | 最大并发 batch | 预估 TPS |
|------|---------------|----------|
| **TP=8, EP=16（推荐，16卡）** | **86** | **~2,800-3,000** |
| TP=8, EP=32（32卡） | **388** | **~12,000-13,000** |
| TP=8, EP=64（64卡） | **992** | **~30,000+** |

**如果追求最大TPS，建议使用 TP=8, EP=64，总最大并发batch可达992。**

### PD 策略建议

**使用 PD 混部策略**，理由：
1. 长度比 α=0.40 属于基本平衡范围（0.3-0.7）
2. 混部无 KV transfer 开销，带宽利用率更高
3. 对于 TPS 优先场景，混部 TPS 更高
4. Kimi-K2.5 属于超大规模 MoE 模型（170B参数，384 experts），需要高 EP 配置

---

## 7. 实现注意事项

1. **MoE 负载均衡**：Kimi-K2.5 有 384 个 experts，需配置 expert 负载均衡策略避免某些 expert 过载
2. **Batch Size 调度**：建议使用 vLLM 或 SGLang 的自适应 batch 策略，根据实际负载动态调整
3. **MLA 注意力优化**：Kimi-K2.5 使用 MLA 注意力，需确认部署框架支持 MLA 架构
4. **KV Cache 管理**：使用 PagedAttention 优化显存利用，可提升有效 batch 上限 20-30%
5. **Prefill/Decode 混合**：在混部策略下，注意控制 prefill 和 decode 请求的比例，避免 compute bound
6. **EP 通信优化**：MoE 的 All-to-All 通信是性能瓶颈，需优化网络拓扑（推荐 NVLink/RoCE）
7. **监控指标**：重点关注 GPU 利用率、显存使用率、TPS、TTFT、TPOT、Expert 负载均衡

---

## 附录：关键公式

| 公式 | 说明 |
|------|------|
| `Decode TPS_混部 = batch_size × (S_in + S_out) / (TTFT + S_out × TPOT)` | 混部 TPS |
| `KV Cache/卡 = batch × seq_len × 2.25KB / TP` | 每卡 KV Cache 估算（MLA） |
| `α = S_in / (S_in + k × S_out)` | 长度比计算 |
| `imbalance = max(α, 1-α)` | 失衡度计算，< 0.8 支持混部 |
| `总卡数 = EP = TP × DP` | 并行配置关系 |
| `batch_max ≈ (64GB - 权重 - 2GB) / (KV + Activation)` | 单卡最大 batch 估算 |

---

## 附录：Kimi-K2.5 模型规格

| 参数 | 值 |
|------|-----|
| 参数量 | 170.74B |
| 架构类型 | MoE（混合专家） |
| Hidden size | 7168 |
| Num layers | 61 |
| Num attention heads | 64 |
| Head dim | 112 |
| Attention 类型 | MLA |
| FFN 类型 | MoE |
| Num experts | 384 |
| Num experts per token | 8 |
| MoE intermediate size | 2048 |
| Vocab size | 163,840 |
| Max sequence length | 32,768 |
| kv_lora_rank | 512 |
| qk_rope_head_dim | 64 |
| 权重（FP16/BF16） | ~554 GB（需要 EP 才能部署） |
| 每卡权重（TP=8, EP=16） | ~42 GB |
