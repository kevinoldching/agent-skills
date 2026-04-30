# LLM PD 部署规划

## 1. 输入摘要

| 字段 | 值 |
|------|-----|
| 模型 | MiniMax-M2.5（MoE模型） |
| 硬件 | Ascend 910B-64GB × 8卡（单机） |
| 输入长度 | 1024 tokens |
| 输出长度 | 512 tokens |
| 场景 | chat |
| 性能目标 | TPS优先 |

### 模型规格详情

| 参数 | 值 |
|------|-----|
| 总参数量 | 228.70B |
| 层数 | 62 |
| 隐藏层维度 | 3072 |
| 注意力头数 | 48 |
| KV头数 | 8 (GQA) |
| FFN类型 | MoE |
| MoE专家数 | 256 |
| 每token激活专家数 | 8 |
| 量化方式 | FP8 |

---

## 2. PD 策略决策

### 决策结果

**推荐策略: PD 混部**

### 决策依据

| 优先级 | 因素 | 分析 |
|--------|------|------|
| 场景 | chat | 在线推理场景，但TPS优先时，混部可避免KV cache传输开销，提升整体吞吐 |
| 长度比 | α=0.4 | α=1024/(1024+3×512)=0.4，imbalance=0.6 < 0.8，属于基本平衡，混部可行 |
| SLO | TPS优先 | TPS优先时，混部无KV cache跨实例传输开销，所有卡都参与decode，吞吐更高 |

### 策略对比

| 策略 | 优点 | 缺点 |
|------|------|------|
| PD混部 | 无KV cache传输开销；所有卡参与decode；TPS更高 | Prefill和Decode共享资源，可能相互影响 |
| PD分离 | TTFT/TPOT可独立优化 | KV cache传输开销；Decode只有1个实例，吞吐受限 |

---

## 3. 显存分析

### 模型显存占用（单卡）

| 组件 | 内存 (GB) | 说明 |
|------|----------|------|
| 权重（FP8） | 26.95 | MoE专家权重通过EP=8分片 |
| KV Cache/Token | ~0.05 | 与batch_size成正比 |
| 激活值 | ~0.09 | 与batch_size成正比 |
| 系统预留 | 2.00 | 固定开销 |
| **单卡总计** | **~29.08** | batch_size=1时 |

### 并行配置显存验证

| 配置 | 总卡数 | 权重/卡 | KV Cache | Activation | 总占用 | 可用余量 |
|------|--------|--------|----------|------------|--------|----------|
| TP=8, EP=8, DP=1 | 8 | 26.95 GB | 0.05 GB | 0.09 GB | 29.08 GB | 28.52 GB |
| TP=8, EP=32, DP=4 | 32 | 7.33 GB | 0.05 GB | 0.09 GB | 9.46 GB | 48.14 GB |
| TP=8, EP=64, DP=8 | 64 | 4.06 GB | 0.05 GB | 0.09 GB | 6.19 GB | 51.41 GB |

**结论**: TP=8, EP=8 配置下，单卡显存占用仅29GB，剩余28GB+可用于更大batch。

### 最大Batch Size（TP=8, EP=8, 8卡总计）

| 配置 | 最大Batch Size | 权重 | KV Cache | Activation | 总占用 |
|------|---------------|------|----------|------------|--------|
| BS=214 | 214 | 26.95 GB | 9.72 GB | 18.81 GB | 57.48 GB |

**最大可支持Batch Size: 214**

---

## 4. 并行策略配置

### PD 混部部署（推荐）

**总卡数: 8**

| 参数 | 值 |
|------|-----|
| 总GPU数 | 8 |
| Tensor Parallel (TP) | 8 |
| Expert Parallel (EP) | 8 |
| Data Parallel (DP) | 1 |
| 每卡显存占用 | ~30 GB（BS=16时） |
| 最大Batch Size | 214 |

#### 配置说明

- **TP=8**: 适合单机8卡配置，注意力计算在8卡间按列切分
- **EP=8**: MoE专家（256个）在8卡间分片，每卡承载32个专家
- **DP=1**: 单实例部署，数据并行度为1

### 性能预估

| 指标 | 预估公式 | 预估值 |
|------|----------|--------|
| 最大Batch Size | 显存约束 | 214 |
| Decode TPS | batch × (S_in + S_out) / (TTFT + S_out × TPOT) | ~1500-2000 tokens/s |
| TTFT | num_layers × prefill_time_per_layer | ~50-100ms |
| TPOT | num_layers × decode_time_per_layer | ~10-20ms |

**注**: 具体TPS需根据Ascend 910B实际算力测试确定，以上为理论估算。

### 推荐运行配置

```yaml
# 推荐配置
tp_size: 8
ep_size: 8
dp_size: 1
max_batch_size: 214
prompt_length: 1024
generation_length: 512
```

---

## 5. 备选方案：PD 分离部署

如需进一步优化TTFT/TPOT，可考虑PD分离（但TPS会降低）:

### Prefill 阶段

| 参数 | 值 |
|------|-----|
| 实例数 | 1 |
| 每实例卡数 | 8 (TP×DP) |
| 总卡数 | 8 |
| 并行策略 | TP=8, EP=8, DP=1 |
| 预估吞吐量 | ~10000 tokens/s (Prefill阶段) |

### Decode 阶段

| 参数 | 值 |
|------|-----|
| 实例数 | 1 |
| 每实例卡数 | 8 (TP×DP) |
| 总卡数 | 8 |
| 并行策略 | TP=8, EP=8, DP=1 |
| 预估吞吐量 | ~2000-3000 tokens/s (Decode阶段) |

**总卡数: 16 (Prefill 8 + Decode 8)**

---

## 6. 实现注意事项

1. **EP配置**: MiniMax-M2.5 MoE专家需配置为EP分片，不可用replicated（会导致显存不足）

2. **Batch Size选择**:
   - 在线chat建议batch_size=16-32（延迟较低）
   - 离线批处理可使用batch_size=100-214（吞吐优先）

3. **量化要求**: 当前配置基于FP8量化，需确保Ascend 910B支持FP8计算

4. **单机限制**: TP=8需8卡全部在同一机器上，不支持跨机

5. **KV Cache优化**: 可根据实际TTFT需求调整KV Cache缓存策略

---

## 7. 关键公式参考

| 公式 | 说明 |
|------|------|
| `Decode TPS = batch_size × (S_in + S_out) / (TTFT + S_out × TPOT)` | 混部 TPS |
| `EP = TP × DP` | EP与TP×DP的关系 |
| `KV Cache = 2 × batch_size × seq_len × num_kv_heads × head_dim × num_layers / (TP × CP)` | KV Cache计算 |
| `Activation = batch_size × seq_len × hidden_size × num_experts_per_tok × factor / CP` | 激活值计算 |

---

## 8. 约束条件验证

| 约束 | 配置 | 是否满足 |
|------|------|----------|
| TP ≤ 单机卡数 | TP=8, 单机8卡 | ✓ |
| TP × DP = EP | 8×1=8, EP=8 | ✓ |
| EP ≤ MoE专家数 | EP=8 ≤ 256 | ✓ |
| 总卡数为单机整数倍 | 8 % 8 = 0 | ✓ |
| TP/EP/DP为2的幂 | 8=2³ | ✓ |
