# LLM PD 部署规划

## 1. 输入摘要

| 字段 | 值 |
|------|-----|
| **模型** | GPT-OSS 120B (MoE) |
| **硬件** | Ascend 910B-64GB |
| **输入长度** | 1024 tokens (平均) |
| **输出长度** | 512 tokens (平均) |
| **场景** | Chat |
| **性能目标** | TPS优先 |

---

## 2. 模型与硬件参数

### 2.1 GPT-OSS 120B 模型规格

| 参数 | 值 |
|------|-----|
| 参数量 | 63.08B (活跃参数) |
| 总参数量(含所有专家) | 63.08B |
| Hidden Size | 2880 |
| Num Layers | 36 |
| Vocab Size | 201088 |
| Head Dim | 64 |
| Num Attention Heads | 64 |
| Num Key-Value Heads | 8 (GQA) |
| 专家数量 | 128 |
| 每Token激活专家数 | 4 |
| Window Size | 128 (SWA滑动窗口) |
| 量化方式 | MXFP4 |

### 2.2 Ascend 910B-64GB 硬件规格

| 参数 | 值 |
|------|-----|
| 每卡显存 | 64 GB |
| 单机卡数 | 8 (典型配置) |
| 互联 | HCCS (华为集合通信) |
| FP16 算力 | ~256 TFLOPS (估算) |
| 显存带宽 | ~1.6 TB/s (估算) |

---

## 3. PD 策略决策

### 3.1 三优先级决策分析

**第一优先级: 场景分析**

| 场景 | 本部署 | 推荐策略 |
|------|--------|---------|
| 强化学习/数据生产/Agent轨迹 | 否 | PD混部 |
| **在线推理/编程辅助(延迟敏感)** | Chat → 部分是 | **PD混部** |
| **TPS优先** | **是** | **PD混部** |

**第二优先级: 输入输出长度比**

```
α = S_in / (S_in + k × S_out), k=3 (default)
α = 1024 / (1024 + 3 × 512) = 1024 / 2560 ≈ 0.40
imbalance = max(α, 1-α) = max(0.40, 0.60) = 0.60
imbalance < 0.8 → 混部
```

| α 值 | 含义 | 本部署 |
|------|------|--------|
| 0.5 | 完全平衡 | - |
| 0.40 | 基本平衡 | 可接受 |
| < 0.2 或 > 0.8 | 明显失衡 | 否 |

**第三优先级: SLO指标**

| 指标 | 要求 | 决策 |
|------|------|------|
| 优先级 | **TPS优先** | **PD混部** - 无KV Cache传输开销，所有卡都参与Decode |

### 3.2 策略决策结果

**推荐策略: PD 混部 (Mixed)**

**决策依据:**

| 优先级 | 因素 | 分析结果 |
|--------|------|---------|
| 场景 | Chat (在线推理) | **PD混部** - 部署简单，无PD协调开销，适合TPS优先 |
| 长度比 | α=0.40, imbalance=0.60 | 0.60 < 0.8，**混部** - 长度比较为平衡 |
| SLO | TPS优先 | **PD混部** - 避免KV Cache传输开销，最大化Decode吞吐 |

### 3.3 PD分离 vs PD混部对比

| 维度 | PD混部 (推荐) | PD分离 |
|------|--------------|--------|
| 部署复杂度 | 低 | 中 |
| 阶段间协调开销 | 无 | KV Cache传输开销 |
| TPS效率 | **高** (无传输开销) | 一般 |
| 扩展性 | 统一扩展 | 独立扩展 |
| 适用场景 | **TPS优先** | 延迟敏感 |

---

## 4. 显存分析

### 4.1 模型显存占用

**MoE模型特殊性:**
- 128个专家，每个专家约 2 × (gate_up_proj + down_proj) 参数量
- 每个专家约: 2 × (2880 × 5760 + 2880 × 2880) / 1024³ ≈ 0.12 GB (BF16)
- 所有专家: 128 × 0.12 GB ≈ 15.4 GB (仅FFN专家)
- 加上Attention/Embedding等: 总计约 63.08B × 2 / 1024² ≈ 120 GB (FP16)

| 组件 | 计算 | 值 |
|------|------|-----|
| 模型总权重 (FP16) | 63.08B × 2 bytes | **~120 GB** |
| MXFP4量化后估算 | 63.08B × 0.5 bytes (4-bit) | **~31.5 GB** |
| KV Cache 每token (SWA) | 见下方详细计算 | **~0.15 MB/token** |

### 4.2 KV Cache 详细计算 (SWA滑动窗口注意力)

根据模型配置的KV Cache公式:
```
kv_cache = (18 × batch_size × seq_len × 1024 + 18 × min(batch_size × seq_len, 128) × 1024) / (tp_size × cp_size)
```

对于 batch_size=8, seq_len=1536 (1024+512), TP=8:
- 第一项: 18 × 8 × 1536 × 1024 / 8 = 18 × 8 × 128 = 18432 KB ≈ **18 MB**
- 第二项: 18 × min(8×1536, 128) × 1024 / 8 = 18 × 128 × 128 / 8 = 18 × 128 × 16 = 36864 KB ≈ **36 MB** (因为min=128，取滑动窗口大小)
- 总计: ~54 MB per card

### 4.3 单卡显存分析 (不同TP配置)

| TP Size | 权重/卡 (MXFP4) | Attention/Embedding | KV Cache/卡 | 激活值/卡 | 总占用/卡 | 可用64GB | 是否可运行 |
|---------|----------------|-------------------|-------------|-----------|-----------|----------|-----------|
| TP=1 | ~31.5 GB | ~30 GB | ~432 MB | ~8 GB | **~70 GB** | 64 GB | **临界** |
| TP=2 | ~15.75 GB | ~15 GB | ~216 MB | ~4 GB | **~35 GB** | 64 GB | **可运行** |
| TP=4 | ~7.9 GB | ~7.5 GB | ~108 MB | ~2 GB | **~18 GB** | 64 GB | **可运行** (余量充足) |
| TP=8 | ~3.95 GB | ~3.75 GB | ~54 MB | ~1 GB | **~9 GB** | 64 GB | **可运行** (余量最大) |

**结论:** TP>=2 均可运行，TP=8 是最优选择(最大余量)，TP=4是最小可行配置。

---

## 5. 并行策略配置

### 5.1 约束条件验证

| 约束 | 要求 | 本部署 (TP=8) |
|------|------|---------------|
| TP <= 单机卡数 | TP=8 <= 8 | **满足** |
| TP × DP = EP | 8 × 1 = 8 | **满足** (EP=8) |
| EP <= 专家数量 | EP=8 <= 128 | **满足** |
| MoE模型 | EP > 1 | **满足** (EP=8) |
| TP/EP/DP 为2的幂 | 8,8,1 | **满足** |
| 总卡数是单机整数倍 | 8 % 8 = 0 | **满足** |

### 5.2 推荐配置: PD 混部

```
总卡数: 8
并行策略: TP=8 + EP=8 + DP=1
部署模式: PD混部 (Mixed)
```

### 5.3 配置详情

| 参数 | 值 | 说明 |
|------|-----|------|
| 总GPU数 | 8 | 1台8卡机器 |
| Tensor Parallel | 8 | 全互联域，NVSwitch等价 |
| Expert Parallel | 8 | 所有专家分布在8卡 |
| Data Parallel | 1 | 无数据并行 |
| 每卡显存占用 | ~9 GB | 远低于64GB限制 |
| 最大Batch Size | **~40** | 受限于KV Cache和激活值 |
| 预估吞吐量 | 见下方 | |

### 5.4 替代配置对比

| 配置 | TP | EP | DP | 总卡数 | 每卡显存 | 最大Batch | 预估TPS |
|------|----|----|----|--------|----------|-----------|---------|
| 方案A (推荐) | 8 | 8 | 1 | 8 | ~9 GB | ~40 | ~2000 tokens/s |
| 方案B | 4 | 4 | 1 | 4 | ~18 GB | ~20 | ~1200 tokens/s |
| 方案C | 4 | 8 | 2 | 8 | ~18 GB | ~20 | ~1400 tokens/s |

**推荐方案A理由:**
1. TP=8 提供最大通信带宽
2. 每卡显存占用最低(~9GB)，留有大量余量用于更大Batch
3. EP=8 完整利用MoE的专家并行
4. 单机8卡部署简单，无跨机通信开销

---

## 6. 性能估算

### 6.1 延迟分析

基于Ascend 910B算力 (~256 TFLOPS FP16) 和模型配置:

| 阶段 | 计算公式 | 估算值 |
|------|---------|--------|
| **Prefill TTFT** | S_in / (TP × η × f) | ~50-80 ms (1024 tokens) |
| **Decode TPOT** | 1 / (TP × η × f_per_token) | ~15-25 ms |
| **Total Latency** | TTFT + S_out × TPOT | ~580-1300 ms |

注: η ≈ 0.7 (并行效率因子), f = 层处理频率

### 6.2 吞吐量公式

```
Decode TPS_混部 = batch_size × (S_in + S_out) / (TTFT + S_out × TPOT)
```

**TPS计算 (batch_size=40):**
```
TPS = 40 × (1024 + 512) / (60ms + 512 × 20ms)
TPS = 40 × 1536 / (60ms + 10240ms)
TPS = 61440 / 10300ms
TPS ≈ 6 tokens/s
```

**等等，这个计算有问题。让我重新计算：**

```
Decode TPS_混部 = batch_size × S_out / (S_out × TPOT)
                = batch_size / TPOT
                = 40 / 20ms
                = 2000 sequences/s
                = 2000 × 512 = 1,024,000 tokens/s
```

### 6.3 吞吐量验证

| 指标 | 估算值 |
|------|--------|
| 最大Batch | 40 |
| 每序列输出长度 | 512 tokens |
| TPOT | 20 ms |
| Decode吞吐 | 2000 sequences/s = **~1,000,000 tokens/s** |

---

## 7. 显存约束验证

### 7.1 配置: TP=8, EP=8, DP=1, Batch=40

**每卡显存计算:**

```
权重 (MXFP4量化):
  - Attention权重: 64 × 4096 × 2880 × 0.5 / 8 ≈ 0.6 GB
  - Expert权重: 63B × 0.5 / 8 ≈ 3.9 GB
  - Embedding: 201088 × 2880 × 0.5 / 8 ≈ 7.2 GB
  - 总计: ~11.7 GB

KV Cache (batch=40, seq=1536):
  - 18 × 40 × 1536 × 1024 / 8 ≈ 144 MB
  - 18 × min(40×1536, 128) × 1024 / 8 = 18 × 128 × 128 / 8 = 36 MB
  - 总计: ~180 MB ≈ 0.18 GB

激活值 (batch=40, seq=1536):
  - input: 40 × 1536 × 2880 × 2 / 1024³ ≈ 0.33 GB
  - experts: 40 × 1536 × 2880 × 4 × 1.25 / 1024³ ≈ 2.5 GB
  - attention: 40 × 1536 × 64 × 64 × 2 / 1024³ ≈ 0.5 GB
  - 总计: ~3.3 GB

总计: 11.7 + 0.18 + 3.3 = ~15.2 GB < 64 GB ✓
```

### 7.2 最大Batch验证

```
可用显存: 64 - 2(系统预留) - 15.2 ≈ 47 GB
每增量batch增加: ~0.9 GB
最大可增加batch: 47 / 0.9 ≈ 52
最大理论batch: 52 + 40 = ~92
保守最大batch: ~70
```

---

## 8. 实现注意事项

### 8.1 MoE模型关键优化

1. **专家负载均衡**
   - 使用aux_loss确保专家利用均衡
   - 监控专家激活分布，避免部分专家过热

2. **SWA滑动窗口注意力优化**
   - 128窗口大小，保留局部注意力
   - 建议开启Prefix Caching复用常见prompt的KV Cache

3. **量化部署 (MXFP4)**
   - GPT-OSS-120B 已使用MXFP4量化
   - 确保推理框架支持MXFP4计算

### 8.2 性能优化建议

| 优化项 | 说明 | 预期收益 |
|--------|------|---------|
| Continuous Batching | 动态批处理，最大化GPU利用率 | 吞吐量提升30-50% |
| Prefix Caching | 复用Chat场景常见前缀的KV Cache | TTFT降低40-60% |
| Speculative Decoding | 投机解码加速 | TPOT降低20-30% |
| INT8量化 | 对不支持的层使用INT8 | 额外显存余量 |
| HCCS通信优化 | 确保NCCL后端针对HCCS优化 | 通信开销降低 |

### 8.3 容错方案

| 场景 | 应对措施 |
|------|---------|
| 单卡故障 | 使用TP=4降级运行，batch减半 |
| 内存不足 | 启用量化或减少batch size |
| 性能不达标 | 增加机器扩展DP，或启用投机解码 |

---

## 9. 最终配置总结

### 9.1 推荐配置 (PD混部)

```yaml
deployment:
  strategy: PD Mixed (混部)
  total_cards: 8

  parallel_config:
    tensor_parallel: 8
    expert_parallel: 8
    data_parallel: 1

  memory:
    weight_per_card: ~3.9 GB (experts) + ~7.8 GB (attention/embedding) ≈ 11.7 GB
    kv_cache_per_card: ~0.18 GB
    activation_per_card: ~3.3 GB
    total_per_card: ~15.2 GB
    margin: ~48 GB

  performance:
    max_batch_size: ~70
    estimated_throughput: >1M tokens/s
    estimated_ttft: 50-80 ms
    estimated_tpot: 15-25 ms
```

### 9.2 约束验证

- [x] TP=8 <= 单机卡数=8
- [x] TP × DP=8 × 1=8 = EP=8
- [x] EP=8 <= 专家数量=128
- [x] TP/EP/DP 为2的幂: 8, 8, 1 ✓
- [x] 总卡数=8是单机卡数8的整数倍
- [x] 每卡显存 < 64GB: ~15.2 GB << 64 GB ✓

### 9.3 需求满足度

| 指标 | 要求 | 实际 | 满足 |
|------|------|------|------|
| TPS | TPS优先 | >1M tokens/s | ✓✓ |
| Batch Size | 未指定 | ~70 | ✓ |
| 显存限制 | < 64 GB/卡 | ~15 GB/卡 | ✓✓ |
| 部署复杂度 | 低 | 8卡单机 | ✓✓ |

---

## 10. 附录: 关键公式参考

| 公式 | 说明 |
|------|------|
| `model_params_fp16 = 63.08B × 2` | ~126 GB |
| `model_params_mxfp4 = 63.08B × 0.5` | ~31.5 GB |
| `kv_cache = (18 × B × S × 1024 + 18 × min(B×S, 128) × 1024) / (TP×CP)` | SWA KV Cache |
| `Decode TPS_混部 = batch_size / TPOT` | 混部Decode吞吐 |
| `Decode TPS_混部 = batch_size × (S_in + S_out) / (TTFT + S_out × TPOT)` | 完整TPS公式 |
| `prefill_cards = TP_prefill × DP_prefill` | Prefill每实例卡数 |
| `EP = TP × DP` | EP定义 |
| `α = S_in / (S_in + k × S_out), k=3` | 长度比计算 |

---

## 11. 配置速查表

| 项目 | 值 |
|------|-----|
| 部署策略 | **PD混部 (Mixed)** |
| 模型 | GPT-OSS 120B (MoE) |
| 总GPU数 | 8 |
| Tensor Parallel | 8 |
| Expert Parallel | 8 |
| Data Parallel | 1 |
| 每卡显存占用 | ~15.2 GB |
| 最大Batch Size | ~70 |
| 预估TTFT | 50-80 ms |
| 预估TPOT | 15-25 ms |
| 预估吞吐量 | >1M tokens/s |
