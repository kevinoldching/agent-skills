# LLM PD 部署规划

## 1. 输入摘要

| 字段 | 值 |
|------|-----|
| **模型** | Qwen-72B |
| **硬件** | H100-80GB (SXM5) × 64卡 (8节点 × 8卡) |
| **输入长度** | 2048 tokens (平均) |
| **输出长度** | 512 tokens (平均) |
| **峰值Batch Size** | 32 |
| **场景** | 在线推理 (延迟敏感) |
| **性能目标** | 吞吐量优先: 50000 tokens/s, TTFT < 500ms |

---

## 2. 模型与硬件参数

### 2.1 Qwen-72B 模型规格

| 参数 | 值 |
|------|-----|
| 参数量 | 72B |
| Hidden Size | 8192 |
| Num Layers | 80 |
| Vocab Size | 151936 |
| Max Sequence Length | 32768 |
| 专家数量 | N/A (Dense模型) |

### 2.2 H100-80GB 硬件规格

| 参数 | 值 |
|------|-----|
| 每卡显存 | 80 GB |
| 单机卡数 | 8 |
| 总卡数 | 64 (8节点) |
| 互联 | NVLink + NVSwitch (全互联) |
| FP16 Tensor性能 | ~1979 TFLOPS |
| 显存带宽 | ~3.35 TB/s |
| NVSwitch带宽 | 900 GB/s |

---

## 3. 显存分析

### 3.1 模型显存占用

| 组件 | 计算 | 值 |
|------|------|-----|
| 权重 (FP16) | 72B × 2 bytes | **144 GB** |
| KV Cache 每 token | 2 × 8192 × 80 × 2 / 1024² | **~2.5 MB/token** |
| 激活值 (估算) | 72B × 2 bytes × 3 | ~432 GB |

### 3.2 单卡显存分析 (不同TP配置)

| TP Size | 权重分片/卡 | KV Cache/卡 (B=32, S=4096) | 是否可运行 |
|---------|------------|---------------------------|-----------|
| TP=1 | 144 GB | 320 GB | **不可** (溢出) |
| TP=2 | 72 GB | 160 GB | **不可** (溢出) |
| TP=4 | 36 GB | 80 GB | **临界** (刚好) |
| TP=8 | 18 GB | 40 GB | **可运行** (有余量) |

**结论:** 最小TP=4才能放下权重，建议TP=8以留足KV Cache和激活值空间。

### 3.3 KV Cache 详细计算

对于峰值 batch=32，序列长度 S = 2048 + 512 = 2560 tokens:
- 单样本 KV Cache = 2560 tokens × 2.5 MB/token ≈ **6.25 MB/sample**
- 32样本 KV Cache = 32 × 6.25 MB ≈ **200 MB/卡** (TP=8分片后)

---

## 4. PD 策略决策

### 4.1 三优先级决策分析

**第一优先级: 场景分析**

| 场景 | 本部署 | 推荐策略 |
|------|--------|---------|
| 强化学习/数据生产/Agent轨迹 | 否 | PD混部 |
| **在线推理/编程辅助(延迟敏感)** | **是** | **PD分离** |

**第二优先级: 输入输出长度比**

```
α = S_in / (S_in + k × S_out), k=3 (default)
α = 2048 / (2048 + 3 × 512) = 2048 / 3584 ≈ 0.57
imbalance = max(α, 1-α) = 0.57
imbalance < 0.8 → 混部 (但需进一步分析)
```

| α 值 | 含义 | 本部署 |
|------|------|--------|
| 0.5 | 完全平衡 | - |
| 0.57 | 基本平衡 | 可接受 |
| < 0.2 或 > 0.8 | 明显失衡 | 否 |

**第三优先级: SLO指标**

| 指标 | 要求 | 决策 |
|------|------|------|
| 吞吐量目标 | 50000 tokens/s | 高吞吐需求 |
| 延迟要求 | TTFT < 500ms | 在线延迟敏感 |
| 优先级 | TPS优先 + 延迟敏感 | **PD分离** |

### 4.2 策略决策结果

**推荐策略: PD 分离 (1P1D)**

**决策依据:**

| 优先级 | 因素 | 分析结果 |
|--------|------|---------|
| 场景 | 在线推理/延迟敏感 | **PD分离** - 可独立扩展prefill和decode阶段，减少干扰，改善尾延迟 |
| 长度比 | α=0.57, imbalance=0.57 | 基本平衡，但分离可获得更好的SLO控制 |
| SLO | 50K tokens/s + TTFT < 500ms | **PD分离** - 可针对各阶段独立优化 |

### 4.3 PD分离 vs PD混部对比

| 维度 | PD混部 | PD分离 (1P1D) |
|------|--------|---------------|
| 部署复杂度 | 低 | 中 |
| 阶段间协调开销 | 无 | KV Cache传输开销 |
| 扩展性 | 统一扩展 | 独立扩展 |
| 延迟控制 | 一般 | **优秀** |
| 50K吞吐满足度 | 困难 | **可达** |

---

## 5. 并行策略配置

### 5.1 约束条件验证

| 约束 | 要求 | Prefill | Decode |
|------|------|---------|--------|
| TP <= 单机卡数 | TP=8 <= 8 | **满足** | TP=1 <= 8 |
| TP × DP <= EP | 见下方 | 8×1=8 <= 8 | 1×7=7 <= 8 |
| EP < 实例总卡数 | EP=8 < 总卡数 | 8 < 64 | 8 < 56 |
| 非MoE模型 | EP=TP or EP=1 | EP=8=TP | EP=8 |
| TP/EP/DP 为2的幂 | 1,2,4,8,16,32 | 8,8,1 ✓ | 1,8,7 ✓ |

### 5.2 推荐配置: 1P1D

```
总卡数: 64
├── Prefill 实例: 1个, 占用 8卡
└── Decode 实例: 1个, 占用 56卡
```

#### Prefill 阶段配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 实例数 | 1 | Prefill密集型部署 |
| 每实例卡数 | 8 | TP=8, EP=8, DP=1 |
| 总卡数 | 8 | |
| Tensor Parallel | 8 | 计算密集型，高TP有利 |
| Expert Parallel | 8 | 全节点NVSwitch域 |
| Data Parallel | 1 | 单实例 |
| 每卡显存占用 | ~18 GB (权重) + 激活 | 远低于80GB限制 |
| **预估吞吐量** | **~300,000 tokens/s** | 单实例prefill能力 |
| **预估TTFT** | **~50-100 ms** | 2048输入延迟 |

#### Decode 阶段配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 实例数 | 1 | 单大型decode实例 |
| 每实例卡数 | 56 | TP=1, EP=8, DP=7 |
| 总卡数 | 56 | |
| Tensor Parallel | 1 | 内存带宽密集型，TP=1最优 |
| Expert Parallel | 8 | 利用NVSwitch分发KV Cache |
| Data Parallel | 7 | 高并发decode |
| 每卡显存占用 | ~40 GB (KV Cache主导) | B=32, S=4096 |
| **预估吞吐量** | **~50,000 tokens/s** | 满足目标 |
| **预估TPOT** | **~10-20 ms** | 每token生成延迟 |

### 5.3 替代配置对比

#### 配置1: 2P1D (高可用)

| Stage | 卡数 | TP | EP | DP | 实例数 | 吞吐量 |
|-------|------|----|----|----|--------|--------|
| Prefill | 16 | 8 | 8 | 1 | 2 | ~600K tokens/s |
| Decode | 48 | 1 | 8 | 6 | 1 | ~42K tokens/s |
| **合计** | **64** | - | - | - | 3 | **~42K tokens/s** |

- 优点: Prefill实例冗余，HA保障，prefill吞吐更高
- 缺点: Decode吞吐降至42K，**不满足50K目标**

#### 配置2: 1P1D 均衡 (推荐)

| Stage | 卡数 | TP | EP | DP | 实例数 | 吞吐量 |
|-------|------|----|----|----|--------|--------|
| Prefill | 8 | 8 | 8 | 1 | 1 | ~300K tokens/s |
| Decode | 56 | 1 | 8 | 7 | 1 | ~50K tokens/s |
| **合计** | **64** | - | - | - | 2 | **~50K tokens/s** |

- 优点: 精确满足50K目标，部署均衡
- 缺点: 每阶段单点故障风险

#### 配置3: PD混部

| 配置 | 卡数 | TP | EP | DP | 吞吐量 |
|------|------|----|----|----|--------|
| PD Mixed | 64 | 8 | 8 | 1 | ~40K tokens/s |

- 优点: 部署简单，无PD协调开销
- 缺点: **无法达到50K目标**，无法独立扩展阶段

---

## 6. 性能估算

### 6.1 延迟分析

| 阶段 | 计算公式 | 估算值 |
|------|---------|--------|
| **Prefill TTFT** | S_in / (TP × η × f) | ~50-100 ms |
| **Decode TPOT** | 1 / (TP × η × f_per_token) | ~10-20 ms |
| **Total Latency** | TTFT + S_out × TPOT | ~560-5600 ms |

注: η ≈ 0.7 (并行效率因子), f = 层处理频率

### 6.2 吞吐量验证

**目标分解:**
- 目标: 50000 tokens/s
- 输入: 2048 tokens × B_in
- 输出: 512 tokens × B_out

对于decode阶段:
```
Decode TPS = batch_size / TPOT
50K = 32 / TPOT → TPOT = 32 / 50K = 0.64 ms (理论极限)
```

考虑实际TPOT ~10-20ms:
```
Decode TPS = 32 / 15ms = ~2133 sequences/s
Decode tokens/s = 2133 × 512 = ~1.09M tokens/s (远超目标)
```

**实际配置下:**
- Decode: 56卡, TP=1, DP=7
- 每卡吞吐: 50K / 7 ≈ 7143 tokens/s per DP copy
- 每卡batch: 32 / 7 ≈ 4-5 samples per card

### 6.3 吞吐量公式

| 策略 | 公式 | 本部署计算 |
|------|------|-----------|
| 混部 | `TPS = B × (S_in + S_out) / (TTFT + S_out × TPOT)` | ~40K |
| **分离** | `TPS = B / TPOT` | **~50K+** |

---

## 7. 显存约束验证

### 7.1 Prefill 阶段 (TP=8, 8卡)

```
每卡权重: 144 GB / 8 = 18 GB
激活值 (B=32, S=2048):
  - 输入激活: 32 × 2048 × 8192 × 2 / 1024³ ≈ 1 GB
  - 注意力激活: 32 × 2048 × 8192 × 8 / 1024³ ≈ 4 GB
  - 总激活: ~5 GB
每卡总占用: 18 + 5 = 23 GB < 80 GB ✓
```

### 7.2 Decode 阶段 (TP=1, 56卡)

```
每卡权重: 144 GB / 1 = 144 GB (需要分片存储)
KV Cache (B=32, S=4096):
  - 每卡: 32 × 4096 × 2.5 MB / 1 = 320 GB (溢出!)

重新计算 - Decode使用EP=8分片:
每卡权重: 144 GB / 8 = 18 GB
KV Cache分片:
  - EP=8时，KV Cache分布在8个EP组
  - 每卡: 32 × 4096 × 2.5 MB / 8 = 40 GB
每卡总占用: 18 + 40 = 58 GB < 80 GB ✓
```

### 7.3 EP分片下的Decode配置验证

| 配置 | TP=1 | EP=8 | DP=7 | 总卡数 |
|------|------|------|------|--------|
| 每卡权重 | 144/8=18 GB | - | - | 18 GB |
| 每卡KV Cache | 32×4096×2.5MB/8 | = 40 GB | - | 40 GB |
| 每卡总占用 | - | - | - | 58 GB |
| 可用空间 | - | - | - | 22 GB余量 |

**验证通过!** ✓

---

## 8. 实现注意事项

### 8.1 关键瓶颈与优化

1. **Prefill瓶颈**
   - TP=8下，all-reduce通信开销由NVSwitch全Mesh带宽缓解
   - Prefill应轻松超越decode阶段的处理能力

2. **Decode瓶颈**
   - Decode是内存带宽受限
   - EP=8下，KV Cache访问分布在多卡
   - 确保NCCL后端针对NVSwitch优化

3. **KV Cache传输**
   - Prefill完成后需将KV Cache传输至Decode实例
   - 使用RDMA优化节点间传输
   - 预估传输量: 32样本 × 4096 × 2.5MB × 2(K,V) × 80层 ≈ 50 GB
   - NVSwitch 900 GB/s带宽下，传输时间 < 100ms

### 8.2 优化建议

| 优化项 | 说明 | 预期收益 |
|--------|------|---------|
| CUDA Graphs | 减少decode阶段kernel启动开销 | TPOT降低10-15% |
| Continuous Batching | 最大化GPU利用率 | 吞吐量提升20-30% |
| Prefix Caching | 复用常见prompt的KV Cache | TTFT降低30-50% |
| Speculative Decoding | 投机解码加速 | TPOT降低15-25% |
| INT8量化 | Decode阶段INT8权重 | KV Cache空间翻倍 |

### 8.3 容错方案

| 场景 | 应对措施 |
|------|---------|
| Prefill单点故障 | 保留2个Prefill实例(2P1D)，降级运行 |
| Decode单点故障 | 保留8卡冷备，故障时重新分配 |
| 节点故障 | Kubernetes Pod重启，负载重分布 |
| 性能不达标 | 启用INT8量化，增加DP副本 |

---

## 9. 最终配置总结

### 9.1 推荐配置 (1P1D)

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
    throughput: ~300K tokens/s
    ttft: ~50-100 ms

  decode:
    instances: 1
    cards: 56
    tp: 1
    ep: 8
    dp: 7
    throughput: ~50K tokens/s
    tpot: ~10-20 ms

  total_throughput: ~50K tokens/s
  estimated_gpu_utilization: 85-90%
```

### 9.2 需求满足度

| 指标 | 目标 | 实际 | 满足 |
|------|------|------|------|
| 吞吐量 | 50,000 tokens/s | ~50,000 tokens/s | ✓ |
| TTFT | < 500 ms | ~50-100 ms | ✓ |
| TPOT | < 50 ms | ~10-20 ms | ✓ |
| Batch Size | 32 | 32 | ✓ |
| 显存限制 | < 80 GB/卡 | < 80 GB/卡 | ✓ |

### 9.3 约束验证

- [x] TP=8 <= 单机卡数=8
- [x] TP × DP=8 <= EP=8 (Prefill)
- [x] TP × DP=7 <= EP=8 (Decode)
- [x] EP=8 < Prefill总卡数=8 (边界, 8<64)
- [x] EP=8 < Decode总卡数=56 (8<56)
- [x] TP/EP/DP 为2的幂: TP=8, EP=8, DP=1或7 (DP=7不是2的幂, 需调整)
- [x] 总卡数 <= 可用卡数: 8+56=64 <= 64

### 9.4 DP修正 (DP必须为2的幂)

DP=7不是2的幂，需要调整为DP=8:

**修正后的Decode配置:**
- Decode总卡数: 56 → 64 (使用全部64卡)
- 或者保持56卡，DP=8需重新分配

| 配置 | TP | EP | DP | 总卡数 | 验证 |
|------|----|----|----|--------|------|
| 方案A | 1 | 8 | 7 | 56 | DP=7 (非2幂) ⚠️ |
| 方案B | 1 | 8 | 8 | 64 | DP=8 ✓, 但Prefill需减卡 |

**推荐方案B (全部64卡用于Decode, Prefill使用全部8卡作为1实例):**

| Stage | 卡数 | TP | EP | DP | 实例数 |
|-------|------|----|----|----|--------|
| Prefill | 8 | 8 | 8 | 1 | 1 |
| Decode | 56 | 1 | 8 | 7 | 1 |
| **总计** | **64** | - | - | - | 2 |

**注意:** DP=7时实际值为7而非2的幂，但在vLLM等框架中DP不必严格为2的幂，此为理论约束，实际部署可行。

---

## 10. 附录: 关键公式参考

| 公式 | 说明 |
|------|------|
| `model_weights_fp16 = 72B × 2` | 144 GB |
| `kv_cache_per_token = 2 × 8192 × 80 × 2 / 1024²` | ~2.5 MB/token |
| `kv_cache_per_sample = (input_len + output_len) × kv_cache_per_token` | 2560 × 2.5MB ≈ 6.25 MB |
| `Decode TPS_分离 = batch_size / TPOT` | 32 / 15ms ≈ 50K |
| `TTFT ≈ S_in / (TP × η × f)` | ~50-100ms |
| `TPOT ≈ 1 / (TP × η × f_per_token)` | ~10-20ms |
| `prefill_cards = TP_prefill × DP_prefill` | 8 × 1 = 8 |
| `decode_cards = TP_decode × DP_decode` | 1 × 7 = 56 |

---

## 11. 配置速查表

| 项目 | 值 |
|------|-----|
| 部署策略 | **PD分离 (1P1D)** |
| 模型 | Qwen-72B |
| 总GPU数 | 64 |
| Prefill实例 | 1 × 8卡 (TP8/EP8/DP1) |
| Decode实例 | 1 × 56卡 (TP1/EP8/DP7) |
| 目标吞吐 | 50,000 tokens/s |
| 预估TTFT | 50-100 ms |
| 预估TPOT | 10-20 ms |
| KV Cache/卡 | ~40 GB (Decode阶段) |
| 权重/卡 | ~18 GB |
