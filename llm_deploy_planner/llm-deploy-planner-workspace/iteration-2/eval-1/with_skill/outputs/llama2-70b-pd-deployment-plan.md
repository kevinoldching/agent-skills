# LLM PD 部署规划

## 1. 输入摘要

| 字段 | 值 |
|------|-----|
| 模型 | Llama2-70B |
| 硬件 | Ascend-910B-64GB × 4卡/单机 × 8机 = 32卡 |
| 输入长度 | 平均1024 tokens，std 2000（高方差） |
| 输出长度 | 平均256 tokens |
| 场景 | 离线batch processing |
| 性能目标 | TPS优先，吞吐量 >= 20000 tokens/s |

---

## 2. PD 策略决策

### 决策结果
**推荐策略: PD分离部署**

### 决策依据

| 优先级 | 因素 | 分析 |
|--------|------|------|
| 场景 | 离线batch processing | 离线批处理对时延不敏感，但对吞吐量要求高，适合PD分离以获得更好的整体吞吐 |
| 长度比 | α=0.57 | α ∈ [0.3, 0.7]，基本平衡，但高方差（std=2000）使得静态长度比失去参考意义 |
| SLO | TPS优先 | TPS优先场景，PD分离可获得更高吞吐 |
| **关键因素** | **高输入方差（std=2000 vs mean=1024）** | **变异系数CV≈2，输入长度分布极其分散（100~7000+ tokens）。PD混合部署会导致短请求被长请求阻塞，资源利用率低下。PD分离后，Prefill阶段可针对不同输入长度做动态 batching，Decode阶段可保持稳定的大batch输出** |

### 高输入方差的特殊处理
- std=2000意味着输入长度分布范围极广：约68%在[0, 3024]，约95%在[0, 5024]
- 极端情况下输入可达7000+ tokens
- PD分离优势：
  1. Prefill实例专门处理变长输入，避免长序列阻塞
  2. Decode实例处理固定长度输出，可持续大batch
  3. 独立的扩缩容能力，适配输入/输出负载差异

---

## 3. 显存分析

### 模型显存占用

| 项目 | 规格 |
|------|------|
| 模型 | Llama2-70B |
| 权重（FP16） | ~140 GB |
| 参数量 | 70B |
| 层数 | 80 |
| 隐藏维度 | 4096 |
| Attention Heads | 64 |
| KV Heads | 8 (with GQA) |

### 单卡显存估算（TP=4 配置）

| 项目 | 每卡占用 |
|------|----------|
| 模型权重（TP=4分割） | ~35 GB |
| KV Cache（batch=2, seq=4000） | ~4 GB |
| 激活值（batch=2, seq=4000） | ~8 GB |
| 其他开销 | ~5 GB |
| **单卡总占用** | **~52 GB / 64 GB** |

### 并行配置显存验证

**约束条件**：
- TP <= 单机卡数（4卡）或 TP = 单机卡数 × 机器数量
- TP × DP <= EP（非MoE模型 EP=1，约束不适用）
- EP < P/D实例总卡数
- TP/EP/DP 为2的幂（1, 2, 4, 8, 16, 32）

**EP 取值**：Llama2-70B为Dense模型，固定 EP=1

| 配置 | 权重 | KV Cache | 激活值 | 总占用 | 可用显存 | 可行 |
|------|------|----------|--------|--------|----------|------|
| TP=1, DP=32 | 140 GB | ❌ | - | - | 64 GB | ❌ |
| TP=2, DP=16 | 70 GB | ❌ | - | - | 64 GB | ❌ |
| **TP=4, DP=8** | **35 GB** | **~16 GB** | **~10 GB** | **~61 GB** | **64 GB** | **✅** |
| TP=8, DP=4 | 17.5 GB | ~8 GB | ~5 GB | ~31 GB | 64 GB | ✅ (资源浪费) |

**推荐配置**: TP=4, DP=8（总32卡）

---

## 4. 并行策略配置

### PD 分离部署

**配置格式: 4P4D (x=4, y=1)**

#### Prefill 阶段

| 参数 | 值 |
|------|-----|
| 实例数 | 4 |
| 每实例卡数 | 8 = TP(4) × DP(2) |
| 总卡数 | 16 |
| Tensor Parallel | 4 |
| Expert Parallel | 1（Dense模型） |
| Data Parallel | 2 |
| 每卡batch_size | 2（应对高输入方差，避免长序列OOM） |
| 最大序列长度 | 4000 tokens（3σ覆盖） |

**Prefill 预估性能**:
- 平均输入长度1024 tokens时: TTFT ≈ 800 ms
- 长尾输入（4000 tokens）时: TTFT ≈ 2000 ms
- Prefill吞吐量: ~32000 tokens/s（平均输入）

#### Decode 阶段

| 参数 | 值 |
|------|-----|
| 实例数 | 4 |
| 每实例卡数 | 8 = TP(4) × DP(2) |
| 总卡数 | 16 |
| Tensor Parallel | 4 |
| Expert Parallel | 1（Dense模型） |
| Data Parallel | 2 |
| 每卡batch_size | 16（Decode阶段序列短，可加大batch） |
| 输出长度 | 256 tokens |

**Decode 预估性能**:
- TPOT ≈ 4 ms per token
- 单实例Decode吞吐: ~16000 tokens/s
- 4实例总Decode吞吐: **~64000 tokens/s**

#### 系统整体吞吐

| 指标 | 预估值 |
|------|--------|
| Prefill吞吐（平均输入） | ~32000 tokens/s |
| Decode吞吐 | ~64000 tokens/s |
| **系统瓶颈吞吐** | **~32000 tokens/s** |
| 目标吞吐 | 20000 tokens/s |
| **是否满足** | **✅ 是（1.6x余量）** |

---

## 5. 约束验证

- [x] TP=4 <= 单机卡数=4
- [x] TP×DP=8 <= EP=1（N/A for Dense模型）
- [x] EP=1 < 实例总卡数=32
- [x] TP/EP/DP 为2的幂: TP=4, EP=1, DP=2
- [x] 总卡数=32 <= 可用卡数=32
- [x] Prefill总卡数(16) = Decode总卡数(16)

---

## 6. xPyD 实例数量说明

### 当前配置: 4P1D

| 阶段 | 实例数 | 每实例卡数 | 总卡数 | TP | DP |
|------|--------|-----------|--------|----|----|
| Prefill | 4 | 8 | 16 | 4 | 2 |
| Decode | 4 | 8 | 16 | 4 | 2 |

**计算过程**:
- Prefill总卡数 = TP_prefill × DP_prefill × 实例数 = 4 × 2 × 4 = 16
- Decode总卡数 = TP_decode × DP_decode × 实例数 = 4 × 2 × 4 = 16
- x = ceil(Prefill卡数 / 每实例卡数) = ceil(16 / 8) = 2，但实际部署4个实例以平衡负载
- y = 1（当前仅支持1个Decode实例组）

**注**: 由于高输入方差（std=2000），Prefill实例配置为4个而非2个，以支持动态batching和长序列的并发处理。

---

## 7. 实现注意事项

### 动态Batching配置
```python
# Prefill阶段 - 变长输入，需特殊处理
prefill_batching:
  max_batch_size: 32        # 短序列可入大batch
  max_sequence_length: 4000 # 3σ覆盖
  short_sequence_threshold: 512  # 短序列单独batching
  grouping_interval_ms: 50  # 组batch等待时间
```

### KV Cache传输优化
- Prefill → Decode 通过RDMA传输KV cache
- 传输数据量: 2 × seq_len × kv_heads × kv_dim × batch_size × fp16
- 对于seq=1024, batch=32: ~256 MB per transfer
- 建议使用Pipeline并行掩盖传输延迟

### 容错与负载均衡
1. Prefill实例间使用Round-robin分发长序列请求
2. Decode实例间基于当前batch负载动态分发
3. 配置请求超时重试机制（建议超时时间: 10s）

### 资源配置建议
| 资源 | Prefill实例 | Decode实例 |
|------|-------------|------------|
| GPU | 16卡 (4×4) | 16卡 (4×4) |
| Memory | 64GB × 4 | 64GB × 4 |
| 网络 | 100GbE RDMA | 100GbE RDMA |

---

## 8. 备选方案

### 方案B: 更大的Prefill Batch (适用于输入方差较小场景)

| 参数 | Prefill | Decode |
|------|---------|--------|
| 实例数 | 2 | 4 |
| TP | 4 | 4 |
| DP | 4 | 2 |
| 每实例卡数 | 16 | 8 |
| 总卡数 | 32 | 32 |
| 每卡batch | 4 | 16 |
| 预估吞吐 | ~48000 tokens/s | ~64000 tokens/s |

**对比**: 方案B吞吐量更高，但Prefill单实例处理长序列时可能成为瓶颈，不推荐用于std=2000的高方差场景。

---

## 9. 总结

| 项目 | 值 |
|------|-----|
| 部署策略 | PD分离 (4P4D) |
| Prefill实例 | 4 × (TP=4, DP=2) |
| Decode实例 | 4 × (TP=4, DP=2) |
| 总卡数 | 32 |
| 预估系统吞吐 | ~32000 tokens/s |
| 目标吞吐 | 20000 tokens/s |
| 吞吐余量 | 60% |

**核心优势**:
1. PD分离架构天然适配高输入方差场景
2. Prefill阶段可动态处理100~4000 tokens的变长输入
3. Decode阶段保持稳定大batch输出，确保高吞吐
4. 理论吞吐32000 tokens/s，满足20000 tokens/s目标并有充足余量
