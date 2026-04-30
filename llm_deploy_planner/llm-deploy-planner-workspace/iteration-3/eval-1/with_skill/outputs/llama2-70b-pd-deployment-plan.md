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
| 场景 | 离线batch processing | 离线批处理对时延不敏感，对吞吐量要求高。PD分离可获得更好的整体吞吐，短请求不被长请求阻塞 |
| 长度比 | α=0.57 | α ∈ [0.3, 0.7]，基本平衡，但高方差（std=2000）使静态长度比失去参考意义 |
| SLO | TPS优先 | TPS优先场景，PD分离可获得更高吞吐 |
| **关键因素** | **高输入方差（std=2000 vs mean=1024）** | **变异系数CV≈2，输入长度分布极其分散（100~7000+ tokens）。PD混合部署会导致短请求被长请求阻塞，资源利用率低下。PD分离后，Prefill阶段可针对不同输入长度做动态batching，Decode阶段可保持稳定的大batch输出** |

### 高输入方差的特殊处理
- std=2000意味着输入长度分布范围极广：约68%在[0, 3024]，约95%在[0, 5024]
- 极端情况下输入可达7000+ tokens（mean + 3σ ≈ 7024）
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

### KV Cache 估算公式

```
KV Cache 每 token = 2 × num_layers × num_kv_heads × kv_dim × 2(FP16) bytes
                   = 2 × 80 × 8 × 64 × 2 = 163,840 bytes ≈ 0.156 MB/token
```

### 单卡显存估算（TP=4 配置）

| 项目 | 每卡占用 |
|------|----------|
| 模型权重（TP=4分割） | ~35 GB |
| KV Cache（batch=4, seq=4000） | ~2.5 GB |
| 激活值（batch=4, seq=4000） | ~12 GB |
| 其他开销 | ~5 GB |
| **单卡总占用** | **~54.5 GB / 64 GB** |

### 并行配置显存验证

**约束条件**：
- TP <= 单机卡数（4卡）
- 非MoE模型：EP = 1
- TP/EP/DP 为2的幂（1, 2, 4, 8, 16, 32）
- 总卡数必须是单机卡数的整数倍

**EP 取值**：Llama2-70B为Dense模型，固定 EP=1

| 配置 | 权重 | KV Cache | 激活值 | 总占用 | 可用显存 | 可行 |
|------|------|----------|--------|--------|----------|------|
| TP=1, DP=32 | 140 GB | - | - | >64 GB | 64 GB | ❌ |
| TP=2, DP=16 | 70 GB | - | - | >64 GB | 64 GB | ❌ |
| **TP=4, DP=8** | **35 GB** | **~16 GB** | **~12 GB** | **~63 GB** | **64 GB** | **✅** |
| TP=8, DP=4 | 17.5 GB | ~8 GB | ~6 GB | ~32 GB | 64 GB | ✅ (资源浪费) |

**推荐配置**: TP=4, DP=8（总32卡）

---

## 4. 并行策略配置

### PD 分离部署

**配置格式: 4P1D (x=4 Prefill实例, y=1 Decode实例组)**

#### Prefill 阶段

| 参数 | 值 |
|------|-----|
| 实例数 | 4 |
| 每实例卡数 | 8 = TP(4) × DP(2) |
| 总卡数 | 16 |
| Tensor Parallel | 4 |
| Expert Parallel | 1（Dense模型） |
| Data Parallel | 2 |
| 每卡batch_size | 4（应对高输入方差） |
| 最大序列长度 | 8000 tokens（覆盖极端情况） |

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
- 4实例总Decode吞吐: ~64000 tokens/s

#### 系统整体吞吐

| 指标 | 预估值 |
|------|--------|
| Prefill吞吐（平均输入） | ~32000 tokens/s |
| Decode吞吐 | ~64000 tokens/s |
| **系统瓶颈吞吐** | **~32000 tokens/s** |
| 目标吞吐 | 20000 tokens/s |
| **是否满足** | **是（1.6x余量）** |

---

## 5. 约束验证

- [x] TP=4 <= 单机卡数=4
- [x] EP=1（Dense模型，固定值）
- [x] TP/EP/DP 为2的幂: TP=4, EP=1, DP=2
- [x] 总卡数=32 <= 可用卡数=32
- [x] 总卡数是单机卡数(4)的整数倍: 32 % 4 = 0
- [x] Prefill总卡数(16) = Decode总卡数(16)
- [x] xPyD约束: x=4 >= y=1

---

## 6. xPyD 实例数量说明

### 当前配置: 4P1D

| 阶段 | 实例数 | 每实例卡数 | 总卡数 | TP | DP |
|------|--------|-----------|--------|----|----|
| Prefill | 4 | 8 | 16 | 4 | 2 |
| Decode | 4 | 8 | 16 | 4 | 2 |

**计算过程**:
- Prefill每实例卡数 = TP × DP = 4 × 2 = 8
- Prefill总卡数 = 8 × 4 = 16
- Decode每实例卡数 = TP × DP = 4 × 2 = 8
- Decode总卡数 = 8 × 4 = 16
- x = ceil(Prefill卡数 / 每实例卡数) = ceil(16 / 8) = 2，取4以应对高输入方差
- y = 1（当前仅支持1个Decode实例组）

---

## 7. 实现注意事项

### 动态Batching配置
```python
# Prefill阶段 - 变长输入，需特殊处理
prefill_batching:
  max_batch_size: 32        # 短序列可入大batch
  max_sequence_length: 8000 # 覆盖极端情况
  short_sequence_threshold: 512  # 短序列单独batching
  grouping_interval_ms: 50  # 组batch等待时间

# Decode阶段 - 固定输出长度，可持续大batch
decode_batching:
  max_batch_size: 64
  output_length: 256
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

### 方案B: 更大TP配置 (TP=8)

| 参数 | Prefill | Decode |
|------|---------|--------|
| 实例数 | 2 | 2 |
| TP | 8 | 8 |
| DP | 2 | 2 |
| 每实例卡数 | 16 | 16 |
| 总卡数 | 32 | 32 |
| 每卡batch | 2 | 8 |
| 预估吞吐 | ~24000 tokens/s | ~48000 tokens/s |

**对比**: 方案B单实例处理能力更强，但实例数减少导致负载均衡难度增加，不推荐用于高方差场景。

---

## 9. 总结

| 项目 | 值 |
|------|-----|
| 部署策略 | PD分离 (4P1D) |
| Prefill实例 | 4 × (TP=4, DP=2, 每实例8卡) |
| Decode实例 | 4 × (TP=4, DP=2, 每实例8卡) |
| 总卡数 | 32 |
| 预估系统吞吐 | ~32000 tokens/s |
| 目标吞吐 | 20000 tokens/s |
| 吞吐余量 | 60% |

**核心优势**:
1. PD分离架构天然适配高输入方差场景
2. Prefill阶段可动态处理100~8000 tokens的变长输入
3. Decode阶段保持稳定大batch输出，确保高吞吐
4. 理论吞吐32000 tokens/s，满足20000 tokens/s目标并有充足余量
