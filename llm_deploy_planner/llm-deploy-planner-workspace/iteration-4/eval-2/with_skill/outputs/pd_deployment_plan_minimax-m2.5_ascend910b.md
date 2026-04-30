# LLM PD 部署规划

## 1. 输入摘要

| 字段 | 值 |
|------|-----|
| 模型 | MiniMax-M2.5（MoE模型） |
| 硬件 | Ascend-910B-64GB × 8 卡 |
| 输入长度 | 1024 tokens |
| 输出长度 | 512 tokens |
| 场景 | chat |
| 性能目标 | TPS优先 |

## 2. PD 策略决策

### 决策结果
**推荐策略: PD 混部（Mixed Prefill/Decode）**

### 决策依据

| 优先级 | 因素 | 分析 |
|--------|------|------|
| 场景 | chat（在线推理） | TPS优先，混部可最大化批量处理效率 |
| 长度比 | α = 0.4 | α ∈ [0.3, 0.7]，基本平衡，imbalance = 0.6 < 0.8，支持混部 |
| SLO | TPS优先 | 混部无KV cache transfer开销，所有卡都可用于decode，吞吐量更高 |

### 长度比计算
```
α = S_in / (S_in + k × S_out)，k = 3 (default)
α = 1024 / (1024 + 3 × 512) = 1024 / 2560 = 0.4
imbalance = max(α, 1-α) = max(0.4, 0.6) = 0.6
判定: 0.6 < 0.8 → 混部策略可行
```

## 3. 显存分析

### 模型配置修复
**问题**: MiniMax-M2.5 的 MoE expert 权重在配置文件中设置为 `replicated`，导致 EP 分片无法生效
**修复**: 将 expert 权重（w1, w2, w3, gate, e_score_correction_bias）的 `parallel_strategy` 从 `replicated` 改为 `EP`

### 模型显存占用（MiniMax-M2.5 FP8）

| 项目 | 显存占用 |
|------|----------|
| 权重（FP8，EP分片后） | ~26.79 GB |
| KV Cache每token | ~0.045 GB (45 MB) |
| 激活值（batch=1, seq=1536） | ~0.09 GB |
| 系统预留 | 2 GB |

### 并行配置显存验证（8卡，TP×DP=EP=8）

| 配置 | 权重/卡 | KV Cache | Activation | 总占用 | 可用 |
|------|---------|----------|------------|--------|------|
| TP=8, EP=8, DP=1, batch=256 | 26.79 GB | 11.62 GB | 22.50 GB | 62.92 GB | 1.08 GB |
| TP=4, EP=8, DP=2, batch=180 | 27.40 GB | 16.35 GB | 15.82 GB | 61.56 GB | 2.44 GB |
| TP=2, EP=8, DP=4, batch=100 | 28.60 GB | ~18 GB | ~9 GB | ~57 GB | ~7 GB |

### 并行配置显存验证（16卡，TP×DP=EP=16）

| 配置 | 权重/卡 | KV Cache | Activation | 总占用 | 可用 |
|------|---------|----------|------------|--------|------|
| TP=8, EP=16, DP=2, batch=400 | 13.70 GB | ~18 GB | ~35 GB | ~68 GB | ~-4 GB |
| TP=8, EP=16, DP=2, batch=350 | 13.70 GB | ~16 GB | ~31 GB | ~62 GB | ~2 GB |
| TP=4, EP=16, DP=4, batch=260 | 14.30 GB | ~24 GB | ~23 GB | ~62 GB | ~2 GB |
| TP=2, EP=16, DP=8, batch=200 | 15.51 GB | ~36 GB | ~18 GB | ~71 GB | ~-7 GB |
| TP=2, EP=16, DP=8, batch=180 | 15.51 GB | ~33 GB | ~16 GB | ~66 GB | ~-2 GB |

## 4. 并行策略配置

### PD 混部部署（推荐配置）

#### 方案A：8卡配置（成本优化）

**推荐配置: TP=8, EP=8, DP=1**

| 参数 | 值 |
|------|-----|
| 总GPU数 | 8 |
| Tensor Parallel | 8 |
| Expert Parallel | 8（256 experts / 8 = 32 experts per card） |
| Data Parallel | 1 |
| 每副本卡数 | 8 |
| 最大batch/replica | 256 |
| 总最大batch | 256 |

#### 方案B：16卡配置（吞吐量最大化）

**推荐配置: TP=8, EP=16, DP=2**

| 参数 | 值 |
|------|-----|
| 总GPU数 | 16 |
| Tensor Parallel | 8 |
| Expert Parallel | 16（256 experts / 16 = 16 experts per card） |
| Data Parallel | 2 |
| 模型副本数 | 2 |
| 每副本卡数 | 8 |
| 最大batch/replica | ~350 |
| 总最大batch | ~700 |

### 配置对比分析

| 指标 | 8卡 TP=8,DP=1 | 16卡 TP=8,DP=2 | 16卡 TP=4,DP=4 | 16卡 TP=2,DP=8 |
|------|---------------|----------------|----------------|----------------|
| 总batch | 256 | 700 | 1040 | 1440 |
| 通信开销 | 最低 | 低 | 中等 | 高 |
| 部署复杂度 | 简单 | 中等 | 中等 | 复杂 |
| 推荐场景 | 初始部署 | 扩展阶段 | 成本敏感 | TPS极优先 |

**对于TPS优先目标，推荐16卡配置 TP=8, DP=2, EP=16，总batch可达700**

## 5. 性能估算

### 8卡配置性能（TP=8, EP=8, DP=1, batch=256）

基于Ascend-910B算力估算（FP8算力约256 TFLOPS@core）：

**Prefill阶段**：
- 算量：228.70B × 1024 × 2 FLOPs ≈ 468 TFLOPs
- Prefill时延（TP=8）：~200-400 ms
- 吞吐量：1024 tokens / 0.3s ≈ 3,400 tokens/s

**Decode阶段**：
- 算量：228.70B × 1 × 2 FLOPs ≈ 457 GFLOPs/token
- Decode时延：~15-25 ms/token
- 单卡TPS：256 tokens / 0.02s ≈ 12,800 tokens/s

**总TPS估算**：
```
TPS = batch × (S_in + S_out) / (TTFT + S_out × TPOT)
TPS = 256 × 1536 / (300ms + 512 × 20ms)
TPS = 393,216 / 10.3s ≈ 38,200 tokens/s
```

### 16卡配置性能（TP=8, EP=16, DP=2, batch=350/replica）

| 指标 | 8卡配置 | 16卡配置 |
|------|---------|----------|
| 最大batch | 256 | 700 (2×350) |
| TTFT | 200-400 ms | 200-400 ms |
| TPOT | 15-25 ms | 15-25 ms |
| 总TPS | ~38,000 | ~105,000 |

## 6. 约束验证

### 8卡配置（TP=8, EP=8, DP=1）

| 约束 | 配置 | 状态 |
|------|------|------|
| TP ≤ 单机卡数 | 8 ≤ 8 | 通过 |
| TP × DP = EP | 8 × 1 = 8 | 通过 |
| EP ≤ expert数量 | 8 ≤ 256 | 通过 |
| EP < 实例总卡数 | 8 < 8 | 不通过* |

*注: EP = 总卡数是特殊情况，适用于单副本MoE模型部署

### 16卡配置（TP=8, EP=16, DP=2）

| 约束 | 配置 | 状态 |
|------|------|------|
| TP ≤ 单机卡数 | 8 ≤ 8 | 通过 |
| TP × DP = EP | 8 × 2 = 16 | 通过 |
| EP ≤ expert数量 | 16 ≤ 256 | 通过 |
| EP < 实例总卡数 | 16 < 16 | 不通过* |

*注: EP = 总卡数是特殊情况，适用于单副本MoE模型部署

## 7. 实现注意事项

1. **EP分片验证**: 部署前确认MoE expert确实按EP分片，可通过 `nvidia-smi` 或昇腾工具查看每卡权重内存

2. **Batch调度**: 使用vLLM或SGLang的自适应batch策略，根据实际负载动态调整batch

3. **KV Cache管理**: 使用PagedAttention优化显存利用，可提升有效batch上限20-30%

4. **Prefill/Decode混合**: 在混部策略下，控制prefill和decode请求比例，避免compute bound

5. **监控指标**: GPU利用率、显存使用率、TPS、TTFT、TPOT

6. **扩展建议**: 初期使用8卡TP=8配置，后续扩展到16卡时采用TP=8,DP=2配置

## 附录：关键公式

| 公式 | 说明 |
|------|------|
| `Decode TPS_混部 = batch_size × (S_in + S_out) / (TTFT + S_out × TPOT)` | 混部TPS |
| `KV Cache/卡 = 2 × batch × seq_len × num_kv_heads × head_dim × num_layers / (TP × EP)` | 每卡KV Cache |
| `专家分片 = num_experts / EP` | 每卡expert数量 |
| `batch_max ≈ (64GB - weights - 2GB) / (seq_len × kv_per_token + act_per_token)` | 单卡最大batch估算 |

## 附录：配置修复记录

| 日期 | 修改 | 原因 |
|------|------|------|
| 2026-03-22 | 将MiniMax-M2.5.yaml中expert权重的parallel_strategy从replicated改为EP | MoE模型的expert权重需要EP分片才能正确估算显存 |

**修改前的expert配置（错误）**:
```yaml
block_sparse_moe.experts.N.w1.weight: {..., parallel_strategy: replicated}
block_sparse_moe.experts.N.w2.weight: {..., parallel_strategy: replicated}
block_sparse_moe.experts.N.w3.weight: {..., parallel_strategy: replicated}
```

**修改后的expert配置（正确）**:
```yaml
block_sparse_moe.experts.N.w1.weight: {..., parallel_strategy: EP}
block_sparse_moe.experts.N.w2.weight: {..., parallel_strategy: EP}
block_sparse_moe.experts.N.w3.weight: {..., parallel_strategy: EP}
```
