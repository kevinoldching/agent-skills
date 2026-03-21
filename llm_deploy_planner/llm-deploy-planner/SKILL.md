---
name: llm-deploy-planner
description: |
  规划 LLM 推理部署，支持 Prefill/Decode (PD) 混合部署或分离部署策略。
  适用于以下场景：
  - LLM 推理部署规划
  - PD（Prefill/Decode）分离或混合部署
  - GPU/NPU 资源规划
  - 并行策略（TP/EP/DP）配置
  - xPyD 部署（如 4P1D 表示 4 个 Prefill 实例 + 1 个 Decode 实例）
  - LLM 推理所需 GPU 数量评估
  关键词：LLM部署规划、PD分离、Prefill Decode、xPyD、GPU数量、LLM推理规划、TP EP DP、并行策略
---

# LLM PD 部署规划工具

本工具用于规划 LLM 推理部署，支持 Prefill/Decode（PD）混合部署或分离部署策略。

## 第一步：收集用户输入

按以下分类逐项提示用户输入信息：

### 1. 模型信息
请提供：
- **模型名称/路径**: 模型标识（如 Qwen-72B、Llama2-70B）或本地权重路径

### 2. 硬件信息
请提供：
- **显存大小**: 每卡显存，如 80 GB、64 GB
- **单机卡数**: 每台机器的卡数，如 8 卡、4 卡

### 3. 输入输出特征
请提供：
- **典型输入长度**: 平均输入 token 数，如 1024、2048 tokens
- **典型输出长度**: 平均输出 token 数，如 512、128 tokens
- **Prefill 单层时延（TP=1 基准）**: 单位 ms，如需其他 TP 配置需除以 TP×η（η=0.7）
- **Decode 单层时延（TP=1 基准）**: 单位 ms
- **batch_size**（可选）: 如不提供将自动反推

### 4. 用户场景
请选择或提供：
- **场景类型**: RL（强化学习）、数据生产、Agent轨迹生成、编程辅助、chat 等

### 5. 性能目标
请选择：
- **优先目标**: TPS优先 / 时延优先（TTFT/TPOT）

**示例**：
- TPS优先：`要求达到700TPS，TPOT<100ms`
- 时延优先：`要求TTFT<=70ms，TPS>680`

---

### 输入模板（供参考）

```
模型名称/路径: [模型标识或本地权重路径]
硬件信息:
- 显存大小: [X] GB
- 单机卡数: [X] 卡
输入输出特征:
- 典型输入长度: [X] tokens
- 典型输出长度: [X] tokens
- Prefill单层时延（TP=1基准）: [X] ms
- Decode单层时延（TP=1基准）: [X] ms
- batch_size: [X]（可选）
用户场景: [场景类型]
性能目标: [TPS优先/时延优先]
```

### 输入字段说明

| 字段 | 说明 |
|------|------|
| 模型名称/路径 | 模型标识（如 Qwen-72B）或本地权重路径 |
| 硬件信息 | 显存大小、单机卡数 |
| 输入输出特征 | 典型输入输出时延、prefill/decode单层时延（TP=1基准，其他需除以TP*η，η=0.7） |
| batch_size | 可选，不提供则反推 |
| 用户场景 | RL、数据生产、Agent轨迹生成、编程辅助、chat等 |
| 性能目标 | TPS优先 或 时延优先（TTFT/TPOT） |

## 第二步：PD 策略决策框架

### 三优先级决策

**第一优先级：场景**

| 场景 | 推荐策略 |
|------|----------|
| 强化学习、数据生产、Agent轨迹数据生成 | PD混部 |
| 在线推理、编程辅助（时延优先） | PD分离 |

**第二优先级：输入/输出长度比**

```
α ≈ S_in / (S_in + k × S_out), k=2~6, default=3
imbalance = max(α, 1-α)
imbalance < 0.8 → 混部
imbalance >= 0.8 → 分离
```

| α | 含义 |
|---|------|
| 0.5 | 完全平衡 |
| 0.3~0.7 | 基本平衡 |
| 0.2~0.8 | 可接受 |
| <0.2 或 >0.8 | 明显失衡 |

**第三优先级：SLO指标**
- TTFT&TPOT优先 → PD分离
- TPS优先+卡数有限 → PD混部（无kvcache传输开销、所有卡都可以用于decode）

## 第三步：调用显存估算

根据 PD 策略决策结果，使用 `llm_mem_estimator` skill 估算显存占用：

### PD 混部场景
- 模型权重大小
- KV Cache 显存占用（基于 batch_size 和序列长度）
- 不同 TP 配置下的显存分布

### PD 分离场景
分别估算 Prefill 和 Decode 的显存需求：
- Prefill：权重 + 激活值（compute-bound，batch_size 较大）
- Decode：权重 + KV Cache（memory-bound，需要缓存更多上下文）

## 第四步：并行策略计算

### 约束条件

- TP <= 单机卡数 或 TP = 单机卡数 × 机器数量（当单机无法放下模型权重时，需多机扩展 TP）
- TP × DP <= EP
- EP < P/D实例总卡数
- EP < MoE expert数量（仅MoE模型）
- TP/EP/DP 为2的幂（1, 2, 4, 8, 16, 32）

### EP 标注规则

**重要：EP 仅适用于 MoE（混合专家）模型**

| 模型类型 | EP 取值 | 说明 |
|----------|---------|------|
| Dense 模型 | EP = 1 或 EP = TP | 无需 Expert Parallel |
| MoE 模型 | 1 < EP < min(总卡数, expert数量) | EP 必须小于 expert 数量 |

### 估算公式

**混部:**
```
Decode TPS = batch_size × (S_in + S_out) / (TTFT + S_out × TPOT)
```

**分离:**
```
Decode TPS = batch_size / TPOT
```

### 估算流程

**情况1: 未提供 batch_size**
1. 调用 llm_mem_estimator 反推符合要求的 (batch_size, TP, EP, DP)
2. 根据 TTFT vs (batch_size×S_in)/(TP×η×DP)、TPOT vs batch_size/(TP×η×DP) 推导最高TPS的组合
3. 结合约束和单机卡数，给出最终GPU数和并行策略

**情况2: 已提供 batch_size**
1. 调用 llm_mem_estimator 验证
2. 后续步骤同上（跳过batch遍历）

### xPyD 计算（仅 PD 分离场景）

- Prefill卡数 = TP_prefill × DP_prefill
- Decode卡数 = TP_decode × DP_decode
- Prefill实例数 x = ceil(prefill总卡数 / 单实例卡数)
- 当前仅支持 y=1
- Prefill和Decode需使用相同总卡数

## 第五步：输出格式

```markdown
# LLM PD 部署规划

## 1. 输入摘要
| 字段 | 值 |
|------|-----|
| 模型 | [模型名称] |
| 硬件 | [显存大小] GB × [单机卡数] 卡 |
| 输入长度 | [S_in] tokens |
| 输出长度 | [S_out] tokens |
| 场景 | [场景类型] |
| 性能目标 | [TPS优先/时延优先] |

## 2. PD 策略决策

### 决策结果
**推荐策略: [PD混部/PD分离]**

### 决策依据
| 优先级 | 因素 | 分析 |
|--------|------|------|
| 场景 | [场景类型] | [分析结果] |
| 长度比 | α=[X] | [分析结果] |
| SLO | [TPS/时延优先] | [分析结果] |

## 3. 显存分析
### 模型显存占用
- 权重（FP16）: [X] GB
- KV Cache每token: [X] MB
- 单卡最大batch: [X]

### 并行配置显存验证

列出满足约束条件的 TP/EP/DP 组合：

**约束条件**：
- TP <= 单机卡数 或 TP = 单机卡数×机器数量
- TP × DP <= EP
- TP/EP/DP 为2的幂（1, 2, 4, 8, 16, 32）

**EP 取值**：
- 非MoE模型：固定 EP=1
- MoE模型：1 < EP < min(实例总卡数, expert数量)

| 配置 | 权重 | KV Cache | 总占用 | 可用 |
|------|------|----------|--------|------|
| TP=1, EP=1, DP=1 | [X] GB | [X] GB | [X] GB | [X] GB |
| TP=2, EP=2, DP=1 | [X] GB | [X] GB | [X] GB | [X] GB |
| TP=4, EP=4, DP=1 | [X] GB | [X] GB | [X] GB | [X] GB |
| ... | ... | ... | ... | ... |

## 4. 并行策略配置

根据第二章的决策结果，输出对应的部署配置：

**（若决策为 PD 混部）**

### PD 混部部署
**总卡数: [X]**
**并行策略: TP[X] + EP[X] + DP[X]**

| 参数 | 值 |
|------|-----|
| 总GPU数 | [X] |
| Tensor Parallel | [X] |
| Expert Parallel | [X]（仅MoE模型）|
| Data Parallel | [X] |
| 预估吞吐量 | [X] tokens/s |
| 预估TTFT | [X] ms |
| 预估TPOT | [X] ms |

---

**（若决策为 PD 分离）**

### PD 分离部署
**配置: [x]P[y]D (x=[x], y=[y])**

#### Prefill 阶段
| 参数 | 值 |
|------|-----|
| 实例数 | [x] |
| 每实例卡数 | [X] = TP×DP |
| 总卡数 | [X] |
| Tensor Parallel | [X] |
| Expert Parallel | [X]（仅MoE模型）|
| Data Parallel | [X] |
| 预估吞吐量 | [X] tokens/s |

#### Decode 阶段
| 参数 | 值 |
|------|-----|
| 实例数 | [y] |
| 每实例卡数 | [X] = TP×DP |
| 总卡数 | [X] |
| Tensor Parallel | [X] |
| Expert Parallel | [X]（仅MoE模型）|
| Data Parallel | [X] |
| 预估吞吐量 | [X] tokens/s |

**总卡数: [X]**

## 5. 约束验证
- [ ] TP=[X] <= 单机卡数=[X]
- [ ] TP×DP=[X] <= EP=[X]
- [ ] EP=[X] < 实例总卡数=[X]
- [ ] [若MoE] EP=[X] < expert数量=[X]
- [ ] TP/EP/DP 为2的幂: [X]

## 6. 实现注意事项
- [具体配置建议和注意事项]
```

## 关键公式参考

| 公式 | 说明 |
|------|------|
| `Decode TPS_混部 = batch_size × (S_in + S_out) / (TTFT + S_out × TPOT)` | 混部 TPS |
| `Decode TPS_分离 = batch_size / TPOT` | 分离 TPS |
| `prefill_cards = TP_prefill × DP_prefill` | Prefill 每实例卡数 |
| `decode_cards = TP_decode × DP_decode` | Decode 每实例卡数 |
| `x = ceil(prefill_cards_total / cards_per_instance)` | Prefill 实例数 |

## 约束条件总结

始终验证：
1. `TP <= 单机卡数 或 TP = 单机卡数 × 机器数量`
2. `TP × DP <= EP`
3. `EP < 实例总卡数`
4. `[若MoE] EP < expert数量`
5. xPyD: `y = 1`（当前支持），`x >= 1`
6. 总卡数 <= 可用卡数
7. TP/EP/DP 为2的幂（1, 2, 4, 8, 16, 32）
