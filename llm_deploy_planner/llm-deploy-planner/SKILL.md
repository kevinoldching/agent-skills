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
  - LLM 推理所需 卡 数量评估
  关键词：LLM部署规划、PD分离、Prefill Decode、xPyD、卡数量、LLM推理规划、TP EP DP、并行策略
---

# LLM PD 部署规划工具

本工具用于规划 LLM 推理部署，支持 Prefill/Decode（PD）混合部署或分离部署策略。

## 第一步：收集用户输入

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
| 硬件信息 | 显存大小、单机卡数（默认8卡） |
| 输入输出特征 | 典型输入输出时延、prefill/decode单层时延（TP=1基准，其他需除以TP*η，η=0.7） |
| batch_size | 可选，不提供则反推 |
| 用户场景 | RL、数据生产、Agent轨迹生成、低QPS、RAG、编程辅助、chat、多轮对话等 |
| 性能目标 | TPS优先 或 时延优先（TTFT/TPOT） |

**重要**：先分析用户的输入，如果缺少以下信息，必须逐项提示用户输入或选择跳过。**不得跳过此步骤直接进入下一步**。

### 1. 模型信息
**请告诉我是哪个模型？**
- 模型名称/路径: （如 Qwen-72B、Llama2-70B 或本地权重路径）

### 2. 硬件信息
**请告诉我硬件配置：**
- 显存大小: （每卡显存，如 80 GB、64 GB）
- 单机卡数: （每台机器的卡数，如 8 卡、4 卡；未提供时默认为 8 卡）

### 3. 输入输出特征
**请提供输入输出特征：**
- 典型输入长度: （平均输入 token 数，如 1024、2048 tokens）
- 典型输出长度: （平均输出 token 数，如 512、128 tokens）
- Prefill 单层时延（TP=1 基准）: （单位 ms，如需其他 TP 配置需除以 TP×η，η=0.7）
- Decode 单层时延（TP=1 基准）: （单位 ms）
- batch_size: （可选，不提供将自动反推）

### 4. 用户场景
**请选择您的场景类型：**
- RL（强化学习）、数据生产、Agent轨迹生成、低QPS
- 在线推理、RAG、编程辅助（时延优先）
- chat 等

### 5. 性能目标
**询问性能目标：**
例如：
- TPS优先：`要求达到700TPS，TPOT<100ms`
- 时延优先：`要求TTFT<=70ms，TPS>680`

---


## 第二步：PD 策略决策框架

**决策顺序：按优先级从高到低判断，遇到明确结论则停止。**

### 四优先级决策

**第 0 优先级：模型规模**

- 模型参数规模 <= 8B：默认 PD 混部（除非用户有充足的理由要求使用PD分离）。
- 模型参数规模 > 8B：继续后续决策流程。

**第一优先级：场景**

| 场景 | 推荐策略 |
|------|----------|
| 强化学习、数据生产、Agent轨迹数据生成、多轮对话、低QPS | PD混部 |
| 在线推理、RAG、编程辅助（时延优先） | PD分离 |

> **重要**：场景优先级最高。若场景匹配，**直接得出结论**，不再参考长度比和SLO。

**第二优先级：输入/输出长度比**

```
α ≈ S_in / (S_in + k × S_out), k=2~6, default=3
imbalance = max(α, 1-α)
imbalance < 0.7 → 混部
imbalance >= 0.7 → 分离
```

**重要：长度比分析仅作为辅助参考，场景优先级高于长度比。**

| α | 含义 | 决策影响 |
|---|------|----------|
| 0.5 | 完全平衡 | 无特殊倾向 |
| 0.3~0.7 | 基本平衡 | 支持混部（imbalance < 0.7） |
| <0.3 或 >0.7 | 失衡倾向 | 若 imbalance >= 0.7 则支持分离 |

**第三优先级：SLO指标**（仅在场景和长度比均无明确倾向时参考）
- TTFT&TPOT优先 → PD分离
- TPS优先+卡数有限 → PD混部（无kvcache传输开销、所有卡都可以用于decode）

## 第三步：遍历并行策略组合

在约束条件下，遍历所有可能的 TP/EP/DP 组合，对每个组合调用 `llm-mem-estimator` 验证显存是否满足要求。

### 约束条件

- TP <= 单机卡数 或 TP = 单机卡数 × 机器数量
- TP × DP = EP
- EP_Prefill = TP_prefill × DP_prefill（PD分离时）
- EP_Decode = y × TP_decode × DP_decode（PD分离时，y=1）
- EP <= MoE expert数量（仅MoE模型）
- TP/EP 取值为2的幂；TP 在 [1, 单机卡数] 范围内为 1, 2, 4, 8, ...；超出单机卡数后为 单机卡数×2^k（k=1,2,3,...）；EP=1 仅限非MoE模型；MoE模型 EP >= 单机卡数
- 总卡数必须是单机卡数的整数倍

### EP 标注规则

**重要：EP 仅适用于 MoE（混合专家）模型**

| 模型类型 | EP 取值 | 说明 |
|----------|---------|------|
| Dense 模型 | EP = 1（固定） | 无需 Expert Parallel |
| MoE 模型 | 1 < EP <= min(总卡数, expert数量) | EP 必须不超过 expert 数量 |

### 遍历策略

**核心逻辑**：对每个 EP 候选值，枚举所有满足 `TP × DP = EP` 且 `TP <= 单机卡数` 的 TP 值。

#### 情况1：已提供 batch_size
1. 固定 batch_size
2. **确定 EP 候选列表**：
   - Dense 模型：EP = [1]（固定）
   - MoE 模型：EP = [单机卡数, 单机卡数×2, 单机卡数×3, ...]（EP >= 单机卡数）
3. **对每个 EP 值，枚举所有有效 TP**：
   - 遍历 TP/EP 值：2的幂；[1, 单机卡数] 范围内为 1, 2, 4, 8, ...；超出单机卡数后为 单机卡数×2^k（k=1,2,3,...）；仅当 TP 能整除 EP 且 TP/EP 在候选列表中时有效
   - 计算 DP = EP / TP
   - **仅当 TP 能整除 EP（即 DP 为整数）且 TP <= 单机卡数 或 TP = 单机卡数×N 时**，该 TP 值有效
   - 跳过无效 TP（如 EP=32 时，TP=6 因 32/6 非整数而无效）
4. 对每个 (batch_size, TP, EP, DP) 组合调用 `llm-mem-estimator` 验证显存是否满足
5. **必须验证所有有效 TP 值，不得在找到第一个通过的配置后停止**

#### 情况2：未提供 batch_size
1. **确定 EP 候选列表**：
   - Dense 模型：EP = [1]（固定）
   - MoE 模型：EP = [单机卡数, 单机卡数×2, 单机卡数×3, ...]（EP >= 单机卡数）
2. **对每个 EP 值，枚举所有有效 TP**：
   - 遍历 TP/EP 值：2的幂；[1, 单机卡数] 范围内为 1, 2, 4, 8, ...；超出单机卡数后为 单机卡数×2^k（k=1,2,3,...）；仅当 TP 能整除 EP 且 TP/EP 在候选列表中时有效
   - 计算 DP = EP / TP
   - **仅当 TP 能整除 EP 且 TP <= 单机卡数 或 TP = 单机卡数×N 时**，该 TP 值有效
3. 对每个 (TP, EP, DP) 组合调用 `llm-mem-estimator`，记录该配置下的 max_batch_size
4. 最终得到所有 (max_batch_size, TP, EP, DP) 候选配置
5. **必须验证所有有效 TP 值，不得在找到第一个通过的配置后停止**

### 调用方式

使用 Skill tool 调用 `llm-mem-estimator`，每次调用传入不同的 (TP, EP, DP) 组合：

```
模型: <模型名称>
芯片: <芯片型号，如 H100-80GB、A100-80GB>
并行策略: TP=<TP值>, EP=<EP值>, DP=<DP值>
batch_size: <目标batch_size>
输入长度: <输入token数>
```

### PD 分离场景

Prefill 和 Decode 阶段分别遍历和验证：
- **Prefill**：compute-bound，batch_size 较大，激活值是主要瓶颈
- **Decode**：memory-bound，KV Cache 是主要瓶颈

### 收集结果

对每个有效配置记录：
- `(TP, EP, DP)` 值
- `max_batch_size`：该配置下最大 batch_size
- `每卡显存占用`：权重 + KV Cache + 激活值

### 错误处理

若 `llm-mem-estimator` 调用失败，使用手动估算：
```
权重（FP16）≈ 参数数量（B）× 2 GB
KV Cache 每 token ≈ 2 × hidden_size × num_layers × 2 / 1024² MB
```
在输出中注明"部分数据来自手动估算"

## 第四步：验证与选择

基于第三步收集的有效配置，使用性能公式验证并选择最优配置。

### 估算公式

**混部:**
```
Decode TPS = batch_size × (S_in + S_out) / (TTFT + S_out × TPOT)
```

**分离:**
```
Decode TPS = batch_size / TPOT
```

### 情况1: 未提供 batch_size

1. 第三步已筛选出所有候选 (max_batch_size, TP, EP, DP) 配置
2. 根据 max_batch_size 计算每个配置的预估 TPS
3. 根据性能目标选择最优配置：
   - **TPS优先**：选择 max_batch_size × (S_in + S_out) / (TTFT + S_out × TPOT) 最大的配置，如果用户也指定了TTFT/TPOT，也需要满足
   - **时延优先（TTFT/TPOT）**：选择满足用户指定TTFT/TPOT时TPS最高的配置
4. 输出最优 (max_batch_size, TP, EP, DP) 组合及总卡数

### 情况2: 已提供 batch_size

1. 第三步已对目标 batch_size 做过验证，得到所有满足显存的 (TP, EP, DP) 候选配置
2. 根据性能目标选择最优配置：
   - **TPS优先**：选择 TP × DP（总计算力）最大的配置，如果用户也指定了TTFT/TPOT，也需要满足
   - **时延优先（TTFT/TPOT）**：选择满足用户指定TTFT/TPOT时TPS最高的配置
3. 若没有配置能满足目标 batch_size，报告不可行并建议调整，否则输出最优 (TP, EP, DP) 组合及总卡数

### xPyD 计算（仅 PD 分离场景）

**关键约束**：
- **y = 1（固定，不可更改）** - Decode 实例数只能为 1
- Prefill每实例卡数 = TP_prefill × DP_prefill
- Decode每实例卡数 = TP_decode × DP_decode
- **Prefill总卡数 = Decode总卡数**（即 x × TP_prefill × DP_prefill = y × TP_decode × DP_decode）
- **x × TP_prefill × DP_prefill + y × TP_decode × DP_decode <= 总可用卡数**

**计算步骤**：
1. 确定总卡数（满足单机卡数整数倍约束）
2. 确定 Prefill 和 Decode 的并行策略（TP, EP, DP）
3. 使 Prefill总卡数 = Decode总卡数
4. 计算 x = ceil(Prefill总卡数 / (TP_prefill × DP_prefill))
5. 验证总卡数不超过可用卡数

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
| 总卡数 | [X] |
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

## 5. 实现注意事项
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
2. `TP × DP = EP`
3. `EP < 实例总卡数`
4. `[若MoE] EP >= 2 且 EP <= expert数量`
5. **xPyD: y = 1（固定，不可更改），x × TP_prefill × DP_prefill = y × TP_decode × DP_decode（Prefill和Decode总卡数必须相等）**
6. 总卡数 <= 可用卡数
7. TP/EP 取值为2的幂；TP 在 [1, 单机卡数] 范围内为 1, 2, 4, 8, ...；超出单机卡数后为 单机卡数×2^k（k=1,2,3,...）；EP=1 仅限非MoE模型；MoE模型 EP >= 单机卡数
8. **总卡数必须是单机卡数的整数倍**（total_cards % 单机卡数 == 0，避免出现56、42等无法整除的值）
