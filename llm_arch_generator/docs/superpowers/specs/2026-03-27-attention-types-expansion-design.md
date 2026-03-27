# Attention Types Expansion Design

## Context

当前llm_arch_generator的templates对attention类型的描述不准确或不完整：
1. **DeepSeek V3.2** 实际使用 MLA + DSA (DeepSeek Sparse Attention)，但模板只声明了MLA
2. **GLM家族** 未声明attention_impl，实际使用GQA（GLM-4）或MLA+DSA（GLM-5）
3. **Baichuan家族** 未声明attention_impl，各模型使用不同attention类型
4. **混合attention**（MLA+SWA、DSA+MLA等）的mermaid展开逻辑缺失

用户要求：
- 每个模型显式声明attention类型，不依赖common.yaml隐式继承
- 混合attention在mermaid图中展开，并展示各组件间的连接关系
- 宁可重复也要准确

---

## Attention类型分类体系

### 单Attention类型

每个具体的attention机制是一个单类型，独立声明：

| attention_type | 说明 | 基础架构 | 代表模型 |
|---|---|---|---|
| `standard` | 标准MHA | MHA | 旧模型 |
| `mqa` | Multi-Query Attention | MHA | 研究模型 |
| `gqa` | Grouped-Query Attention | MHA | Llama3, Qwen3 |
| `mla` | Multi-Head Latent Attention | MLA | DeepSeek V3, Kimi K2 |
| `swa` | Sliding Window Attention | 在GQA/MLA上叠加mask | Mistral, Gemma 3 |
| `dsa` | DeepSeek Sparse Attention | MLA + indexer | DeepSeek V3.2, GLM-5 |
| `linear` | Linear Attention | 近似替代softmax | Kimi Lightning |
| `gated_deltanet` | Gated DeltaNet | 线性注意力变体 | Qwen3-Next |
| `gated_attention` | Gated Attention | 带门控的注意力 | Trinity 400B |

**关键区分**：
- `swa` = 在GQA/MLA/GQA+MLA上叠加滑动窗口mask，是单一机制
- `dsa` = 在MLA基础上加了indexer选择top-K KV cache，是MLA的变体
- `swa`和`dsa`**不是hybrid**，是带变体的单attention类型

### Hybrid类型（真正的组合）

Hybrid = **2个及以上attention类型同时作用于同一层或链式叠加**，真正的组合：

| Hybrid模式 | 说明 | 数据流 | 例子 |
|---|---|---|---|
| **Parallel** | 同层多attention并行，输出合并 | Input → [A] → + → [B] → → Output | GatedDeltaNet + GatedAttn (Qwen3-Next) |
| **Alternating** | 不同层用不同attention | Layer1: A → Layer2: B → Layer3: A | GPT-OSS, MiMo |
| **Chain** | 同层内链式叠加（一个attention的输出直接作为下一个输入） | A投影 → B选择 → C处理 → Softmax | MLA + DSA（MLA→indexer→Softmax） |

**Parallel模式的merge方式**：

| merge_type | 说明 |
|---|---|
| `add` | 元素级相加 |
| `gate` | 门控（逐元素乘） |
| `concat` | 拼接后投影 |

**注意**：
- `gqa + swa` (Mistral) = 单attention类型`swa`（因为GQA是基础框架，SWA是在它上面叠加mask）
- `mla + dsa` (DeepSeek V3.2) = 单attention类型`dsa`（因为DSA = MLA + indexer）
- `gqa + swa` (GPT-OSS) = **Hybrid Alternating**（因为不同层用不同类型，且有layer_types_key区分）
- `gated_deltanet + gated_attention` (Qwen3-Next) = **Hybrid Parallel**（同层并行，输出merge）

---

## Template Schema变更

### 每个模型的YAML文件必须显式声明

```yaml
model_type: <identifier>
family: <family-name>
model_name: <human-name>
source: <huggingface-id>

config:
  hidden_size: <int>
  num_hidden_layers: <int>
  # ... 其他config字段

# === Attention类型（必须显式声明） ===
# 单一attention类型：直接声明
attention_type: <standard|mqa|gqa|mla|swa|dsa|linear|gated_deltanet|gated_attention>

# Hybrid（Parallel/Alternating/Chain模式）：列表声明
# 仅当同层有多个attention并行运作，或不同层用不同attention时使用
attention_hybrid_types:
  - <type1>
  - <type2>
attention_hybrid_mode: <parallel|alternating|chain>   # Hybrid模式
attention_merge_type: <add|gate|concat>                # 仅parallel模式需要

# === 各类型的参数（按需声明）===
# MLA参数
mla:
  kv_lora_rank_key: <config-key>
  q_lora_rank_key: <config-key>
  qk_rope_head_dim_key: <config-key>
  qk_nope_head_dim_key: <config-key>
  v_head_dim_key: <config-key>

# SWA参数
sliding_window_key: <config-key>   # e.g., sliding_window

# DSA参数
dsa:
  sparse_topk_key: <config-key>     # top-K tokens selected
  block_size_key: <config-key>      # block indexing granularity

# GQA/MQA参数
kv_heads_key: <config-key>         # e.g., num_key_value_heads

# Hybrid Alternating模式（不同层用不同attention）
layer_types_key: <config-key>       # e.g., layer_types
hybrid_ratio: <string>              # e.g., "5:1" (SWA:Full)

template: <family>/<model-name>.yaml   # 指向自身（显式声明，不继承）
```

### Schema迁移路径（attention_impl → attention_type / attention_hybrid_types）

现有模板使用 `attention_impl: <type>` 声明attention类型：

| 旧字段 | 新字段 | 说明 |
|---|---|---|
| `attention_impl: mla` | `attention_type: mla` | 单MLA，无变化 |
| `attention_impl: gqa` | `attention_type: gqa` | 单GQA，无变化 |
| `attention_impl: sliding_window` | `attention_type: swa` | 重命名 |
| 新增 | `attention_hybrid_types: [mla, dsa]` | DeepSeek V3.2等混合类型 |

**迁移规则：**
- 新模板文件使用 `attention_type`（单一）或 `attention_hybrid_types`（组合）
- `attention_impl` 字段保留但标记为deprecated，不在新模板中使用
- SKILL.md生成逻辑：优先读 `attention_type` 或 `attention_hybrid_types`，如不存在则回退到 `attention_impl`

### common.yaml的role变更

common.yaml仅作为**默认值参考**，每个模型文件必须override自己的attention类型：

```yaml
# common.yaml — 仅提供家族默认值参考
model_type: <family-model-type>
family: <family-name>

# 家族默认（可被模型文件override）
attention_type: gqa

# 家族默认MLA参数（可被override）
mla: {...}

# 家族默认MoE参数（可被override）
moe: {...}
```

---

## Mermaid图生成逻辑变更

### 核心规则

1. **单attention类型**：直接展开对应投影结构
2. **DSA**：MLA框架（Q/K/V压缩 → 低秩） + indexer（top-K选择） → Softmax
3. **SWA**：基础框架（GQA/MLA） + sliding_window_mask → Softmax
4. **Hybrid Parallel**：同层多attention并行，输出通过merge_type合并
5. **Hybrid Alternating**：不同层用不同attention，在stack中描述层交替模式
6. **Hybrid Chain**：同层内链式叠加（如A的输出直接作为B的输入）

### 各Attention类型的Mermaid展开模板

#### 1. GQA (Standard)

```mermaid
Q_proj["Q_proj<br/>H → num_heads×head_dim"]:::attention
K_proj["K_proj<br/>H → num_kv_heads×head_dim"]:::attention
V_proj["V_proj<br/>H → num_kv_heads×head_dim"]:::attention
Q_proj --> Softmax["Softmax(Q·Kᵀ/√d)"]:::attention
K_proj --> Softmax
V_proj --> Softmax
Softmax --> O_proj["O_proj<br/>num_heads×head_dim → H"]:::attention
```

#### 2. MLA (DeepSeek/Kimi)

```mermaid
Q_A["Q_A_Proj<br/>H → q_lora_rank"]:::attention
Q_A --> Q_LN["Q RMSNorm"]:::attention
Q_LN --> Q_B["Q_B_Proj<br/>q_lora_rank → H"]:::attention
Q_B --> ConcatQ["Concat(Q_nope + Q_rope)"]:::attention

KV_A["KV_A_Proj<br/>H → kv_lora_rank"]:::attention
KV_A --> KV_LN["KV RMSNorm"]:::attention
KV_LN --> ConcatK["Concat(K_rope + K_nope)"]:::attention
KV_LN --> K_B["K_B_Proj<br/>kv_lora_rank → num_heads×head_dim"]:::attention
ConcatK --> RoPE["RoPE"]:::attention
RoPE --> Softmax["Softmax(Q·Kᵀ/√d)"]:::attention
ConcatQ --> Softmax
K_B --> Softmax
Softmax --> O_Proj["O_Proj<br/>H → H"]:::attention
```

#### 3. DSA (DeepSeek V3.2, GLM-5) — MLA + indexer

```mermaid
%% MLA框架
Q_A["Q_A_Proj"]:::attention --> Q_LN["Q RMSNorm"]:::attention
Q_LN --> Q_B["Q_B_Proj"]:::attention
Q_B --> ConcatQ["Concat(Q_nope + Q_rope)"]:::attention

KV_A["KV_A_Proj"]:::attention --> KV_LN["KV RMSNorm"]:::attention
KV_LN --> ConcatK["Concat(K_rope + K_nope)"]:::attention
KV_LN --> K_B["K_B_Proj"]:::attention
ConcatK --> RoPE["RoPE"]:::attention

%% indexer（top-K选择）
RoPE --> Indexer["Indexer<br/>top-K by block"]:::attention
ConcatQ --> Indexer
K_B --> Indexer
Indexer --> Softmax["Softmax(Q·Kᵀ/√d)"]:::attention
Softmax --> O_Proj["O_Proj"]:::attention
```

#### 4. SWA (Mistral, Gemma 3, MiMo) — GQA/MLA + sliding window mask

```mermaid
Q_proj["Q_proj"]:::attention
K_proj["K_proj"]:::attention
V_proj["V_proj"]:::attention
Q_proj --> SWA_Mask["Sliding Window<br/>mask(window_size)"]:::attention
K_proj --> SWA_Mask
SWA_Mask --> Softmax["Softmax(Q·Kᵀ/√d)"]:::attention
V_proj --> Softmax
Softmax --> O_proj["O_proj"]:::attention
```

#### 5. Hybrid Parallel (Qwen3-Next) — Gated DeltaNet + Gated Attention

```mermaid
%% Parallel: 同层两attention并行，输出门控合并
Q["Q_proj"]:::attention
Q --> DeltaNet["Gated DeltaNet"]:::attention
Q --> GatedAttn["Gated Attention"]:::attention
DeltaNet --> Gate["Gate"]:::attention
GatedAttn --> Gate
Gate --> O_Proj["O_Proj"]:::attention
```

#### 6. Hybrid Alternating (GPT-OSS, MiMo) — 不同层用不同attention

```mermaid
%% Alternating: 层交替模式
subgraph Transformer_Stack ["[SWA] × N, [Full] × 1, ... (交替)"]
  direction TB
  SWA_Block["SWA Block<br/>GQA + SWA"]:::attention
  Full_Block["Full Block<br/>GQA + Full"]:::attention
  SWA_Block --> Full_Block
end
```

**Alternating模式展开**：
- `layer_types_key`（如 `layer_types`）声明每层的类型数组
- `hybrid_ratio`（如 `"5:1"`）描述交替比例
- mermaid stack pattern: `[swa_block] × 5, [full_block] × 1, ...`

#### 7. Hybrid Chain (未来可能)

同层内链式叠加（如MLA + Linear），按A → B → C顺序串联数据流。

---

## 需要修改的文件清单

### Templates（逐个模型显式声明attention类型）

| 文件 | 修改内容 |
|---|---|
| `templates/deepseek/common.yaml` | 添加 `attention_type: mla` 默认值 |
| `templates/deepseek/deepseek-v3.yaml` | `attention_type: mla` |
| `templates/deepseek/deepseek-v3.1.yaml` | `attention_type: mla` |
| `templates/deepseek/deepseek-v3.2.yaml` | `attention_type: dsa` + dsa参数 |
| `templates/deepseek/deepseek-v2.yaml` | `attention_type: mla` |
| `templates/deepseek/deepseek-v2.5.yaml` | `attention_type: mla` |
| `templates/deepseek/deepseek-r1.yaml` | `attention_type: mla` |
| `templates/kimi/common.yaml` | 添加 `attention_type: mla` 默认值 |
| `templates/kimi/kimi-k2.0.yaml` | `attention_type: mla` |
| `templates/kimi/kimi-k2.5.yaml` | `attention_type: mla` |
| `templates/qwen/common.yaml` | 添加 `attention_type: gqa` 默认值 |
| `templates/qwen/qwen3-32b.yaml` | `attention_type: gqa` |
| `templates/qwen/qwen3-8b.yaml` | `attention_type: gqa` |
| `templates/qwen/qwen3-4b.yaml` | `attention_type: gqa` |
| `templates/qwen/qwen3-1.7b.yaml` | `attention_type: gqa` |
| `templates/qwen/qwen3-0.6b.yaml` | `attention_type: gqa` |
| `templates/qwen/qwen3-vl-8b.yaml` | `attention_type: gqa` |
| `templates/qwen/qwen3.5-397b-a17b.yaml` | `attention_type: gqa` |
| `templates/qwen/qwen3.5-122b-a10b.yaml` | `attention_type: gqa` |
| `templates/qwen/qwen3.5-35b-a3b.yaml` | `attention_type: gqa` |
| `templates/qwen/qwen3.5-27b.yaml` | `attention_type: gqa` |
| `templates/qwen/qwen2.5-vl-7b.yaml` | `attention_type: gqa` |
| `templates/llama/common.yaml` | 保留默认值 |
| `templates/llama/llama-2-7b.yaml` | `attention_type: standard` |
| `templates/llama/llama-3-8b.yaml` | `attention_type: gqa` |
| `templates/llama/llama-3.1-8b.yaml` | `attention_type: gqa` |
| `templates/llama/llama-3.2-1b.yaml` | `attention_type: gqa` |
| `templates/llama/llama-3.2-3b.yaml` | `attention_type: gqa` |
| `templates/llama/llama-3.2-11b.yaml` | `attention_type: gqa` |
| `templates/llama/llama-3.3-70b.yaml` | `attention_type: gqa` |
| `templates/glm/common.yaml` | 添加 `attention_type: gqa` 默认值 |
| `templates/glm/glm-4-9b.yaml` | `attention_type: gqa` |
| `templates/glm/glm-4-28b.yaml` | `attention_type: gqa` |
| `templates/glm/glm-4v.yaml` | `attention_type: gqa` |
| `templates/glm/glm-5.yaml` | `attention_type: dsa` + dsa参数 |
| `templates/baichuan/common.yaml` | 添加attention类型注释说明 |
| `templates/baichuan/baichuan-13b.yaml` | `attention_type: standard` (MHA) |
| `templates/baichuan/baichuan2-7b.yaml` | `attention_type: standard` (MHA) |
| `templates/baichuan/baichuan-m3.yaml` | `attention_type: gqa` |
| `templates/mistral/common.yaml` | 添加 `attention_type: swa` 默认值 |
| `templates/mistral/mistral-7b-v0.1.yaml` | `attention_type: swa` |
| `templates/mistral/mistral-7b-v0.3.yaml` | `attention_type: swa` |
| `templates/mistral/mistral-nemo.yaml` | `attention_type: swa` |
| `templates/mistral/mistral-large.yaml` | `attention_type: swa` |
| `templates/minimax/common.yaml` | 添加 `attention_type: gqa` 默认值 |
| `templates/minimax/minimax-m2.yaml` | `attention_type: gqa` |
| `templates/minimax/minimax-m2.5.yaml` | `attention_type: gqa` |
| `templates/mimo/common.yaml` | 添加 `attention_type: swa` 默认值 |
| `templates/mimo/mimo-v2-flash.yaml` | `attention_type: swa`, `hybrid_ratio: "5:1"` (SWA+Global层交替) |
| `templates/mimo/mimo-vl-7b.yaml` | `attention_type: swa` |
| `templates/gpt-oss/common.yaml` | 添加 `attention_type: swa` 默认值 |
| `templates/gpt-oss/gpt-oss-120b.yaml` | `attention_hybrid_types: [swa, full]`, `attention_hybrid_mode: alternating`, `layer_types_key: layer_types`, `hybrid_ratio: "1:1"` |
| `templates/gpt-oss/gpt-oss-20b.yaml` | `attention_hybrid_types: [swa, full]`, `attention_hybrid_mode: alternating`, `layer_types_key: layer_types`, `hybrid_ratio: "1:1"` |

### SKILL.md变更

在mermaid生成逻辑中增加attention类型判断分支：

```
Step 4生成逻辑：
if attention_hybrid_types 存在:
    if attention_hybrid_mode == "parallel":
        展开Parallel并行合并结构 + attention_merge_type
    elif attention_hybrid_mode == "alternating":
        展开Alternating层交替stack结构
    elif attention_hybrid_mode == "chain":
        按链式顺序串联各类型展开
elif attention_type == "dsa":
    展开DSA结构（MLA框架 + indexer）
elif attention_type == "swa":
    展开SWA结构（基础框架 + sliding_window_mask）
elif attention_type == "mla":
    展开MLA链
elif attention_type == "gqa":
    展开GQA链
elif attention_type == "standard":
    展开标准MHA链
...（其他单类型同理）
```

### verify_mermaid.py 变更（可扩展性）

**核心改动：只做结构验证，不依赖hardcode节点名**

```python
# 1. ALWAYS_ALONE_NODES 改为通用规则匹配
ALWAYS_ALONE_NODES_REGEX = re.compile(
    r'.*_[Dd]etail|'      # *_Detail 子图节点
    r'Hybrid_|'            # Hybrid_* 子图节点
    r'DSA_|'               # DSA_* 节点
    r'Linear_|'           # Linear_* 节点
    r'Gated_'             # Gated_* 节点
)

# 2. TERMINAL_NODES 改为通用规则
TERMINAL_NODE_SUFFIXES = {'_proj', '_out', 'lm_head', 'final_norm',
                          'softmax', '_head', 'o_proj'}

# 3. 结构验证保持不变
# - 所有节点引用有定义 ✓
# - 路径从Input到Output连续 ✓
# - 不检查节点叫什么名字（新类型自动合规）
```

**效果**：未来出现任何新attention类型（如`FlashAttention`、`StateSpace_Attn`等），只要mermaid结构正确，自动pass验证，无需更新白名单。

---

## 实现顺序

1. **更新verify_mermaid.py** — 改为通用规则匹配，移除hardcode白名单
2. **更新SKILL.md mermaid生成逻辑** — 支持attention_hybrid_types列表
3. **更新所有template YAML文件** — 显式声明attention类型
4. **更新SKILL.md"Expected false positives"段落** — 移除过时白名单描述
5. **验证现有mermaid图** — 重新生成DeepSeek V3.2等关键模型的图确认正确
6. **测试完整流程** — 用新模板生成各模型架构图
