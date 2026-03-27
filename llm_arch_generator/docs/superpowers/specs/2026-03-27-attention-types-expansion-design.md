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

### 分类维度

**维度一：按是否稀疏（压缩）**
| 类别 | 说明 | 代表 |
|---|---|---|
| **Full attention** | 所有token都参与attention计算 | MHA, GQA, MQA, MLA |
| **Sparse attention** | 只有部分token参与（取了子集或用近似） | SWA, DSA, Linear Attention, Gated DeltaNet |

**维度二：按具体类型（单attention）**
| attention_type | 说明 | 类别 | 代表模型 |
|---|---|---|---|
| `standard` | 标准MHA，每Q头独立K/V | Full | 旧模型 |
| `mqa` | 所有Q头共享单一K/V头 | Full | 研究模型 |
| `gqa` | Q头分组，每组共享K/V | Full | Llama3, Qwen3, Mistral |
| `mla` | K/V压缩到低秩潜空间 | Full | DeepSeek V3, Kimi K2 |
| `swa` | Sliding Window Attention | Sparse | Mistral, Gemma 3 |
| `dsa` | DeepSeek Sparse Attention（block索引top-K选择） | Sparse | DeepSeek V3.2, GLM-5 |
| `linear` | Linear Attention（近似，O(n)复杂度） | Sparse | Kimi Lightning |
| `gated_deltanet` | Gated DeltaNet | Sparse | Qwen3-Next |
| `gated_attention` | Gated Attention | Full | Trinity 400B |
| `...` | 任何未来出现的新类型 | — | — |

### Hybrid类型（组合）

**Hybrid = 2个及以上单attention类型的组合**，如：

| 组合 | 例子 | 数据流 |
|---|---|---|
| MLA + DSA | DeepSeek V3.2, GLM-5 | MLA投影 → DSA选择 → Softmax |
| GQA + SWA | Mistral全系列, Gemma 3 | GQA投影 → SWA mask → Softmax |
| GQA + SWA + Full | GPT-OSS, Step 3.5 | GQA投影 → 5:1 SWA/Full交替 |
| Gated DeltaNet + Gated Attention | Qwen3-Next | 3:1 GatedDeltaNet/GatedAttn交替 |
| MLA + Linear | Kimi Lightning | MLA投影 → Linear Attention |
| MLA + DSA + SWA | 未来可能 | MLA → DSA → SWA链式 |

**核心规则**：Hybrid在mermaid图中**按数据流顺序串联**各组件节点。

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
attention_type: <standard|mqa|gqa|mla|swa|dsa|linear|gated_deltanet|gated_attention|...>

# Hybrid（2个及以上类型组合）：列表声明
# attention_hybrid_types:
#   - <type1>
#   - <type2>
attention_hybrid_types:
  - mla
  - dsa

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

# Hybrid交替模式（如果层类型交替）
layer_types_key: <config-key>      # e.g., layer_types
hybrid_ratio: <string>             # e.g., "5:1" (SWA:Full)

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

1. **单attention类型**：直接展开投影结构
2. **MLA**：展开Q_A_Proj → Q_RMSNorm → Q_B_Proj链；KV_A_Proj → KV_RMSNorm → KV_B_Proj链
3. **DSA**：在MLA之后增加sparse selection节点（indices → block_select → Softmax）
4. **SWA**：在Softmax前增加滑动窗口mask节点
5. **混合**：按数据流顺序串联各组件

### 数据流规则

```
attention输入
  → [Q_proj / MLA_Q_chain]
  → [K_proj / MLA_K_chain]
  → [V_proj / MLA_V_chain]
  → [RoPE] (if using rotary)
  → [SWA_mask] (if sliding_window)
  → [DSA_sparse_select] (if dsa, block-indexed top-K)
  → [Softmax]
  → O_proj
  → attention输出
```

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

#### 3. MLA + DSA (DeepSeek V3.2, GLM-5)

```mermaid
%% MLA投影链
Q_A["Q_A_Proj"]:::attention --> Q_LN["Q RMSNorm"]:::attention
Q_LN --> Q_B["Q_B_Proj"]:::attention
Q_B --> ConcatQ["Concat(Q_nope + Q_rope)"]:::attention

KV_A["KV_A_Proj"]:::attention --> KV_LN["KV RMSNorm"]:::attention
KV_LN --> ConcatK["Concat(K_rope + K_nope)"]:::attention
KV_LN --> K_B["K_B_Proj"]:::attention
ConcatK --> RoPE["RoPE"]:::attention

%% DSA稀疏选择（在Softmax前）
RoPE --> DSA_Indices["Block Indices<br/>[B,S,topk]"]:::attention
ConcatQ --> DSA_Indices
DSA_Indices --> DSA_Select["Sparse Select<br/>top-K KV by block"]:::attention
K_B --> DSA_Select
DSA_Select --> Softmax["Softmax(Q·Kᵀ/√d)"]:::attention
Softmax --> O_Proj["O_Proj"]:::attention
```

#### 4. GQA + SWA (Mistral, Gemma 3)

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

#### 5. Hybrid Layer Alternation (GPT-OSS, MiMo, Qwen3-Next)

当 `attention_hybrid_types` 包含层交替模式时（如 `[gqa, swa]` 或 `[gated_deltanet, gated_attention]`）：

**层交替逻辑：**
- `layer_types_key`（如 `layer_types`）声明每层的类型数组，如 `["sliding_attention", "full_attention", ...]`
- `hybrid_ratio`（如 `"5:1"`）声明交替比例，用于验证和文档说明
- mermaid图中用stack pattern描述：`[swa_block] × 5, [global_block] × 1, ...`

```mermaid
%% 混合交替模式：GPT-OSS (1:1)、MiMo (5:1)
subgraph Transformer_Stack ["[SWA] × N, [Full] × 1, ... (交替)"]
  direction TB
  SWA_Block["SWA Block<br/>GQA + SWA"]:::attention
  Full_Block["Full Block<br/>GQA + Full"]:::attention
  SWA_Block --> Full_Block
end
```

**注意**：混合交替的展开是**层级式**的，不是链式串联（不像MLA→DSA是单层内的数据流串联）。

---

## 需要修改的文件清单

### Templates（逐个模型显式声明attention类型）

| 文件 | 修改内容 |
|---|---|
| `templates/deepseek/common.yaml` | 添加 `attention_type: mla` 默认值 |
| `templates/deepseek/deepseek-v3.yaml` | `attention_type: mla` |
| `templates/deepseek/deepseek-v3.1.yaml` | `attention_type: mla` |
| `templates/deepseek/deepseek-v3.2.yaml` | `attention_hybrid_types: [mla, dsa]` + dsa参数 |
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
| `templates/glm/glm-5.yaml` | `attention_hybrid_types: [mla, dsa]` + DSA参数 |
| `templates/baichuan/common.yaml` | 添加attention类型注释说明 |
| `templates/baichuan/baichuan-13b.yaml` | `attention_type: standard` (MHA) |
| `templates/baichuan/baichuan2-7b.yaml` | `attention_type: standard` (MHA) |
| `templates/baichuan/baichuan-m3.yaml` | `attention_type: gqa` |
| `templates/mistral/common.yaml` | 添加 `attention_type: gqa`, `attention_hybrid_types` 默认值 |
| `templates/mistral/mistral-7b-v0.1.yaml` | `attention_hybrid_types: [gqa, swa]` |
| `templates/mistral/mistral-7b-v0.3.yaml` | `attention_hybrid_types: [gqa, swa]` |
| `templates/mistral/mistral-nemo.yaml` | `attention_hybrid_types: [gqa, swa]` |
| `templates/mistral/mistral-large.yaml` | `attention_hybrid_types: [gqa, swa]` |
| `templates/minimax/common.yaml` | 添加 `attention_type: gqa` 默认值 |
| `templates/minimax/minimax-m2.yaml` | `attention_type: gqa` |
| `templates/minimax/minimax-m2.5.yaml` | `attention_type: gqa` |
| `templates/mimo/common.yaml` | 添加 `attention_type: gqa`, `attention_hybrid_types` 默认值 |
| `templates/mimo/mimo-v2-flash.yaml` | `attention_hybrid_types: [gqa, swa]`, `hybrid_ratio: "5:1"` |
| `templates/mimo/mimo-vl-7b.yaml` | `attention_hybrid_types: [gqa, swa]` |
| `templates/gpt-oss/common.yaml` | 添加 `attention_type: gqa`, `attention_hybrid_types` 默认值 |
| `templates/gpt-oss/gpt-oss-120b.yaml` | `attention_hybrid_types: [gqa, swa, full]`, `hybrid_ratio: "1:1"` |
| `templates/gpt-oss/gpt-oss-20b.yaml` | `attention_hybrid_types: [gqa, swa, full]`, `hybrid_ratio: "1:1"` |

### SKILL.md变更

在mermaid生成逻辑中增加attention类型判断分支：

```
Step 4生成逻辑：
if attention_hybrid_types 存在（2个及以上）:
    # 按顺序串联各类型的展开结构
    按列表顺序串联展开
elif attention_type == "mla":
    展开MLA链
elif attention_type == "gqa":
    展开GQA链
elif attention_type == "standard":
    展开标准MHA链
elif attention_type == "swa":
    展开SWA链
elif attention_type == "dsa":
    展开DSA链
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
