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

### 按投影结构分类

| attention_type | 说明 | KV Cache | 代表模型 |
|---|---|---|---|
| `standard` | 标准MHA，每Q头独立K/V | Full | 旧模型 |
| `mqa` | 所有Q头共享单一K/V头 | Minimal | 研究模型 |
| `gqa` | Q头分组，每组共享K/V | Medium | Llama3, Qwen3, Mistral |
| `mla` | K/V压缩到低秩潜空间 | Very small | DeepSeek V3, Kimi K2 |

### 按Token选择分类（可叠加）

| token_selection | 说明 | 与投影类型关系 |
|---|---|---|
| `dense` | 注意力稠密，所有token参与 | 默认 |
| `swa` (sliding_window) | 每token只attend到固定窗口 | 可叠加到任意投影类型 |
| `dsa` (deepseek_sparse) | Block索引稀疏选择top-K KV token | 可叠加到MLA |
| `hybrid_swa` | SWA层+全局注意力层交替 | 可叠加到任意投影类型 |

### 混合Attention组合

以下组合需要在mermaid图中展开连接关系：

| 组合 | 例子 | 展开内容 |
|---|---|---|
| `mla + swa` | Mistral Large 3 | MLA投影 → SWA窗口限制 |
| `mla + dsa` | DeepSeek V3.2, GLM-5 | MLA投影 → DSA稀疏选择 → Softmax |
| `gqa + swa` | Gemma 3, MiMo-V2-Flash | GQA投影 → 滑动窗口 |
| `gqa + swa + global` | GPT-OSS, Step 3.5 | GQA投影 → 5:1 SWA/Global交替 |
| `mla + dsa + swa` | 未来可能 | MLA → DSA → SWA链式 |

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

# === 必须显式声明 ===
attention_type: <standard|mqa|gqa|mla>   # 投影结构类型
token_selection: <dense|swa|dsa|hybrid_swa>  # token选择策略，可选

# 如果是MLA，声明MLA参数
mla:
  kv_lora_rank_key: <config-key>
  q_lora_rank_key: <config-key>
  qk_rope_head_dim_key: <config-key>
  qk_nope_head_dim_key: <config-key>
  v_head_dim_key: <config-key>

# 如果是SWA，声明窗口大小
sliding_window_key: <config-key>   # e.g., sliding_window

# 如果是DSA，声明DSA参数
dsa:
  sparse_topk_key: <config-key>     # top-K tokens selected
  block_size_key: <config-key>      # block indexing granularity

# 如果是GQA/MQA，声明kv_heads
kv_heads_key: <config-key>         # e.g., num_key_value_heads

# 如果是混合SWA（交替层类型）
layer_types_key: <config-key>      # e.g., layer_types
hybrid_ratio: <string>             # e.g., "5:1" (SWA:Global)

template: <family>/<model-name>.yaml   # 指向自身（显式声明，不继承）
```

### common.yaml的role变更

common.yaml仅作为**默认值**，每个模型文件必须override自己的attention类型：

```yaml
# common.yaml — 仅提供默认值，不被模型文件隐式依赖
model_type: <family-model-type>
family: <family-name>

# 默认值（可被模型文件override）
attention_type: gqa  # 家族默认
token_selection: dense

# 默认MLA参数（可被override）
mla: {...}

# 默认MoE参数（可被override）
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

#### 5. Hybrid SWA + Global (GPT-OSS, MiMo)

```mermaid
%% 混合模式：交替SWA层和全局层
subgraph Hybrid_Attention ["Hybrid SWA + Global (5:1)"]
  direction TB
  SWA_Layer["SWA Layer<br/>window=128"]:::attention
  Global_Layer["Global Layer<br/>full attention"]:::attention
  SWA_Layer --> Global_Layer
end
```

#### 6. GQA + SWA + DSA（未来扩展）

```mermaid
%% Q/K/V投影
Q_proj --> RoPE --> SWA_Mask --> DSA_Select --> Softmax --> O_proj
K_proj --> RoPE --> SWA_Mask --> DSA_Select
V_proj --> SWA_Mask --> DSA_Select
```

---

## 需要修改的文件清单

### Templates（逐个模型显式声明attention类型）

| 文件 | 修改内容 |
|---|---|
| `templates/deepseek/common.yaml` | 保留MLA默认值，添加dsa section |
| `templates/deepseek/deepseek-v3.yaml` | `attention_type: mla`, `token_selection: dense` |
| `templates/deepseek/deepseek-v3.1.yaml` | `attention_type: mla`, `token_selection: dense` |
| `templates/deepseek/deepseek-v3.2.yaml` | `attention_type: mla`, `token_selection: dsa` + dsa参数 |
| `templates/deepseek/deepseek-v2.yaml` | `attention_type: mla`, `token_selection: dense` |
| `templates/deepseek/deepseek-v2.5.yaml` | `attention_type: mla`, `token_selection: dense` |
| `templates/deepseek/deepseek-r1.yaml` | `attention_type: mla`, `token_selection: dense` |
| `templates/kimi/common.yaml` | 保留MLA默认值 |
| `templates/kimi/kimi-k2.0.yaml` | `attention_type: mla`, `token_selection: dense` |
| `templates/kimi/kimi-k2.5.yaml` | `attention_type: mla`, `token_selection: dense` |
| `templates/qwen/common.yaml` | 保留gqa默认值，添加swa说明 |
| `templates/qwen/qwen3-32b.yaml` | `attention_type: gqa`, `token_selection: dense` |
| `templates/qwen/qwen3-8b.yaml` | `attention_type: gqa`, `token_selection: dense` |
| `templates/qwen/qwen3-4b.yaml` | `attention_type: gqa`, `token_selection: dense` |
| `templates/qwen/qwen3-1.7b.yaml` | `attention_type: gqa`, `token_selection: dense` |
| `templates/qwen/qwen3-0.6b.yaml` | `attention_type: gqa`, `token_selection: dense` |
| `templates/qwen/qwen3-vl-8b.yaml` | `attention_type: gqa`, `token_selection: dense` |
| `templates/qwen/qwen3.5-397b-a17b.yaml` | `attention_type: gqa`, `token_selection: dense` |
| `templates/qwen/qwen3.5-122b-a10b.yaml` | `attention_type: gqa`, `token_selection: dense` |
| `templates/qwen/qwen3.5-35b-a3b.yaml` | `attention_type: gqa`, `token_selection: dense` |
| `templates/qwen/qwen3.5-27b.yaml` | `attention_type: gqa`, `token_selection: dense` |
| `templates/qwen/qwen2.5-vl-7b.yaml` | `attention_type: gqa`, `token_selection: dense` |
| `templates/llama/common.yaml` | 保留默认值 |
| `templates/llama/llama-2-7b.yaml` | `attention_type: standard` (MHA) |
| `templates/llama/llama-3-8b.yaml` | `attention_type: gqa` |
| `templates/llama/llama-3.1-8b.yaml` | `attention_type: gqa` |
| `templates/llama/llama-3.2-1b.yaml` | `attention_type: gqa` |
| `templates/llama/llama-3.2-3b.yaml` | `attention_type: gqa` |
| `templates/llama/llama-3.2-11b.yaml` | `attention_type: gqa` |
| `templates/llama/llama-3.3-70b.yaml` | `attention_type: gqa` |
| `templates/glm/common.yaml` | 添加 `attention_type: gqa` 声明 |
| `templates/glm/glm-4-9b.yaml` | `attention_type: gqa` |
| `templates/glm/glm-4-28b.yaml` | `attention_type: gqa` |
| `templates/glm/glm-4v.yaml` | `attention_type: gqa` |
| `templates/glm/glm-5.yaml` | `attention_type: mla`, `token_selection: dsa` + DSA参数 |
| `templates/baichuan/common.yaml` | 添加attention类型注释说明 |
| `templates/baichuan/baichuan-13b.yaml` | `attention_type: standard` (MHA) |
| `templates/baichuan/baichuan2-7b.yaml` | `attention_type: standard` (MHA) |
| `templates/baichuan/baichuan-m3.yaml` | `attention_type: gqa` (基于Qwen3) |
| `templates/mistral/common.yaml` | 保留gqa+swa默认值 |
| `templates/mistral/mistral-7b-v0.1.yaml` | `attention_type: gqa`, `token_selection: swa` |
| `templates/mistral/mistral-7b-v0.3.yaml` | `attention_type: gqa`, `token_selection: swa` |
| `templates/mistral/mistral-nemo.yaml` | `attention_type: gqa`, `token_selection: swa` |
| `templates/mistral/mistral-large.yaml` | `attention_type: gqa`, `token_selection: swa` |
| `templates/minimax/common.yaml` | 保留gqa默认值 |
| `templates/minimax/minimax-m2.yaml` | `attention_type: gqa` |
| `templates/minimax/minimax-m2.5.yaml` | `attention_type: gqa` |
| `templates/mimo/common.yaml` | 保留gqa+swa默认值 |
| `templates/mimo/mimo-v2-flash.yaml` | `attention_type: gqa`, `token_selection: hybrid_swa`, `hybrid_ratio: "5:1"` |
| `templates/mimo/mimo-vl-7b.yaml` | `attention_type: gqa`, `token_selection: swa` |
| `templates/gpt-oss/common.yaml` | 保留hybrid_swa默认值 |
| `templates/gpt-oss/gpt-oss-120b.yaml` | `attention_type: gqa`, `token_selection: hybrid_swa`, `hybrid_ratio: "1:1"` |
| `templates/gpt-oss/gpt-oss-20b.yaml` | `attention_type: gqa`, `token_selection: hybrid_swa`, `hybrid_ratio: "1:1"` |

### SKILL.md变更

在mermaid生成逻辑中增加attention类型判断分支：

```
Step 4生成逻辑：
if attention_type == "mla":
    if token_selection == "dsa":
        展开MLA + DSA连接
    elif token_selection == "swa":
        展开MLA + SWA连接
    else:
        展开纯MLA
elif attention_type == "gqa":
    if token_selection == "swa":
        展开GQA + SWA
    elif token_selection == "hybrid_swa":
        展开Hybrid交替模式
    else:
        展开标准GQA
elif attention_type == "standard":
    展开标准MHA
```

---

## 实现顺序

1. **更新所有template YAML文件** — 显式声明attention类型
2. **更新SKILL.md mermaid生成逻辑** — 增加混合attention判断
3. **验证现有mermaid图** — 重新生成DeepSeek V3.2等关键模型的图确认正确
4. **测试完整流程** — 用新模板生成各模型架构图
