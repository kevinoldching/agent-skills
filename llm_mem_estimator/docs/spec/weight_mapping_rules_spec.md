# 权重分类规则配置说明

本文档说明 `configs/weight_mapping_rules.yaml` 的格式和使用方法，用于将不同模型的权重名称映射到标准的模块类型。

## 概述

权重分类规则用于自动识别模型权重并将其分类到不同的模块类型（如 embedding、attention、ffn_moe 等）。这个配置文件支持：

- 通用规则（适用于大多数 Transformer 模型）
- 模型特定规则（如 DeepSeek、GPT-OSS、Llama、Qwen 等）
- 规则继承（避免重复定义）
- 正则表达式模式匹配
- 排除规则（exclude）

---

## 文件结构

```yaml
# 通用规则
generic:
  <module_type>:
    patterns: [...]
    exclude: [...]

# 模型特定规则
<model_type>:
  inherit: <parent_rule_set>  # 可选：继承其他规则集
  <module_type>:
    patterns: [...]
    exclude: [...]
```

---

## 架构配置字段映射（Architecture Config Field Mappings）

### 概述

`architecture_config` 部分定义了如何将不同模型的 `config.json` 字段映射到标准的字段名。因为不同模型可能使用不同的字段名来表示相同的参数（例如 `num_hidden_layers` vs `n_layer`），这个映射机制可以自动识别并转换。

### 字段说明

| 标准字段名 | 可能使用的字段名 | 说明 |
|-----------|-----------------|------|
| `hidden_size` | `hidden_size` | 隐藏层维度 |
| `head_dim` | `head_dim` | 注意力头维度 |
| `num_layers` | `num_hidden_layers`, `n_layer`, `num_layers` | Transformer 层数 |
| `vocab_size` | `vocab_size` | 词表大小 |
| `num_attention_heads` | `num_attention_heads`, `n_heads` | 注意力头数 |
| `num_key_value_heads` | `num_key_value_heads`, `num_kv_heads`, `n_kv_heads` | KV 头数 |
| `intermediate_size` | `intermediate_size`, `ffn_hidden_size` | FFN 中间层维度 |
| `num_experts` | `num_experts`, `n_routed_experts`, `num_local_experts` | MoE 专家总数 |
| `num_experts_per_tok` | `num_experts_per_tok`, `top_k`, `moe_top_k` | 每个 token 激活的专家数 |
| `moe_intermediate_size` | `moe_intermediate_size`, `moe_ffn_hidden_size` | MoE 专家中间层维度 |
| `q_lora_rank` | `q_lora_rank`, `q_rank` | MLA Q LoRA 秩 |
| `kv_lora_rank` | `kv_lora_rank`, `kv_rank` | MLA KV LoRA 秩 |
| `v_head_dim` | `v_head_dim`, `v_head_ratio` | V 头维度 |
| `qk_rope_head_dim` | `qk_rope_head_dim`, `rope_head_dim` | RoPE 头维度 |
| `qk_nope_head_dim` | `qk_nope_head_dim` | 非 RoPE 头维度 |

### 自动计算 head_dim

如果模型 config 中没有直接提供 `head_dim`，系统会自动计算：
```
head_dim = hidden_size / num_attention_heads
```

### 示例

```yaml
architecture_config:
  field_mappings:
    hidden_size:
      - "hidden_size"
    num_layers:
      - "num_hidden_layers"
      - "n_layer"
    num_attention_heads:
      - "num_attention_heads"
      - "n_heads"
    num_key_value_heads:
      - "num_key_value_heads"
      - "num_kv_heads"
    num_experts:
      - "num_experts"
      - "n_routed_experts"
```

---

## 计算规则（Computation Rules）

### 概述

`computation_rules` 定义了 KV Cache 和 Activation 的显存计算公式，以及系统预留显存和 GPU 利用率配置。

### 字段说明

| 字段 | 类型 | 是否必填 | 说明 |
|------|------|---------|------|
| `recommended_capacity_factor` | float 或 dict | 否 | 激活值计算的容量因子。<br>旧格式：单一浮点数，默认 1.25<br>新格式：字典 `{has_prefill: 1.25, decode: 12.5}` |
| `system_reserved_gb` | float | 否 | 系统预留显存（GB），默认 2.0 |
| `gpu_util` | float | 否 | GPU 利用率（0-1.0），默认 1.0。<br>可用显存 = 实际显存 × gpu_util |
| `kv_cache` | string | 是 | KV Cache 显存计算公式 |
| `activation` | string | 是 | Activation 显存计算公式 |

### recommended_capacity_factor 说明

- **has_prefill**: Prefill 阶段或混部场景使用（默认 1.25）
- **decode**: 纯 Decode 阶段使用（默认 12.5）

### 使用场景

| 场景 | factor |
|------|--------|
| 搜索最大 gen_len（Decode） | decode |
| 搜索最大 prompt_len（Prefill） | has_prefill |

### 示例

```yaml
computation_rules:
  recommended_capacity_factor:
    has_prefill: 1.25
    decode: 12.5
  system_reserved_gb: 2.0
  gpu_util: 0.9
  kv_cache: "2 * batch_size * seq_len * num_key_value_heads * head_dim * num_layers / (tp_size * cp_size)"
  activation: "batch_size * seq_len * hidden_size * num_experts_per_tok * recommended_capacity_factor / cp_size"
```

---

## 模块类型

支持的模块类型（`<module_type>`）：

| 模块类型 | 说明 | 典型权重 |
|---------|------|---------|
| `embedding` | 词嵌入和输出层 | `embed_tokens.weight`, `lm_head.weight` |
| `attention` | 注意力模块 | `self_attn.q_proj.weight`, `self_attn.k_proj.weight` |
| `ffn_moe` | MoE FFN（包含路由和专家） | `mlp.router.weight`, `mlp.experts.*` |
| `ffn_shared_expert` | MoE 共享专家 | `mlp.shared_experts.*` |
| `ffn_dense` | 标准稠密 FFN | `mlp.gate_proj.weight`, `mlp.up_proj.weight` |
| `norm` | 归一化层 | `input_layernorm.weight`, `model.norm.weight` |
| `others` | 其他组件 | 任何不属于上述类别的权重 |

---

## TP 变体配置（TP Variants）

### 概述

`tp_variants` 用于定义可引用的 TP（Tensor Parallel）变体及其默认大小。这些变体在 `parallel_defaults` 中被引用，允许对不同权重组使用不同的 TP size。

### 适用场景

当需要对不同权重组使用不同的 TP 并行度时使用，例如：
- `o_proj.weight` 使用 TP=8
- `gate_proj.weight` 使用 TP=2
- `shared_experts` 使用 TP=4

### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `tp_variants` | dict[string, int] | 否 | 变体名称到 TP size 的映射 |

### 变体命名规范

变体名称格式：`TP_XXX`，其中 `XXX` 表示该变体应用的权重组。目前支持的变体：

| 变体名称 | 默认 TP Size | 说明 |
|---------|-------------|------|
| `TP_O_PROJ` | 8 | o_proj.weight 的 TP size |
| `TP_MLP` | 8 | FFN 权重的 TP size |
| `TP_SHARED_EXPERT` | 8 | 共享专家的 TP size |
| `TP_EMBEDDING` | 1 | embedding 的 TP size |

### CLI 覆盖

可通过 CLI 参数覆盖 YAML 中的默认值：

```bash
--tp-o-proj 4          # 覆盖 TP_O_PROJ 的默认值
--tp-mlp 2             # 覆盖 TP_MLP 的默认值
--tp-shared-expert 4   # 覆盖 TP_SHARED_EXPERT 的默认值
--tp-embedding 2       # 覆盖 TP_EMBEDDING 的默认值
```

### 示例

```yaml
generic:
  # 定义 TP 变体及其默认值
  tp_variants:
    TP_O_PROJ: 8
    TP_MLP: 8
    TP_SHARED_EXPERT: 8
    TP_EMBEDDING: 1

  # 在 parallel_defaults 中引用变体
  parallel_defaults:
    embedding: TP_EMBEDDING
    attention:
      o_proj.weight: TP_O_PROJ
    ffn_dense: TP_MLP
    ffn_shared_expert: TP_SHARED_EXPERT
```

---

## 并行策略配置（parallel_defaults）

### 概述

`parallel_defaults` 定义了不同模块类型的默认并行策略，用于在 YAML 配置中指定权重如何被切分到不同的并行维度。

### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `<module_type>` | string 或 dict | 是 | 模块的并行策略 |

### 并行策略类型

| 策略 | 说明 |
|------|------|
| `TP` | Tensor Parallel，切分到 `tp` 维度 |
| `EP` | Expert Parallel，切分到 `ep` 维度 |
| `PP` | Pipeline Parallel，切分到 `pp` 维度 |
| `DP` | Data Parallel，切分到 `dp` 维度 |
| `replicated` | 完全复制，不切分 |
| `TP_XXX` | 使用 `tp_variants` 中定义的 `TP_XXX` 变体的 TP size |

### 子键匹配

对于 `attention` 和 `ffn_moe` 等模块类型，`parallel_defaults` 支持按子键（权重名）指定不同的策略：

```yaml
parallel_defaults:
  attention:
    q_proj.weight: TP
    o_proj.weight: TP_O_PROJ  # 使用变体指定的 TP size
  ffn_moe:
    experts: EP
    gate_proj.weight: TP_MLP  # 使用变体指定的 TP size
```

### PD 分离支持

`parallel_defaults` 支持三层结构：

- `parallel_defaults`：混部/通用场景（stage=hybrid）
- `parallel_defaults.prefill`：PD分离-prefill专用
- `parallel_defaults.decode`：PD分离-decode专用

继承规则：模型定义了某 stage → 使用模型的；模型未定义 → 使用 generic 的同名 stage。

---

## 规则定义

### 基本格式

```yaml
<model_type>:
  <module_type>:
    patterns:
      - "正则表达式模式1"
      - "正则表达式模式2"
    exclude:  # 可选
      - "排除模式1"
      - "排除模式2"
```

### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `patterns` | list[string] | 是 | 正则表达式模式列表，用于匹配权重名称 |
| `exclude` | list[string] | 否 | 排除模式列表，匹配的权重会被排除 |

### 模式匹配规则

1. **正则表达式格式**：模式使用 Python 正则表达式语法
2. **自动添加锚点**：模式会自动添加 `^` 和 `$` 锚点（完全匹配）
3. **匹配顺序**：按照模块类型的定义顺序进行匹配
4. **首次匹配**：权重名称匹配到第一个模块类型后，不再继续匹配
5. **排除优先**：如果权重匹配了 `exclude` 模式，则不会被分类到该模块类型

### 正则表达式语法

常用的正则表达式元字符：

| 元字符 | 说明 | 示例 |
|-------|------|------|
| `.` | 匹配任意单个字符 | `self_attn..*` 匹配 `self_attn.q_proj.weight` |
| `*` | 前面的字符重复0次或多次 | `.*` 匹配任意字符串 |
| `+` | 前面的字符重复1次或多次 | `.+` 匹配至少一个字符 |
| `\d` | 匹配数字 | `layers\.\d+` 匹配 `layers.0`, `layers.12` |
| `\\.` | 匹配字面点号 | `model\\.norm` 匹配 `model.norm` |
| `^` | 字符串开始 | `^embed` 匹配以 `embed` 开头 |
| `$` | 字符串结束 | `weight$` 匹配以 `weight` 结尾 |
| `\|` | 或运算符 | `(gate\|up)_proj` 匹配 `gate_proj` 或 `up_proj` |
| `()` | 分组 | `(gate_up_proj\|down_proj)` |
| `[]` | 字符类 | `[0-9]` 匹配任意数字 |

**注意**：在 YAML 中，反斜杠需要转义，所以 `\.` 要写成 `\\.`，`\d` 要写成 `\\d`。

---

## 规则继承

使用 `inherit` 字段可以继承其他规则集，避免重复定义。

### 语法

```yaml
<model_type>:
  inherit: <parent_rule_set>
  <module_type>:
    patterns: [...]
```

### 继承规则

1. 子规则集会继承父规则集的所有模块类型定义
2. 如果子规则集重新定义了某个模块类型，会**覆盖**父规则集的定义（不是合并）
3. 可以继承 `generic` 或其他模型特定规则集

### 示例

```yaml
# 通用规则
generic:
  embedding:
    patterns:
      - "^embed_tokens\\.weight$"
      - "^lm_head\\.weight$"

  attention:
    patterns:
      - "^self_attn\\..*"

# Llama 继承通用规则
llama:
  inherit: generic
  # 不需要重新定义 embedding 和 attention，自动继承

# GPT-OSS 继承通用规则，但覆盖 ffn_moe 定义
gpt_oss:
  inherit: generic

  ffn_moe:  # 覆盖父规则集的 ffn_moe 定义
    patterns:
      - "^mlp\\.router\\..*"
      - "^mlp\\.experts\\.(gate_up_proj|down_proj).*"
```

---

## 模式示例

### 1. Embedding 模块

```yaml
embedding:
  patterns:
    - "^embed_tokens\\.weight$"           # 精确匹配
    - "^model\\.embed_tokens\\.weight$"   # 带前缀的精确匹配
    - "^lm_head\\.weight$"                # 输出层
    - ".*wte.*"                           # GPT 风格（包含 wte）
    - ".*word_embeddings.*"               # BERT 风格
```

### 2. Attention 模块

```yaml
attention:
  patterns:
    - "^self_attn\\..*"                   # 以 self_attn. 开头的所有权重
    - ".*\\.self_attn\\..*"               # 任意前缀 + .self_attn. + 任意后缀
```

### 3. MoE FFN 模块

```yaml
ffn_moe:
  patterns:
    - "^mlp\\.gate\\..*"                  # 路由器（gate）
    - "^mlp\\.router\\..*"                # 路由器（router）
    - ".*\\.mlp\\.gate\\..*"              # 带前缀的路由器
    - ".*\\.mlp\\.router\\..*"            # 带前缀的路由器
    - "^mlp\\.experts\\.\\d+\\..*"        # 专家权重（带编号）
    - "^mlp\\.experts\\.(gate_up_proj|down_proj).*"  # 专家投影权重
    - ".*\\.mlp\\.experts\\..*"           # 任意专家权重
  exclude:
    - ".*\\.shared_experts\\..*"          # 排除共享专家
```

### 4. 标准 FFN 模块

```yaml
ffn_dense:
  patterns:
    - "^mlp\\.(gate_proj|up_proj|down_proj)\\..*"  # 标准 FFN 投影
    - ".*\\.mlp\\.(gate_proj|up_proj|down_proj)\\..*"
  exclude:
    - ".*\\.experts\\..*"                 # 排除专家权重
    - ".*\\.shared_experts\\..*"          # 排除共享专家
```

### 5. Norm 模块

```yaml
norm:
  patterns:
    - "^input_layernorm\\..*"             # 输入归一化
    - "^post_attention_layernorm\\..*"    # 注意力后归一化
    - ".*\\.input_layernorm\\..*"         # 带前缀的输入归一化
    - ".*\\.post_attention_layernorm\\..*"
    - "^model\\.norm\\.weight$"           # 最终归一化
    - "^norm\\.weight$"
```

---

## 完整示例

### 示例 1: 通用规则

```yaml
generic:
  embedding:
    patterns:
      - "^embed_tokens\\.weight$"
      - "^model\\.embed_tokens\\.weight$"
      - "^lm_head\\.weight$"
      - ".*wte.*"
      - ".*word_embeddings.*"

  attention:
    patterns:
      - "^self_attn\\..*"
      - ".*\\.self_attn\\..*"

  ffn_moe:
    patterns:
      - "^mlp\\.gate\\..*"
      - "^mlp\\.router\\..*"
      - ".*\\.mlp\\.gate\\..*"
      - ".*\\.mlp\\.router\\..*"
      - "^mlp\\.experts\\.\\d+\\..*"
      - "^mlp\\.experts\\.(gate_up_proj|down_proj).*"
      - ".*\\.mlp\\.experts\\..*"
    exclude:
      - ".*\\.shared_experts\\..*"

  ffn_shared_expert:
    patterns:
      - "^mlp\\.shared_experts\\..*"
      - ".*\\.mlp\\.shared_experts\\..*"

  ffn_dense:
    patterns:
      - "^mlp\\.(gate_proj|up_proj|down_proj)\\..*"
      - ".*\\.mlp\\.(gate_proj|up_proj|down_proj)\\..*"
    exclude:
      - ".*\\.experts\\..*"
      - ".*\\.shared_experts\\..*"

  norm:
    patterns:
      - "^input_layernorm\\..*"
      - "^post_attention_layernorm\\..*"
      - ".*\\.input_layernorm\\..*"
      - ".*\\.post_attention_layernorm\\..*"
      - "^model\\.norm\\.weight$"
      - "^norm\\.weight$"
```

### 示例 2: DeepSeek 特定规则

```yaml
deepseek:
  embedding:
    patterns:
      - "^embed_tokens\\.weight$"
      - "^model\\.embed_tokens\\.weight$"
      - "^lm_head\\.weight$"
      - "^shared_head\\.head\\.weight$"  # DeepSeek 特有

  attention:
    patterns:
      - "^self_attn\\..*"
      - ".*\\.self_attn\\..*"

  ffn_moe:
    patterns:
      - "^mlp\\.gate\\..*"
      - ".*\\.mlp\\.gate\\..*"
      - "^mlp\\.experts\\.\\d+\\..*"
      - ".*\\.mlp\\.experts\\.\\d+\\..*"
    exclude:
      - ".*\\.shared_experts\\..*"

  ffn_shared_expert:
    patterns:
      - "^mlp\\.shared_experts\\..*"
      - ".*\\.mlp\\.shared_experts\\..*"

  ffn_dense:
    patterns:
      - "^mlp\\.(gate_proj|up_proj|down_proj)\\..*"
      - ".*\\.mlp\\.(gate_proj|up_proj|down_proj)\\..*"
    exclude:
      - ".*\\.experts\\..*"
      - ".*\\.shared_experts\\..*"

  norm:
    patterns:
      - "^input_layernorm\\..*"
      - "^post_attention_layernorm\\..*"
      - ".*\\.input_layernorm\\..*"
      - ".*\\.post_attention_layernorm\\..*"
      - "^model\\.norm\\.weight$"
      - "^enorm\\.weight$"              # DeepSeek 特有
      - "^hnorm\\.weight$"              # DeepSeek 特有
      - "^shared_head\\.norm\\.weight$" # DeepSeek 特有

  others:
    patterns:
      - "^eh_proj\\.weight$"            # DeepSeek 特有
```

### 示例 3: 使用继承

```yaml
# Llama 系列使用通用规则
llama:
  inherit: generic

# Qwen 系列继承通用规则，但覆盖 attention 定义
qwen:
  inherit: generic

  attention:
    patterns:
      - "^self_attn\\..*"
      - ".*\\.c_attn\\..*"    # Qwen 特有
      - ".*\\.c_proj\\..*"    # Qwen 特有
```

---

## 如何自定义规则

### 步骤 1: 确定模型类型

首先确定你的模型类型（model_type），这通常可以从 HuggingFace 的 `config.json` 中的 `model_type` 字段获取。

### 步骤 2: 检查是否可以继承现有规则

如果你的模型与现有模型（如 Llama、GPT）结构相似，可以使用 `inherit` 继承现有规则。

### 步骤 3: 列出所有权重名称

使用以下命令查看模型的所有权重名称：

```bash
python -c "
from huggingface_hub import get_safetensors_metadata
metadata = get_safetensors_metadata('your-model-name')
for name in sorted(metadata.weight_map.keys()):
    print(name)
"
```

### 步骤 4: 编写正则表达式模式

根据权重名称的规律，编写正则表达式模式：

1. 找出共同的前缀或后缀
2. 识别层编号的位置（如 `layers.0`, `layers.1`）
3. 使用 `\d+` 匹配层编号
4. 使用 `.*` 匹配任意字符
5. 使用 `\\.` 匹配字面点号

### 步骤 5: 测试模式

使用 Python 测试你的正则表达式：

```python
import re

pattern = r"^mlp\.experts\.\d+\..*"
test_names = [
    "mlp.experts.0.gate_proj.weight",
    "mlp.experts.127.down_proj.weight",
    "mlp.shared_experts.gate_proj.weight",  # 应该不匹配
]

for name in test_names:
    if re.match(f"^{pattern}$", name):
        print(f"✓ {name}")
    else:
        print(f"✗ {name}")
```

### 步骤 6: 添加到配置文件

将规则添加到 `configs/weight_mapping_rules.yaml`：

```yaml
your_model_type:
  inherit: generic  # 如果适用

  # 覆盖或添加特定模块类型
  ffn_moe:
    patterns:
      - "你的正则表达式模式"
```

### 步骤 7: 验证

重新生成配置文件，检查权重是否被正确分类：

```bash
python scripts/calculate_mem.py --model your-model-name --generate-config
```

---

## 常见问题

### Q1: 为什么我的权重被分类到 `others`？

**A**: 可能的原因：
1. 没有为该模型类型定义规则（使用了 `generic` 但模式不匹配）
2. 正则表达式模式不正确
3. 权重被 `exclude` 规则排除了

**解决方法**：
- 检查权重名称是否匹配任何模式
- 使用 Python 测试正则表达式
- 检查是否有 `exclude` 规则排除了该权重

### Q2: 如何匹配带层编号的权重？

**A**: 使用 `\d+` 匹配层编号：

```yaml
patterns:
  - "^model\\.layers\\.\\d+\\.self_attn\\..*"  # 匹配 model.layers.0.self_attn.q_proj.weight
```

### Q3: 如何排除某些权重？

**A**: 使用 `exclude` 字段：

```yaml
ffn_dense:
  patterns:
    - "^mlp\\.(gate_proj|up_proj|down_proj)\\..*"
  exclude:
    - ".*\\.experts\\..*"  # 排除专家权重
```

### Q4: 继承规则后如何覆盖某个模块类型？

**A**: 直接重新定义该模块类型即可，会完全覆盖父规则集的定义：

```yaml
your_model:
  inherit: generic

  ffn_moe:  # 完全覆盖 generic 的 ffn_moe 定义
    patterns:
      - "你的新模式"
```

### Q5: 如何处理特殊字符（如点号）？

**A**: 在 YAML 中，反斜杠需要转义：
- 匹配点号：`\\.` （在 YAML 中写成 `\\.`）
- 匹配数字：`\\d` （在 YAML 中写成 `\\d`）
- 匹配反斜杠：`\\\\` （在 YAML 中写成 `\\\\`）

---

## 调试技巧

### 1. 打印权重分类结果

修改 `llm_mem_estimator/model_detector.py` 的 `_classify_weights` 方法，添加调试输出：

```python
print(f"Weight: {first_weight_name} -> Module: {module_type}")
```

### 2. 测试正则表达式

使用在线工具测试正则表达式：
- https://regex101.com/ （选择 Python 语法）
- https://regexr.com/

### 3. 查看所有权重名称

创建测试脚本查看模型的所有权重：

```python
from llm_mem_estimator.model_detector import ModelDetector

weights = ModelDetector.get_weights_metadata("your-model-name")
for name in sorted(weights.keys()):
    print(name)
```

---

## 相关文档

- `docs/yaml_config_spec.md`: YAML 配置文件格式说明
- `docs/plan.md`: 详细的实现计划和架构说明
- `CLAUDE.md`: 项目概览和开发指南
