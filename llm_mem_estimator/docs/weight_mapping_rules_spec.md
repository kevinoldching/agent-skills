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
