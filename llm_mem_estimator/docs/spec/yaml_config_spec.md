# YAML 配置文件格式说明

本文档说明 LLM 显存估算器的 YAML 配置文件格式，包括哪些字段是固定的，哪些字段需要根据模型自定义。

## 配置文件结构

YAML 配置文件包含四个主要部分：

```yaml
model_identity:      # 模型标识信息
architecture_config: # 架构参数
modules:            # 模块权重定义
computation_rules:  # 显存计算规则
```

---

## 1. model_identity（模型标识）

**说明**：模型的基本标识信息，用于识别和描述模型。

**字段说明**：

| 字段 | 类型 | 是否必填 | 说明 | 示例 |
|------|------|---------|------|------|
| name | string | 是 | 模型名称 | `gpt-oss-120b` |
| total_params | string/int | 是 | 总参数量 | `120000000000` 或 `"120B"` |
| num_layers | int | 是 | Transformer 层数 | `36` |
| quantization | string | 否 | 量化方法 | `mxfp4`, `int8`, `null` |

**示例**：
```yaml
model_identity:
  name: gpt-oss-120b
  total_params: "120000000000"
  num_layers: 36
  quantization: mxfp4
```

---

## 2. architecture_config（架构配置）

**说明**：模型的架构参数，用于计算 KV Cache 和激活值显存。

**字段说明**：

| 字段 | 类型 | 是否必填 | 说明 | 示例 |
|------|------|---------|------|------|
| hidden_size | int | 是 | 隐藏层维度 | `2880` |
| num_layers | int | 是 | Transformer 层数 | `36` |
| attention_type | string | 是 | 注意力类型 | `mha`, `gqa`, `mla`, `mqa` |
| ffn_type | string | 是 | FFN 类型 | `dense`, `moe` |
| norm_type | string | 是 | 归一化类型 | `layernorm`, `rmsnorm` |
| vocab_size | int | 是 | 词表大小 | `201088` |
| num_attention_heads | int | 否 | 注意力头数 | `64` |
| num_key_value_heads | int | 否 | KV 头数（GQA/MQA） | `8` |
| intermediate_size | int | 否 | FFN 中间层维度 | `11520` |
| num_experts | int | 否 | MoE 专家总数 | `128` |
| num_experts_per_tok | int | 否 | 每个 token 激活的专家数 | `4` |
| moe_intermediate_size | int | 否 | MoE 专家的中间层维度 | `2880` |
| q_lora_rank | int | 否 | MLA 的 Q LoRA 秩 | `1536` |
| kv_lora_rank | int | 否 | MLA 的 KV LoRA 秩 | `512` |
| qk_rope_head_dim | int | 否 | RoPE 部分的 head 维度 | `64` |
| v_head_dim | int | 否 | V 的 head 维度 | `128` |
| qk_nope_head_dim | int | 否 | 非 RoPE 部分的 head 维度 | `128` |
| window_size | int | 否 | 滑动窗口大小（SWA） | `512` |

**注意**：
- 不同的 `attention_type` 需要不同的参数字段（详见 plan.md）
- MoE 模型需要提供 `num_experts` 和 `num_experts_per_tok`
- MLA 架构需要提供 `q_lora_rank` 和 `kv_lora_rank`

**示例**：
```yaml
architecture_config:
  hidden_size: 2880
  num_layers: 36
  attention_type: gqa
  ffn_type: moe
  norm_type: rmsnorm
  vocab_size: 201088
  num_attention_heads: 64
  num_key_value_heads: 8
  intermediate_size: 2880
  num_experts_per_tok: 4
```

---

## 3. modules（模块权重定义）

**说明**：定义模型的各个模块及其权重信息。这是配置文件中最重要的部分。

### 3.1 模块类型

支持的模块类型：

| 模块类型 | 说明 | 示例权重 |
|---------|------|---------|
| `embedding` | 词嵌入和输出层 | `embed_tokens.weight`, `lm_head.weight` |
| `attention` | 注意力模块 | `self_attn.q_proj.weight`, `self_attn.k_proj.weight` |
| `ffn_moe` | MoE FFN（包含路由和专家） | `mlp.router.weight`, `mlp.experts.gate_up_proj` |
| `ffn_shared_expert` | MoE 共享专家 | `mlp.shared_experts.gate_proj.weight` |
| `ffn_dense` | 标准 FFN | `mlp.gate_proj.weight`, `mlp.up_proj.weight` |
| `norm` | 归一化层 | `input_layernorm.weight`, `model.norm.weight` |
| `others` | 其他组件 | 任何不属于上述类别的权重 |

### 3.2 权重信息字段

每个权重条目包含以下字段：

| 字段 | 类型 | 是否必填 | 说明 | 默认值 | 示例 |
|------|------|---------|------|--------|------|
| **shape** | list[int] | **是** | 权重张量的形状 | - | `[2880, 4096]` |
| **dtype** | string | **是** | 数据类型 | - | `BF16`, `FP16`, `U8`, `F8_E4M3` |
| **layers** | int | **是** | 该权重在多少层中出现 | - | `36`（每层都有）, `1`（全局共享） |
| parallel_strategy | string | 否 | 并行策略 | `replicated` | `tp_col`, `tp_row`, `expert_sharded` |
| world_size | int | 否 | 并行世界大小 | `0` | `8`（8 卡并行） |

**固定字段**（必须提供）：
- `shape`: 权重的形状，必须准确
- `dtype`: 数据类型，必须准确
- `layers`: 该权重出现的层数

**可选字段**（有默认值）：
- `parallel_strategy`: 默认为 `replicated`（全复制）
- `world_size`: 默认为 `0`（表示使用全局并行配置）

### 3.3 数据类型支持

| 数据类型 | 别名 | 字节数 | 说明 |
|---------|------|--------|------|
| FP32 | float32, f32 | 4 | 32 位浮点 |
| FP16 | float16 | 2 | 16 位浮点 |
| BF16 | bfloat16 | 2 | Brain Float 16 |
| FP8 | float8, f8_e4m3 | 1 | 8 位浮点 |
| INT8 | int8 | 1 | 8 位整数 |
| U8 | uint8 | 1 | 8 位无符号整数 |
| INT4 | int4 | 0.5 | 4 位整数 |
| U4 | uint4, u4 | 0.5 | 4 位无符号整数 |

### 3.4 并行策略

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| `replicated` | 全复制，不分片 | 小权重、归一化层 |
| `tp_col` | Tensor Parallel 列切分 | Q/K/V 投影、FFN 的 gate/up 投影 |
| `tp_row` | Tensor Parallel 行切分 | 注意力输出投影、FFN 的 down 投影 |
| `expert_sharded` | Expert Parallel 专家切分 | MoE 专家权重 |
| `pp` | Pipeline Parallel 层切分 | 跨层分布 |

### 3.5 权重命名规则

**重要**：权重名称是**模型特定的**，需要根据实际模型的权重名称来定义。

**命名约定**：
- 对于每层都有的权重，使用简化名称（去掉 `model.layers.N.` 前缀）
- 对于全局共享的权重，使用完整名称

**示例**：

```yaml
modules:
  # Embedding 模块
  embedding:
    model.embed_tokens.weight: {shape: [201088, 2880], dtype: BF16, layers: 1}
    lm_head.weight: {shape: [201088, 2880], dtype: BF16, layers: 1}

  # Attention 模块（每层都有，使用简化名称）
  attention:
    self_attn.q_proj.weight: {shape: [4096, 2880], dtype: BF16, layers: 36}
    self_attn.k_proj.weight: {shape: [512, 2880], dtype: BF16, layers: 36}
    self_attn.v_proj.weight: {shape: [512, 2880], dtype: BF16, layers: 36}
    self_attn.o_proj.weight: {shape: [2880, 4096], dtype: BF16, layers: 36}

  # MoE FFN 模块（包含路由和专家）
  ffn_moe:
    mlp.router.weight: {shape: [128, 2880], dtype: BF16, layers: 36}
    mlp.experts.gate_up_proj_blocks: {shape: [128, 5760, 90, 16], dtype: U8, layers: 36}
    mlp.experts.down_proj_blocks: {shape: [128, 2880, 90, 16], dtype: U8, layers: 36}

  # Norm 模块
  norm:
    input_layernorm.weight: {shape: [2880], dtype: BF16, layers: 36}
    post_attention_layernorm.weight: {shape: [2880], dtype: BF16, layers: 36}
    model.norm.weight: {shape: [2880], dtype: BF16, layers: 1}
```

---

## 4. computation_rules（计算规则）

**说明**：定义 KV Cache 和激活值的显存计算公式。

**字段说明**：

| 字段 | 类型 | 是否必填 | 说明 |
|------|------|---------|------|
| recommended_capacity_factor | float 或 dict | 否 | 激活值计算的容量因子。<br>旧格式：单一浮点数，默认 1.25<br>新格式：字典 `{has_prefill: 1.25, decode: 12.5}`<br>- `has_prefill`: Prefill 阶段或混部场景使用（默认 1.25）<br>- `decode`: 纯 Decode 阶段使用（默认 12.5） |
| kv_cache | string | 是 | KV Cache 显存计算公式 |
| activation | string | 是 | 激活值显存计算公式 |

**公式变量**（在公式中使用的占位符）：
- `batch_size`: 批大小（用户运行时传入）
- `prompt_len`: 输入提示词长度（用户运行时传入）
- `gen_len`: 生成输出长度（用户运行时传入）
- `seq_len`: 总序列长度 = prompt_len + gen_len（用户运行时传入）
- `hidden_size`: 隐藏层维度（从 architecture_config 获取）
- `num_layers`: 层数（从 architecture_config 获取）
- `dtype_bytes`: 数据类型字节数（根据数据类型自动计算）
- `num_experts_per_tok`: MoE 激活专家数（从 architecture_config 获取）
- 其他架构参数（如 `kv_lora_rank`）

**公式设计原则**：
- **KV Cache**：需要存储完整的上下文，使用 `(prompt_len + gen_len)` 或 `seq_len`
- **Activation**：只需要考虑生成的 token，使用 `gen_len`

**支持的函数**：

公式中可以使用以下内置函数：

| 函数 | 说明 | 示例 |
|------|------|------|
| `min(a, b, ...)` | 返回最小值 | `min(batch_size * seq_len, 128)` |
| `max(a, b, ...)` | 返回最大值 | `max(batch_size, 1)` |
| `abs(x)` | 返回绝对值 | `abs(x - y)` |
| `round(x)` | 四舍五入 | `round(x)` |
| `pow(x, y)` | 幂运算，等同于 `x ** y` | `pow(2, 10)` |

**注意事项**：
- 公式中的字符串必须用**双引号**包裹
- **不要使用方括号 `[...]`**，YAML 会将其解析为列表
- 函数调用语法与 Python 相同
- 推荐使用 `seq_len` 变量（等于 prompt_len + gen_len），公式内部会自动处理并行因子

**可用变量**：
- `batch_size`: 批次大小
- `seq_len`: 序列长度 (= prompt_len + gen_len)
- `hidden_size`: 隐藏层维度
- `num_layers`: 层数
- `num_attention_heads`: 注意力头数
- `num_key_value_heads`: KV 头数
- `head_dim`: 头维度
- `num_experts_per_tok`: MoE 每 token 激活的专家数
- `recommended_capacity_factor`: 容量因子（PD 分离时根据场景自动选择 has_prefill 或 decode）
- `tp_size`: Tensor Parallel 大小
- `cp_size`: Context Parallel 大小
- `window_size`: 滑动窗口大小（用于 SWA）

**默认值**：如果未指定 `tp_size` 或 `cp_size`，默认为 1

**PD 分离场景支持**：

`recommended_capacity_factor` 支持两种格式：

1. **旧格式（单一浮点数）**：
```yaml
recommended_capacity_factor: 1.25
```

2. **新格式（嵌套字典）**：
```yaml
recommended_capacity_factor:
  has_prefill: 1.25   # prefill（含混部）
  decode: 12.5        # 纯 decode
```

当使用 `--find-max-seq-len` 时：
- 搜索最大 gen_len（固定 prompt_len）：使用 `decode` factor
- 搜索最大 prompt_len（固定 gen_len）：使用 `has_prefill` factor

**示例**：

1. **GQA 注意力 + MoE**：
```yaml
computation_rules:
  recommended_capacity_factor:
    has_prefill: 1.25
    decode: 12.5
  # KV Cache: 按 attention 类型，TP 和 CP 分片
  kv_cache: "2 * batch_size * seq_len * num_key_value_heads * head_dim * num_layers / (tp_size * cp_size)"
  # Activation: 按 ffn 类型，CP 分片
  activation: "batch_size * seq_len * hidden_size * num_experts_per_tok * recommended_capacity_factor / cp_size"
```

2. **带峰值限制的公式**（用于 Prefill/Decode 分离部署）：
```yaml
computation_rules:
  recommended_capacity_factor:
    has_prefill: 1.25
    decode: 12.5
  # KV Cache: 使用 min 限制序列长度峰值
  kv_cache: "(12 * batch_size * seq_len * 1024 + 12 * min(batch_size * seq_len, 128) * 1024) / (tp_size * cp_size)"
  activation: "batch_size * seq_len * hidden_size * 4 * recommended_capacity_factor / cp_size"
```

3. **MLA 注意力**（DeepSeek 系列）：
```yaml
computation_rules:
  recommended_capacity_factor:
    has_prefill: 1.25
    decode: 12.5
  # MLA 使用压缩 KV，只按 CP 分片
  kv_cache: "batch_size * seq_len * (kv_lora_rank + qk_rope_head_dim) * num_layers / cp_size"
  activation: "batch_size * seq_len * hidden_size * num_experts_per_tok * recommended_capacity_factor / cp_size"
```

4. **使用 generic 默认规则**：
   - `weight_mapping_rules.yaml` 中的 `generic.computation_rules` 提供了按 attention_type 和 ffn_type 的默认公式
   - 模型配置可以 inherit generic 并只覆盖特定规则

---

## 5. 如何手动创建配置文件

### 5.1 自动生成（推荐）

使用工具自动从 HuggingFace 模型生成配置：

```bash
python scripts/calculate_mem.py --model openai/gpt-oss-120b --generate-config
```

### 5.2 手动创建步骤

1. **获取模型信息**：
   - 从 HuggingFace 下载 `config.json`
   - 查看模型的权重文件（safetensors）

2. **填写 model_identity**：
   - 从 `config.json` 获取基本信息

3. **填写 architecture_config**：
   - 从 `config.json` 获取架构参数
   - 确定 attention_type、ffn_type、norm_type

4. **定义 modules**：
   - 列出所有权重名称
   - 确定每个权重的 shape、dtype、layers
   - 根据权重名称分类到对应的模块类型
   - 对于每层都有的权重，设置 `layers: <num_layers>`
   - 对于全局共享的权重，设置 `layers: 1`

5. **编写 computation_rules**：
   - 根据 attention_type 确定 KV Cache 公式
   - 根据 ffn_type 确定激活值公式

### 5.3 验证配置

创建配置后，使用工具验证：

```bash
python scripts/calculate_mem.py --config configs/models/your_model.yaml --prompt-len 4096 --gen-len 4096 --batch-size 1
```

---

## 6. 常见问题

### Q1: 如何确定权重的 layers 值？

**A**:
- 如果权重名称包含 `layers.0`, `layers.1`, ... `layers.N`，则 `layers = N + 1`
- 如果权重是全局共享的（如 `embed_tokens.weight`），则 `layers = 1`

### Q2: 如何确定 parallel_strategy？

**A**:
- 大多数情况下使用默认值 `replicated` 即可
- 如果需要模拟分布式训练/推理，根据权重的切分方式设置：
  - Q/K/V 投影：`tp_col`
  - 注意力输出投影：`tp_row`
  - MoE 专家：`expert_sharded`

### Q3: 如何处理量化权重？

**A**:
- 量化权重通常包含多个张量（blocks, scales, zeros）
- 每个张量都需要单独定义
- dtype 使用量化后的类型（如 `U8`, `INT4`）

**示例**：
```yaml
mlp.experts.gate_up_proj_blocks: {shape: [128, 5760, 90, 16], dtype: U8, layers: 36}
mlp.experts.gate_up_proj_scales: {shape: [128, 5760, 90], dtype: U8, layers: 36}
```

### Q4: ffn_moe 和 ffn_dense 有什么区别？

**A**:
- `ffn_moe`: MoE 架构的 FFN，包含路由器（router/gate）和多个专家（experts）
- `ffn_dense`: 标准的稠密 FFN，只有 gate_proj、up_proj、down_proj
- 一个模型只会使用其中一种

---

## 7. 完整示例

参考 `configs/models/gpt-oss-120b.yaml` 查看完整的配置示例。

---

## 8. 相关文档

- `docs/plan.md`: 详细的实现计划和架构说明
- `docs/detailed_design.md`: 详细设计文档
- `CLAUDE.md`: 项目概览和开发指南
