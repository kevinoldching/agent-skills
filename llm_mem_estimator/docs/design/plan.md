# LLM 显存占用估算器 - 实现计划

## 目标

做一个自动计算大模型权重/KV Cache/激活值等显存占用的估算程序，作为一个功能模块被 agent skill 调用。

## 输入参数

| 类别 | 参数 | 说明 |
|------|------|------|
| 模型配置 | yaml_config | 每个模型提供一个 YAML 配置文件，包含模型标识、架构参数、模块定义、并行策略等 |
| 序列配置 | seq_len, batch_size | 序列长度、批大小 |
| 量化配置 | dtype | 数据类型：BF16/FP16 (不量化)、INT8、INT4 |
| 并行策略 | tp, pp, dp, ep, sp, cp | Tensor Parallel、Pipeline Parallel、Data Parallel、Expert Parallel、Sequence Parallel、Context Parallel |
| 硬件信息 | vram_gb, bandwidth_gb/s | 显存容量(GB)、显存带宽(GB/s) |

### 1. 大模型结构参数

#### 1.1 核心维度参数

| 参数 | 说明 | 示例 (Llama-70B) |
|------|------|-----------------|
| hidden_size | 隐藏层维度 | 8192 |
| num_layers | Transformer 层数 | 80 |
| num_attention_heads | Attention 头数 | 64 |
| num_kv_heads | KV 头数 | 64 |
| intermediate_size | FFN 中间层维度 | 28672 |
| vocab_size | 词表大小 | 128256 |
| num_experts | MoE 专家总数 | - |
| num_active_experts | MoE 激活专家数 | - |

#### 1.2 Attention 类型及参数

不同 Attention 类型有不同的参数字段：

| Attention 类型 | 参数字段 |
|---------------|----------|
| MHA | num_attention_heads, num_kv_heads, head_dim |
| MQA | num_attention_heads, head_dim |
| GQA | qk_head_dim, num_attention_heads, num_key_value_heads, v_head_dim, head_dim |
| SWA | num_attention_heads, num_key_value_heads, sliding_window_size, head_dim |
| MLA | kv_lora_rank, num_attention_heads, num_key_value_heads, q_lora_rank, qk_rope_head_dim, qk_nope_head_dim, v_head_dim, n_group, topk_group |
| DSA | kv_lora_rank, index_head_dim, index_n_heads, index_topk, num_attention_heads, num_key_value_heads, q_lora_rank, qk_rope_head_dim, qk_nope_head_dim |
| GDN | linear_num_key_heads, linear_value_head_dim, linear_key_head_dim, linear_num_value_heads, linear_conv_kernel_dim |

没有列出来的可以认为是`UNKNOWN`类型

**各 Attention 类型详细参数**：

**GQA (Grouped-Query Attention)**:
- qk_head_dim: QK 的 head 维度
- num_attention_heads: Attention 头数
- num_key_value_heads: KV 头数 (groups = heads / kv_heads)
- v_head_dim: V 的 head 维度
- head_dim: 通用 head 维度

**SWA (Sliding Window Attention)**:
- num_attention_heads: Attention 头数
- num_key_value_heads: KV 头数
- sliding_window_size: 滑动窗口大小
- head_dim: head 维度

**MLA (Multi-Latent Attention)**:
- kv_lora_rank: KV 压缩维度
- num_attention_heads: Attention 头数
- num_key_value_heads: KV 头数
- q_lora_rank: Q 压缩维度
- qk_rope_head_dim: RoPE 部分的 head 维度
- qk_nope_head_dim: 非 RoPE 部分的 head 维度
- v_head_dim: V 的 head 维度
- n_group: GQA 分组数
- topk_group: topk 专家分组数

**GDN (Generalized Delta Networks)**:
- linear_num_key_heads: K 的头数
- linear_value_head_dim: V 的 head 维度
- linear_key_head_dim: K 的 head 维度
- linear_num_value_heads: V 的头数
- linear_conv_kernel_dim: 卷积核维度

**DSA (DeepSeek Attention)**:
- kv_lora_rank: KV 压缩维度
- index_head_dim: 索引 head 维度
- index_n_heads: 索引头数
- index_topk: 索引 topk
- num_attention_heads: Attention 头数
- num_key_value_heads: KV 头数
- q_lora_rank: Q 压缩维度
- qk_rope_head_dim: RoPE 部分的 head 维度
- qk_nope_head_dim: 非 RoPE 部分的 head 维度

**Attention 类型对显存的影响**

不同 Attention 类型影响 KV Cache 的显存计算：

| 类型 | KV Cache 特点 |
|------|---------------|
| MHA | 标准：num_key_value_heads = num_attention_heads |
| MQA | KV 头数为 1，所有 head 共享 |
| GQA | num_key_value_heads < num_attention_heads |
| SWA | 滑动窗口限制序列长度 |
| MLA / DSA | 使用 LoRA 压缩，大幅减少 KV Cache |
| GDN | 使用线性变换 + 卷积 |

#### 1.3 FFN 类型及参数

不同 FFN 类型有不同的参数字段：

| FFN 类型 | 参数字段 |
|----------|----------|
| dense | intermediate_size |
| moe | moe_intermediate_size, topk, num_router_experts, num_shared_experts |

**MoE 完整参数**:
- moe_intermediate_size: 每个专家的中间层维度
- topk: 激活的专家数量
- num_router_experts: 路由专家总数
- num_shared_experts: 共享专家数量

**注意**: SwiGLU、GeGLU 等激活函数属于 dense 类型，使用与标准 FFN 相同的显存公式。

#### 1.4 Norm 类型

| 类型 | 说明 |
|------|------|
| LayerNorm | 标准层归一化 |
| RMSNorm | 均方根归一化 |
| DeepNorm | 深度归一化 |

#### 1.5 配置示例

**配置文件格式**: 每个模型一个 YAML 文件 (`configs/models/<model_name>.yaml`)

采用模块化配置，包含：模型标识、架构配置、逻辑模块映射、计算规则。

配置文件具体规则如下：
- 非固定字段
  - architecture_config下面的字段，不同模型的超参的名称和数量可能不一样
  - modules下面embedding/attention/attention/ffn/others下面的字段名称，例如：embed_tokens/lm_head/shared_head，可能根据具体的模型权重不一样，但shape/dtype/parallel_strategy/world_size字段名称固定，又如attention.components下面的字段不固定，ffn.router_expert/ffn.shared_expert/ffn.dense_mlp下面的gate_proj/up_proj/down_proj/gate名称不固定
- 固定字段：除非固定字段外，其余均为固定字段，方便程序实现
- 可选字段：`parallel_strategy`及`world_size`是可选字段，在加载后可以被程序修改

配置文件的生成：
- 通过从HuggingFace或本地权重目录读取权重文件的metadata进行解析
- 支持用户手动调用，以便确认生成的文件正确性，也可以根据实际模型名称自动触发调用
- 示例代码如下：

```python
import re
from collections import defaultdict
from huggingface_hub import get_safetensors_metadata

def analyze_model_weights(repo_id, token=None):
    try:
        metadata = get_safetensors_metadata(repo_id, token=token)
        
        # 1. 获取所有 Tensor 的维度映射
        all_tensors = {}
        f_meta = metadata.files_metadata
        for tensor_name, file_name in metadata.weight_map.items():
            file_obj = f_meta[file_name] if isinstance(f_meta, dict) else next(f for f in f_meta if getattr(f, 'file_name', '') == file_name)
            if tensor_name in file_obj.tensors:
                all_tensors[tensor_name] = file_obj.tensors[tensor_name]

        # 2. 统计归类 (使用正则过滤掉层号等无关信息)
        # 不同模型的权重名称可能不一样，需要预先设置过滤和映射规则集，以便将权重名称映射成我们定义的标准的YAML配置文件
        for name, info in all_tensors.items():
          ...

        # 3. 获取config.json
          ...

        # 4. 生成标准的YAML文件
          ...

    except Exception as e:
        print(f"解析失败: {e}")
        return None

# --- 运行分析 ---
info = analyze_model_weights("deepseek-ai/DeepSeek-R1", token="Your HF Access Token")
```

以Deepseek-R1模型为例，其配置文件示例如下：

```yaml
model_identity:
  name: "DeepSeek-R1"
  total_params_b: 684.53
  num_layers: 62
  quantization: "FP8-Blockwise-128x128"

# --- 1. 架构参数组 (来自 config.json) ---
architecture_config:
  num_layers: 62
  hidden_size: 7168
  intermediate_size: 18432
  max_position_embeddings: 163840
  vocab_size: 129280
  num_heads: 128
  q_lora_rank: 1536
  kv_lora_rank: 512
  qk_rope_head_dim: 64
  v_head_dim: 128
  num_experts: 256
  num_experts_per_tok: 8
  moe_layer_freq: 1
  num_shared_experts: 1

# --- 2. 逻辑模块映射 ---
modules:
  # Embedding 模块 (全量存储/复制)
  embedding:
    embed_tokens.weight: {shape: [129280, 7168], dtype: "BF16", layers: 2, parallel_strategy: "replicated"}
    lm_head.weight:      {shape: [129280, 7168], dtype: "BF16", layers: 1, parallel_strategy: "tp_col", world_size: 0}
    shared_head.weight:  {shape: [129280, 7168], dtype: "BF16", layers: 1, parallel_strategy: "replicated"}

  # Norm 模块 (层归一化)
  norm:
    model.norm_weight: {shape: [7168], dtype: "BF16", layers: 1}
    enorm.weight:      {shape: [7168], dtype: "BF16", layers: 1}
    hnorm.weight:      {shape: [7168], dtype: "BF16", layers: 1}
    shared_head.norm_weight: {shape: [7168], dtype: "BF16", layers: 1}
    input_layernorm.weight: {shape: [7168], dtype: "BF16", layers: 62}
    post_attention_layernorm.weight: {shape: [7168], dtype: "BF16", layers: 62}

  # MLA 注意力模块 (共 62 层)
  attention:
    type: "MLA"
    num_layers: 62
    components:
      # Q 投影 (A降维, B升维)
      q_a_proj.weight: {shape: [1536, 7168], dtype: "F8_E4M3", parallel_strategy: "replicated"}
      q_a_proj.weight_scale_inv:  {shape: [12, 56], dtype: "F32", parallel_strategy: "replicated"}
      q_a_layernorm.weight:   {shape: [1536], dtype: "BF16"}
      q_b_pro._weight: {shape: [24576, 1536], dtype: "F8_E4M3", parallel_strategy: "tp_col", world_size: 0}
      q_b_proj.weight_scale_inv:  {shape: [192, 12], dtype: "F32"}
      
      # KV 投影 (MLA 特有的多头潜变量压缩)
      kv_a_proj_with_mqa.weight: {shape: [576, 7168], dtype: "F8_E4M3", parallel_strategy: "replicated"}
      kv_a_proj_with_mqa.weight_scale_inv:  {shape: [5, 56], dtype: "F32", parallel_strategy: "replicated"}
      kv_a_layernorm.weight:   {shape: [512], dtype: "BF16"}
      kv_b_proj.weight: {shape: [32768, 512], dtype: "F8_E4M3", parallel_strategy: "tp_col", world_size: 0}
      kv_b_proj.weight_scale_inv:  {shape: [256, 4], dtype: "F32"}
      
      # 输出投影
      o_proj.weight: {shape: [7168, 16384], dtype: "F8_E4M3", parallel_strategy: "tp_row", world_size: 0}
      o_proj.weight_scale_inv :  {shape: [56, 128], dtype: "F32"}

  # FFN 模块 (分三种子类型)
  ffn:
    # 1. 路由专家 (Router Experts) - 存在于 59 层中
    router_expert:
      layer_count: 59
      count_per_layer: 256
      gate_proj.weight: {shape: [2048, 7168],dtype: "F8_E4M3",parallel_strategy: "expert_sharded", world_size: 0}
      gate_proj.weight_scale_invscale: {shape: [16, 56],dtype: "F32",parallel_strategy: "expert_sharded", world_size: 0}
      down_proj.weight: {shape: [7168, 2048],dtype: "F8_E4M3",parallel_strategy: "expert_sharded", world_size: 0}
      down_proj.weight_scale_invscale: {shape: [56, 16],dtype: "F32",parallel_strategy: "expert_sharded", world_size: 0}
      up_proj.weight: {shape: [2048, 7168],dtype: "F8_E4M3",parallel_strategy: "expert_sharded", world_size: 0}
      up_proj.weight_scale_invscale: {shape: [16, 56],dtype: "F32",parallel_strategy: "expert_sharded", world_size: 0}
      gate.weight: {shape: [256, 7168], dtype: "BF16", parallel_strategy: "replicated"}
      gate.e_score_correction_bias:   {shape: [256], dtype: "F32"}

    # 2. 共享专家 (Shared Experts) - 每一层都有
    shared_expert:
      layer_count: 59
      count_per_layer: 1
      gate_proj.weight: {shape: [2048, 7168], dtype: "F8_E4M3", parallel_strategy: "replicated"}
      gate_proj.weight_scale_invscale: {shape: [16, 56], dtype: "F32", parallel_strategy: "replicated"}
      up_proj.weight: {shape: [2048, 7168], dtype: "F8_E4M3", parallel_strategy: "replicated"}
      up_proj.weight_scale_invscale: {shape: [16, 56], dtype: "F32", parallel_strategy: "replicated"}
      down_proj.weight: {shape: [7168, 2048], dtype: "F8_E4M3", parallel_strategy: "replicated"}
      down_proj.weight_scale_invscale:  {shape: [56, 16], dtype: "F32", parallel_strategy: "replicated"}

    # 3. 稠密层 (Dense MLP) - 仅前几层存在
    dense_mlp:
      layer_count: 3
      count_per_layer: 1
      gate_proj.weight: {shape: [18432, 7168],dtype: "F8_E4M3",parallel_strategy: "replicated"}
      gate_proj.weight_scale_inv: {shape: [144, 56],dtype: "F32",parallel_strategy: "replicated"}
      down_proj.weight: {shape: [7168, 18432],dtype: "F8_E4M3",parallel_strategy: "replicated"}
      down_proj.weight_scale_inv: {shape: [56, 144],dtype: "F32",parallel_strategy: "replicated"}
      up_proj.weight: {shape: [18432, 7168],dtype: "F8_E4M3",parallel_strategy: "replicated"}
      up_proj.weight_scale_inv: {shape: [144, 56],dtype: "F32",parallel_strategy: "replicated"}

  # 其他组件
  others:
    eh_proj.weight: {shape: [7168, 14336], dtype: "BF16", layers: 1, parallel_strategy: "replicated"}

# --- 3. 显存计算算子参考 ---
computation_rules:
    # 负载均衡敏感度配置
  activation_vram:
    ideal_capacity_factor: 1.0
    recommended_capacity_factor: 1.25 # 工业界常用的折中方案
    worst_case_capacity_factor: 8.0  # 对应你的极端不均衡假设
    formula: "S * num_experts_per_tok * hidden_size * recommended_capacity_factor * precision"
  kv_cache_vram:
    formula: "(kv_lora_rank + qk_rope_head_dim) * num_layers * 2 * precision"
```

**并行策略类型**：

| 策略 | 说明 |
|------|------|
| replicated | 全量复制，不切分 |
| tp_col | 张量并行，按列切分 |
| tp_row | 张量并行，按行切分 |
| expert_sharded | 专家并行，切分到不同卡 |
| pp | 流水线并行，按层切分 |

**数据类型与字节数**：

| dtype | 字节数 |
|-------|--------|
| BF16 | 2 |
| FP16 | 2 |
| F8_E4M3 | 1 |
| INT8 | 1 |
| INT4 | 0.5 |
| F32 | 4 |

#### 1.6 参数校验规则

程序不自动推导任何参数。用户必须提供指定model对应的yaml配置文件，否则报错。

#### 1.7 可扩展性设计原则

#### 1.7 可扩展性设计原则

大模型结构持续演进，程序设计需满足以下扩展性要求：

1. **模块化架构**
   - Attention、FFN、Norm 等模块独立实现
   - 新增模块类型时，只需添加新模块类，无需修改核心逻辑

2. **配置驱动**
   - 每个模型存储于独立的 YAML 配置文件 (`configs/models/<model_name>.yaml`)
   - 用户可直接编辑配置文件添加新模型
   - 支持通过命令行参数覆盖任意配置项

3. **插件式模块注册**
   - 模块类型通过注册机制添加
   - 示例：`register_attention('MLA', MLAAttention)` 或在配置文件中指定

4. **字段扩展兼容**
   - 配置文件支持添加任意自定义字段
   - 程序解析时忽略未知字段，保留向前向后兼容

---

## 输出

输出详细的报告，以markdown文件保存

- **总显存占用** (GB)
- **各组件明细（表格形式）**：
  - 模型权重显存
  - KV Cache 显存
  - 激活值显存
- **最大序列长度及Batch估算**：基于给定硬件信息及大模型结构、并行策略下，估算最多能支持多长的输出序列及并发数

## 计算公式

### 显存占用构成

```
总显存 = 权重占用 + 激活占用(峰值) + KV Cache + 系统保留 + 其它
```

#### 1. 权重占用

```
权重占用 = Σ(各模块参数量 × dtype_bytes)
```

注意，需要考虑并行策略。根据并行策略分配到卡（缩放），这是 YAML 中 parallel_strategy 字段发挥威力的地方。
假设你有 N 张卡：
| 策略 (Strategy) | 计算方法 | 含义 |
|-------|--------|--------|
| replicated | Size | 每张显卡都得吃下完整的权重（如 Embedding）|
| tp_col / tp_row | Size/TP_Size | 权重被张量并行组平分（如 MLA 的 B 矩阵）|
| expert_sharded| Size/EP_Size| 专家被均匀分摊到不同显卡（EP 并行）|


#### 2. 激活占用 (峰值)

在推理（Inference）场景下，我们通常只需要估算单层最高峰的激活值，因为显存可以复用。
可以通过模型yaml文件， computation_rules.activation_vram.formula提供了激活的估算公式，对于MoE模型来说，一般的激活峰值为Expert

**Capacity Factor (CF) 限制说明**
框架通常会设定一个 capacity_factor（例如 1.25）。它规定每张卡最多只接收：

$\frac{S \times \text{topK} \times N_{gpu}}{N_{gpu}} \times CF$ 个 Token。

公式变为：**激活占用** = $S \times \text{topK} \times H \times 1.25 \times dtype\_bytes$。

S：序列长度， H：隐藏层维度，CF：默认取yaml文件中的`recommended_capacity_factor`值

#### 3. KV Cache

可以通过模型yaml文件， computation_rules.kvcache_vram.formula提供了激活的估算公式，例如对于deepseek r1模型，采用了MLA的Attention，所以参考yaml文件，kvcache的计算公式如下：

```
KV Cache = (kv_lora_rank + qk_rope_head_dim) * num_layers * 2 * precision
```

- precision：如果采用了采用C8量化，`precision = 1`，否则`precision = 2`

#### 4. 系统保留

```
系统保留 = 固定值 (如 2GB) 或 显存容量 × 预留比例 (默认按显存容量的5%估算，可以配置)
```

#### 5. 其它

```
其它 = 通信 buffer (如 HCCL) + 临时显存 + 框架开销，默认按5%估算，可以配置
```

- 通信 buffer：分布式训练/推理时的 NCCL/HCCL 缓冲区
- 框架开销：CUDA context、临时 allocator 等

**数据类型字节数**：

| 数据类型 | 字节数 |
|---------|--------|
| FP16 / BF16 | 2 |
| FP8 | 1 |
| HiF8 | 1 |
| INT8 | 1 |
| INT4 | 0.5 |


#### 1.8 硬件配置参数

| 参数 | 说明 | 示例 |
|------|------|------|
| vram_gb | 单卡显存容量 (GB) | 80, 141, 64 |
| bandwidth_gb/s | 显存带宽 (GB/s) | 3.35 (H100 HBM3) |
| num_gpus | GPU 数量 | 1, 8, 64 |

**常见芯片规格** (`configs/chips.json`):

```json
{
  "h100-80gb": {
    "vram_gb": 80,
    "bandwidth_gb_s": 3.35
  },
  "a100-80gb": {
    "vram_gb": 80,
    "bandwidth_gb_s": 2.0
  },
  "ascend-910b-64gb": {
    "vram_gb": 64,
    "bandwidth_gb_s": 1.6
  }
}
```

---

## 目录结构

```
llm_mem_estimator/
├── SKILL.md           # Skill 接口定义
├── docs/
│   └── plan.md        # 本计划文档
└── scripts/           # 核心计算脚本
    └── xx.py  
```

## 触发关键词

- "显存占用"、"需要多少显存"
- "计算显存"、"显存估算"
- "这个batch需要多少显存"