# GPT-OSS-120B 元数据分析报告

## 测试命令
```bash
python3 scripts/test_metadata.py openai/gpt-oss-120b
```

## 问题诊断

### 1. YAML 文件过大的根本原因

**当前行为**：
- 总权重数量：687 个
- 每一层的每个权重都被单独存储
- 例如：`model.layers.0.self_attn.q_proj.weight`, `model.layers.1.self_attn.q_proj.weight`, ..., `model.layers.35.self_attn.q_proj.weight` 都被单独存储

**应该的行为**：
- 将相同模式的权重合并为一个条目
- 使用 `layers` 字段表示该权重在多少层中出现
- 例如：`self_attn.q_proj.weight` 只存储一次，设置 `layers: 36`

### 2. 权重分类问题

**当前状态**：
- 所有 687 个权重都被分类为 `others`
- 权重映射规则没有正确匹配 gpt-oss 模型的权重名称

**权重模式分析**（共 22 种模式）：

| 模式 | 出现次数 | 应该分类为 |
|------|---------|-----------|
| `model.layers.N.self_attn.q_proj.weight` | 36 | attention |
| `model.layers.N.self_attn.k_proj.weight` | 36 | attention |
| `model.layers.N.self_attn.v_proj.weight` | 36 | attention |
| `model.layers.N.self_attn.o_proj.weight` | 36 | attention |
| `model.layers.N.mlp.router.weight` | 36 | ffn (router) |
| `model.layers.N.mlp.experts.gate_up_proj_blocks` | 36 | ffn (router_expert) |
| `model.layers.N.mlp.experts.gate_up_proj_scales` | 36 | ffn (router_expert) |
| `model.layers.N.mlp.experts.down_proj_blocks` | 36 | ffn (router_expert) |
| `model.layers.N.mlp.experts.down_proj_scales` | 36 | ffn (router_expert) |
| `model.layers.N.input_layernorm.weight` | 36 | norm |
| `model.layers.N.post_attention_layernorm.weight` | 36 | norm |
| `model.embed_tokens.weight` | 1 | embedding |
| `lm_head.weight` | 1 | embedding |
| `model.norm.weight` | 1 | norm |

### 3. 模型特点

**架构信息**：
- 模型类型：`gpt_oss`
- 层数：36
- 隐藏层维度：2880
- 注意力头数：64 (KV 头数: 8) - GQA
- 专家数量：128 个专家，每次激活 4 个
- 量化方式：mxfp4（MX 格式的 FP4）

**特殊的量化格式**：
- 使用 `blocks` 和 `scales` 分离存储
- 例如：`gate_up_proj_blocks` (U8) + `gate_up_proj_scales` (U8)
- 这是 MX 格式量化的特点

## 需要修复的问题

### 问题 1：权重合并逻辑缺失

**位置**：`llm_mem_estimator/model_detector.py` 的 `ConfigGenerator._classify_weights` 方法

**当前代码**：
```python
for weight_name, metadata in weights_metadata.items():
    module_type = self.classifier.classify_weight(weight_name, model_type)
    # ...
    modules[module_type][weight_name] = WeightInfo(...)  # 每个都单独存储
```

**需要改为**：
1. 提取权重的基础模式（去掉层号）
2. 将相同模式的权重合并
3. 计算该模式在多少层中出现
4. 只存储基础模式，设置正确的 `layers` 值

### 问题 2：权重分类规则不匹配

**位置**：`configs/weight_mapping_rules.yaml`

**当前规则**：
- 通用规则使用 `^self_attn\\..*` 或 `.*\\.self_attn\\..*`
- 但 gpt-oss 的权重名称是 `model.layers.N.self_attn.xxx`
- 规则应该能匹配，但实际上没有匹配成功

**可能的原因**：
- 规则中的正则表达式转义问题
- 或者 gpt_oss 特定规则缺失

### 问题 3：_count_weight_layers 方法不正确

**位置**：`llm_mem_estimator/model_detector.py` 的 `ConfigGenerator._count_weight_layers` 方法

**当前代码**：
```python
def _count_weight_layers(self, weight_name: str, total_layers: int) -> int:
    if 'layers.' in weight_name or 'layer.' in weight_name:
        return 1  # 总是返回 1
    else:
        return 1  # 也是返回 1
```

**问题**：
- 这个方法总是返回 1
- 应该检测权重是否在多层中重复出现，然后返回层数

## 建议的修复方案

### 方案 1：修改权重合并逻辑

在 `_classify_weights` 方法中：
1. 先将权重按基础模式分组
2. 对每个模式，检查是否在多层中出现
3. 如果是，合并为一个条目，设置 `layers` 为出现的层数
4. 如果不是（如 embedding），保持原样

### 方案 2：更新权重映射规则

为 gpt_oss 添加特定规则，支持：
- MoE 的 blocks 和 scales 格式
- router 权重
- experts 权重

### 方案 3：改进 _count_weight_layers 方法

让它能够：
1. 识别权重是否是 per-layer 的
2. 如果是，返回总层数
3. 如果不是，返回 1

## 测试输出文件

已生成以下文件供分析：
- `metadata_openai_gpt-oss-120b.json` (89KB) - 包含完整的 config 和 weights_metadata

## 下一步

1. 修复权重合并逻辑
2. 更新权重分类规则
3. 重新测试 gpt-oss-120b
4. 验证生成的 YAML 文件大小合理（应该只有几百行，而不是几万行）
