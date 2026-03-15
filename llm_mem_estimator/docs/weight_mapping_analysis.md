# 权重映射规则分析和改进建议

基于 DeepSeek-R1 和 GPT-OSS 120B 的实际权重结构分析。

## 1. DeepSeek-R1 权重分析

### 1.1 Embedding 类权重

| 权重名称 | 形状 | 数据类型 | 数量 | 分类建议 |
|---------|------|---------|------|---------|
| `embed_tokens.weight` | [129280, 7168] | BF16 | 1 | embedding.embed_tokens |
| `model.embed_tokens.weight` | [129280, 7168] | BF16 | 1 | embedding.embed_tokens (重复) |
| `lm_head.weight` | [129280, 7168] | BF16 | 1 | embedding.lm_head |
| `shared_head.head.weight` | [129280, 7168] | BF16 | 1 | embedding.shared_head |

**问题**：
1. `embed_tokens.weight` 和 `model.embed_tokens.weight` 重复，需要去重
2. `shared_head.head.weight` 需要新增规则匹配

### 1.2 Norm 类权重

| 权重名称 | 形状 | 数据类型 | 数量 | 分类建议 |
|---------|------|---------|------|---------|
| `enorm.weight` | [7168] | BF16 | 1 | norm (layers=1) |
| `hnorm.weight` | [7168] | BF16 | 1 | norm (layers=1) |
| `model.norm.weight` | [7168] | BF16 | 1 | norm (layers=1) |
| `shared_head.norm.weight` | [7168] | BF16 | 1 | norm (layers=1) |
| `input_layernorm.weight` | [7168] | BF16 | 62 | norm (layers=62) |
| `post_attention_layernorm.weight` | [7168] | BF16 | 62 | norm (layers=62) |

**问题**：
1. `enorm`, `hnorm` 等特殊命名需要匹配
2. `shared_head.norm.weight` 需要匹配

### 1.3 Attention 类权重（MLA 架构）

| 权重名称 | 形状 | 数据类型 | 数量 | 分类建议 |
|---------|------|---------|------|---------|
| `self_attn.q_a_proj.weight` | [1536, 7168] | F8_E4M3 | 62 | attention.q_a_proj |
| `self_attn.q_a_proj.weight_scale_inv` | [12, 56] | F32 | 62 | attention.q_a_proj_scale |
| `self_attn.q_a_layernorm.weight` | [1536] | BF16 | 62 | attention.q_a_layernorm |
| `self_attn.q_b_proj.weight` | [24576, 1536] | F8_E4M3 | 62 | attention.q_b_proj |
| `self_attn.q_b_proj.weight_scale_inv` | [192, 12] | F32 | 62 | attention.q_b_proj_scale |
| `self_attn.kv_a_proj_with_mqa.weight` | [576, 7168] | F8_E4M3 | 62 | attention.kv_a_proj_with_mqa |
| `self_attn.kv_a_proj_with_mqa.weight_scale_inv` | [5, 56] | F32 | 62 | attention.kv_a_proj_scale |
| `self_attn.kv_a_layernorm.weight` | [512] | BF16 | 62 | attention.kv_a_layernorm |
| `self_attn.kv_b_proj.weight` | [32768, 512] | F8_E4M3 | 62 | attention.kv_b_proj |
| `self_attn.kv_b_proj.weight_scale_inv` | [256, 4] | F32 | 62 | attention.kv_b_proj_scale |
| `self_attn.o_proj.weight` | [7168, 16384] | F8_E4M3 | 62 | attention.o_proj |
| `self_attn.o_proj.weight_scale_inv` | [56, 128] | F32 | 62 | attention.o_proj_scale |

**问题**：
1. 需要匹配 `weight_scale_inv` 后缀
2. `q_a_layernorm` 和 `kv_a_layernorm` 应该归类到 attention（因为是 MLA 特有的）

### 1.4 FFN 类权重（MoE 架构）

#### Dense MLP（前3层）
| 权重名称 | 形状 | 数据类型 | 数量 | 分类建议 |
|---------|------|---------|------|---------|
| `mlp.gate_proj.weight` | [18432, 7168] | F8_E4M3 | 3 | ffn.dense_mlp |
| `mlp.gate_proj.weight_scale_inv` | [144, 56] | F32 | 3 | ffn.dense_mlp |
| `mlp.up_proj.weight` | [18432, 7168] | F8_E4M3 | 3 | ffn.dense_mlp |
| `mlp.up_proj.weight_scale_inv` | [144, 56] | F32 | 3 | ffn.dense_mlp |
| `mlp.down_proj.weight` | [7168, 18432] | F8_E4M3 | 3 | ffn.dense_mlp |
| `mlp.down_proj.weight_scale_inv` | [56, 144] | F32 | 3 | ffn.dense_mlp |

#### Router Experts（59层 × 256个expert = 15104）
| 权重名称 | 形状 | 数据类型 | 数量 | 分类建议 |
|---------|------|---------|------|---------|
| `mlp.experts.[N].gate_proj.weight` | [2048, 7168] | F8_E4M3 | 15104 | ffn.router_expert |
| `mlp.experts.[N].gate_proj.weight_scale_inv` | [16, 56] | F32 | 15104 | ffn.router_expert |
| `mlp.experts.[N].up_proj.weight` | [2048, 7168] | F8_E4M3 | 15104 | ffn.router_expert |
| `mlp.experts.[N].up_proj.weight_scale_inv` | [16, 56] | F32 | 15104 | ffn.router_expert |
| `mlp.experts.[N].down_proj.weight` | [7168, 2048] | F8_E4M3 | 15104 | ffn.router_expert |
| `mlp.experts.[N].down_proj.weight_scale_inv` | [56, 16] | F32 | 15104 | ffn.router_expert |
| `mlp.gate.weight` | [256, 7168] | BF16 | 59 | ffn.router |
| `mlp.gate.e_score_correction_bias` | [256] | F32 | 59 | ffn.router |

#### Shared Experts（59层）
| 权重名称 | 形状 | 数据类型 | 数量 | 分类建议 |
|---------|------|---------|------|---------|
| `mlp.shared_experts.gate_proj.weight` | [2048, 7168] | F8_E4M3 | 59 | ffn.shared_expert |
| `mlp.shared_experts.gate_proj.weight_scale_inv` | [16, 56] | F32 | 59 | ffn.shared_expert |
| `mlp.shared_experts.up_proj.weight` | [2048, 7168] | F8_E4M3 | 59 | ffn.shared_expert |
| `mlp.shared_experts.up_proj.weight_scale_inv` | [16, 56] | F32 | 59 | ffn.shared_expert |
| `mlp.shared_experts.down_proj.weight` | [7168, 2048] | F8_E4M3 | 59 | ffn.shared_expert |
| `mlp.shared_experts.down_proj.weight_scale_inv` | [56, 16] | F32 | 59 | ffn.shared_expert |

**问题**：
1. 需要区分 `mlp.gate_proj` (dense), `mlp.experts.[N]` (router), `mlp.shared_experts` (shared)
2. 需要使用排除模式避免误匹配

### 1.5 Others 类权重
| 权重名称 | 形状 | 数据类型 | 数量 | 分类建议 |
|---------|------|---------|------|---------|
| `eh_proj.weight` | [7168, 14336] | BF16 | 1 | others |

---

## 2. GPT-OSS 120B 权重分析

### 2.1 Embedding 类权重
| 权重名称 | 形状 | 数据类型 | 数量 | 分类建议 |
|---------|------|---------|------|---------|
| `model.embed_tokens.weight` | [201088, 2880] | BF16 | 1 | embedding.embed_tokens |
| `lm_head.weight` | [201088, 2880] | BF16 | 1 | embedding.lm_head |

✅ 通用规则可以匹配

### 2.2 Norm 类权重
| 权重名称 | 形状 | 数据类型 | 数量 | 分类建议 |
|---------|------|---------|------|---------|
| `model.norm.weight` | [2880] | BF16 | 1 | norm (layers=1) |
| `input_layernorm.weight` | [2880] | BF16 | 36 | norm (layers=36) |
| `post_attention_layernorm.weight` | [2880] | BF16 | 36 | norm (layers=36) |

✅ 通用规则可以匹配

### 2.3 Attention 类权重
| 权重名称 | 形状 | 数据类型 | 数量 | 分类建议 |
|---------|------|---------|------|---------|
| `self_attn.q_proj.weight` | [4096, 2880] | BF16 | 36 | attention.q_proj |
| `self_attn.q_proj.bias` | [4096] | BF16 | 36 | attention.q_proj |
| `self_attn.k_proj.weight` | [512, 2880] | BF16 | 36 | attention.k_proj |
| `self_attn.k_proj.bias` | [512] | BF16 | 36 | attention.k_proj |
| `self_attn.v_proj.weight` | [512, 2880] | BF16 | 36 | attention.v_proj |
| `self_attn.v_proj.bias` | [512] | BF16 | 36 | attention.v_proj |
| `self_attn.o_proj.weight` | [2880, 4096] | BF16 | 36 | attention.o_proj |
| `self_attn.o_proj.bias` | [2880] | BF16 | 36 | attention.o_proj |
| `self_attn.sinks` | [64] | BF16 | 36 | attention.sinks |

**问题**：
1. 需要匹配 `.bias` 后缀
2. `self_attn.sinks` 是特殊权重（StreamingLLM 的 attention sink），需要匹配

### 2.4 FFN 类权重（特殊 MoE 结构）
| 权重名称 | 形状 | 数据类型 | 数量 | 分类建议 |
|---------|------|---------|------|---------|
| `mlp.router.weight` | [128, 2880] | BF16 | 36 | ffn.router |
| `mlp.router.bias` | [128] | BF16 | 36 | ffn.router |
| `mlp.experts.gate_up_proj_blocks` | [128, 5760, 90, 16] | U8 | 36 | ffn.router_expert |
| `mlp.experts.gate_up_proj_scales` | [128, 5760, 90] | U8 | 36 | ffn.router_expert |
| `mlp.experts.gate_up_proj_bias` | [128, 5760] | BF16 | 36 | ffn.router_expert |
| `mlp.experts.down_proj_blocks` | [128, 2880, 90, 16] | U8 | 36 | ffn.router_expert |
| `mlp.experts.down_proj_scales` | [128, 2880, 90] | U8 | 36 | ffn.router_expert |
| `mlp.experts.down_proj_bias` | [128, 2880] | BF16 | 36 | ffn.router_expert |

**特点**：
1. 使用量化的 `blocks` 和 `scales`（U8 数据类型）
2. `gate_up_proj` 是融合权重（gate 和 up 合并）
3. 第一维 `[128, ...]` 表示 128 个 experts
4. 每层有 36 个这样的权重组（对应 36 层）

---

## 3. 改进的规则配置

### 3.1 通用规则改进

```yaml
generic:
  embedding:
    embed_tokens:
      patterns:
        - "^embed_tokens\\.weight$"
        - "^model\\.embed_tokens\\.weight$"
        - ".*wte.*"
        - ".*word_embeddings.*"
    lm_head:
      patterns:
        - "^lm_head\\.weight$"
        - ".*output\\.weight$"

  attention:
    q_proj:
      patterns: [".*\\.q_proj\\.(weight|bias)$", ".*\\.query.*"]
    k_proj:
      patterns: [".*\\.k_proj\\.(weight|bias)$", ".*\\.key.*"]
    v_proj:
      patterns: [".*\\.v_proj\\.(weight|bias)$", ".*\\.value.*"]
    o_proj:
      patterns: [".*\\.o_proj\\.(weight|bias)$", ".*\\.dense.*"]
    attention_other:
      patterns: [".*self_attn\\..*"]  # 捕获其他 attention 相关权重

  ffn:
    dense_mlp:
      patterns:
        - "^mlp\\.(gate_proj|up_proj|down_proj)\\..*"
      exclude_patterns:
        - ".*\\.experts\\..*"
        - ".*\\.shared_experts\\..*"
    router_expert:
      patterns:
        - ".*\\.experts\\.\\d+\\..*"
        - ".*\\.experts\\.(gate_up_proj|down_proj).*"  # 融合权重
      exclude_patterns:
        - ".*\\.shared_experts\\..*"
    shared_expert:
      patterns:
        - ".*\\.shared_experts\\..*"
    router:
      patterns:
        - ".*\\.gate\\..*"
        - ".*\\.router\\..*"

  norm:
    patterns:
      - ".*input_layernorm.*"
      - ".*post_attention_layernorm.*"
      - "^model\\.norm\\.weight$"
      - ".*[^a-z]norm\\.weight$"  # enorm, hnorm 等
```

### 3.2 DeepSeek 特定规则

```yaml
deepseek:
  embedding:
    embed_tokens:
      patterns:
        - "^embed_tokens\\.weight$"
        - "^model\\.embed_tokens\\.weight$"
    lm_head:
      patterns:
        - "^lm_head\\.weight$"
    shared_head:
      patterns:
        - "^shared_head\\.head\\.weight$"

  attention:
    q_a_proj:
      patterns: [".*\\.q_a_proj\\.weight$"]
    q_a_proj_scale:
      patterns: [".*\\.q_a_proj\\.weight_scale_inv$"]
    q_a_layernorm:
      patterns: [".*\\.q_a_layernorm\\.weight$"]
    q_b_proj:
      patterns: [".*\\.q_b_proj\\.weight$"]
    q_b_proj_scale:
      patterns: [".*\\.q_b_proj\\.weight_scale_inv$"]
    kv_a_proj_with_mqa:
      patterns: [".*\\.kv_a_proj_with_mqa\\.weight$"]
    kv_a_proj_scale:
      patterns: [".*\\.kv_a_proj_with_mqa\\.weight_scale_inv$"]
    kv_a_layernorm:
      patterns: [".*\\.kv_a_layernorm\\.weight$"]
    kv_b_proj:
      patterns: [".*\\.kv_b_proj\\.weight$"]
    kv_b_proj_scale:
      patterns: [".*\\.kv_b_proj\\.weight_scale_inv$"]
    o_proj:
      patterns: [".*\\.o_proj\\.weight$"]
    o_proj_scale:
      patterns: [".*\\.o_proj\\.weight_scale_inv$"]

  ffn:
    dense_mlp:
      patterns:
        - "^mlp\\.(gate_proj|up_proj|down_proj)\\.weight.*"
      exclude_patterns:
        - ".*\\.experts\\..*"
        - ".*\\.shared_experts\\..*"
    router_expert:
      patterns:
        - "^mlp\\.experts\\.\\d+\\..*"
      exclude_patterns:
        - ".*\\.shared_experts\\..*"
    shared_expert:
      patterns:
        - "^mlp\\.shared_experts\\..*"
    router:
      patterns:
        - "^mlp\\.gate\\..*"

  norm:
    patterns:
      - ".*input_layernorm.*"
      - ".*post_attention_layernorm.*"
      - "^model\\.norm\\.weight$"
      - "^enorm\\.weight$"
      - "^hnorm\\.weight$"
      - "^shared_head\\.norm\\.weight$"

  others:
    patterns:
      - "^eh_proj\\.weight$"
```

### 3.3 GPT-OSS 特定规则

```yaml
gpt_oss:
  inherit: generic

  ffn:
    router:
      patterns:
        - "^mlp\\.router\\..*"
    router_expert:
      patterns:
        - "^mlp\\.experts\\.(gate_up_proj|down_proj).*"
```

---

## 4. 关键改进点

### 4.1 更精确的模式匹配
- 使用 `^` 和 `$` 锚定，避免误匹配
- 使用 `(weight|bias)` 匹配多种后缀
- 使用排除模式 `exclude_patterns` 避免冲突

### 4.2 处理重复权重
- `embed_tokens.weight` 和 `model.embed_tokens.weight` 需要去重
- 在配置生成时，检测并合并重复权重

### 4.3 支持特殊权重
- `weight_scale_inv`：量化缩放因子
- `bias`：偏置项
- `blocks` 和 `scales`：量化块
- `sinks`：StreamingLLM 的 attention sink
- `e_score_correction_bias`：MoE 路由偏置

### 4.4 FFN 分类优先级
1. 先匹配 `shared_experts`（最具体）
2. 再匹配 `experts.\d+`（router experts）
3. 最后匹配 `mlp.(gate|up|down)_proj`（dense mlp）
4. 使用排除模式避免误匹配

### 4.5 处理融合权重
- GPT-OSS 的 `gate_up_proj`：gate 和 up 融合
- 需要特殊处理，不能简单拆分

---

## 5. 配置生成逻辑改进

### 5.1 去重逻辑
```python
def deduplicate_weights(all_tensors: Dict) -> Dict:
    """去除重复的权重（如 embed_tokens.weight 和 model.embed_tokens.weight）"""
    seen_shapes = {}
    unique_tensors = {}

    for name, info in all_tensors.items():
        key = (tuple(info.shape), str(info.dtype))
        if key in seen_shapes:
            # 优先保留不带 model. 前缀的
            existing_name = seen_shapes[key]
            if name.startswith("model.") and not existing_name.startswith("model."):
                continue  # 跳过带前缀的
            elif not name.startswith("model.") and existing_name.startswith("model."):
                # 替换为不带前缀的
                del unique_tensors[existing_name]
                unique_tensors[name] = info
                seen_shapes[key] = name
        else:
            unique_tensors[name] = info
            seen_shapes[key] = name

    return unique_tensors
```

### 5.2 统计 layers 字段
```python
def calculate_layers(weights_list: List[Dict]) -> int:
    """
    统计权重出现的层数

    - 如果所有权重都没有 layer_idx，返回 1（全局权重）
    - 如果有 layer_idx，返回唯一层数的数量
    """
    layer_indices = set()
    for w in weights_list:
        if w['layer_idx'] is not None:
            layer_indices.add(w['layer_idx'])

    if not layer_indices:
        return 1  # 全局权重
    else:
        return len(layer_indices)  # 层数
```

### 5.3 处理 Expert 权重
```python
def build_expert_config(weights_list: List[Dict]) -> ExpertConfig:
    """构建 Expert 配置"""
    # 统计层数
    layer_indices = set(w['layer_idx'] for w in weights_list if w['layer_idx'] is not None)
    layer_count = len(layer_indices)

    # 统计每层的 expert 数量
    expert_indices_per_layer = defaultdict(set)
    for w in weights_list:
        if w['layer_idx'] is not None and w['expert_idx'] is not None:
            expert_indices_per_layer[w['layer_idx']].add(w['expert_idx'])

    count_per_layer = len(expert_indices_per_layer[min(layer_indices)])

    # 按权重类型分组
    weights_by_type = defaultdict(list)
    for w in weights_list:
        simplified_name = simplify_weight_name(w['name'])
        weights_by_type[simplified_name].append(w)

    # 生成 weights 字典
    weights = {}
    for weight_type, weight_list in weights_by_type.items():
        first_weight = weight_list[0]
        weights[weight_type] = WeightInfo(
            shape=first_weight['shape'],
            dtype=first_weight['dtype'],
            layers=1,  # Expert 权重的 layers 始终为 1
            parallel_strategy='expert_sharded'
        )

    return ExpertConfig(
        layer_count=layer_count,
        count_per_layer=count_per_layer,
        weights=weights
    )
```

---

## 6. 总结

### 6.1 当前设计的优点
✅ 配置驱动，灵活可扩展
✅ 支持通用规则和模型特定规则
✅ 使用正则表达式，匹配能力强

### 6.2 需要改进的地方
1. **规则精度**：需要更精确的模式匹配（使用锚定）
2. **去重逻辑**：需要处理重复权重
3. **特殊权重**：需要支持 bias, scale, blocks 等
4. **排除模式**：需要使用排除模式避免误匹配
5. **融合权重**：需要处理 gate_up_proj 等融合权重

### 6.3 建议
1. 更新 `weight_mapping_rules.yaml` 使用改进的规则
2. 在 `WeightClassifier` 中实现排除模式逻辑
3. 在 `ConfigLoader.generate_config_from_weights` 中实现去重逻辑
4. 添加单元测试验证规则的正确性
