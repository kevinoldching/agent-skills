# 权重映射规则设计总结

根据 DeepSeek-R1 和 GPT-OSS 120B 的实际权重结构，更新后的规则设计。

## 核心设计原则

### 1. 基于前缀的分类策略

**原则**：根据权重名称的前缀进行分类，而不是逐个匹配具体的权重名称。

| 前缀模式 | 分类目标 | 说明 |
|---------|---------|------|
| `self_attn.*` | attention | 所有 attention 相关权重（包括 q/k/v/o_proj, layernorm, scale 等） |
| `mlp.gate.*` | ffn.router | 所有 router/gate 相关权重（包括 weight, bias 等） |
| `mlp.router.*` | ffn.router | Router 权重（GPT-OSS 风格） |
| `mlp.experts.\d+.*` | ffn.router_expert | Router experts 权重 |
| `mlp.experts.(gate_up_proj\|down_proj).*` | ffn.router_expert | 融合权重（GPT-OSS） |
| `mlp.shared_experts.*` | ffn.shared_expert | Shared experts 权重 |
| `mlp.(gate_proj\|up_proj\|down_proj).*` | ffn.dense_mlp | Dense MLP 权重（排除 experts） |

**优点**：
- 简化规则，不需要为每个子权重单独写规则
- 自动支持新的子权重（如 bias, scale, layernorm 等）
- 更容易维护和扩展

### 2. 使用排除模式避免误匹配

**示例**：Dense MLP 需要排除 experts
```yaml
dense_mlp:
  patterns:
    - "^mlp\\.(gate_proj|up_proj|down_proj)\\..*"
  exclude_patterns:
    - ".*\\.experts\\..*"
    - ".*\\.shared_experts\\..*"
```

### 3. 保留完整权重名称（不去重）

**原因**：
- 所有权重都会被加载到显存中
- `embed_tokens.weight` 和 `model.embed_tokens.weight` 可能是不同的权重
- 在生成 YAML 配置时，保留前缀以区分

**处理方式**：
```yaml
embedding:
  embed_tokens.weight: {shape: [129280, 7168], dtype: "BF16", layers: 1}
  model.embed_tokens.weight: {shape: [129280, 7168], dtype: "BF16", layers: 1}
```

### 4. 特殊模块处理

#### DeepSeek-R1 的 MTP 模块
- `eh_proj.weight` → others
- `enorm.weight`, `hnorm.weight` → norm

#### GPT-OSS 的量化 MoE
- `mlp.experts.gate_up_proj_blocks` → ffn.router_expert
- `mlp.experts.gate_up_proj_scales` → ffn.router_expert
- 融合权重：gate 和 up 合并

## 规则配置文件结构

### 通用规则（generic）

```yaml
generic:
  embedding:
    patterns:
      - "^embed_tokens\\.weight$"
      - "^model\\.embed_tokens\\.weight$"
      - "^lm_head\\.weight$"

  attention:
    patterns:
      - "^self_attn\\..*"
      - ".*\\.self_attn\\..*"

  ffn:
    router:
      patterns:
        - "^mlp\\.gate\\..*"
        - "^mlp\\.router\\..*"

    router_expert:
      patterns:
        - "^mlp\\.experts\\.\\d+\\..*"
      exclude_patterns:
        - ".*\\.shared_experts\\..*"

    shared_expert:
      patterns:
        - "^mlp\\.shared_experts\\..*"

    dense_mlp:
      patterns:
        - "^mlp\\.(gate_proj|up_proj|down_proj)\\..*"
      exclude_patterns:
        - ".*\\.experts\\..*"
        - ".*\\.shared_experts\\..*"

  norm:
    patterns:
      - "^input_layernorm\\..*"
      - "^post_attention_layernorm\\..*"
      - "^model\\.norm\\.weight$"
```

### DeepSeek 特定规则

```yaml
deepseek:
  embedding:
    patterns:
      - "^embed_tokens\\.weight$"
      - "^model\\.embed_tokens\\.weight$"
      - "^lm_head\\.weight$"
      - "^shared_head\\.head\\.weight$"

  attention:
    patterns:
      - "^self_attn\\..*"
      - ".*\\.self_attn\\..*"

  ffn:
    router:
      patterns:
        - "^mlp\\.gate\\..*"

    router_expert:
      patterns:
        - "^mlp\\.experts\\.\\d+\\..*"
      exclude_patterns:
        - ".*\\.shared_experts\\..*"

    shared_expert:
      patterns:
        - "^mlp\\.shared_experts\\..*"

    dense_mlp:
      patterns:
        - "^mlp\\.(gate_proj|up_proj|down_proj)\\..*"
      exclude_patterns:
        - ".*\\.experts\\..*"
        - ".*\\.shared_experts\\..*"

  norm:
    patterns:
      - "^input_layernorm\\..*"
      - "^post_attention_layernorm\\..*"
      - "^model\\.norm\\.weight$"
      - "^enorm\\.weight$"        # MTP 模块
      - "^hnorm\\.weight$"        # MTP 模块
      - "^shared_head\\.norm\\.weight$"

  others:
    patterns:
      - "^eh_proj\\.weight$"      # MTP 模块
```

### GPT-OSS 特定规则

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

## 分类示例

### DeepSeek-R1 权重分类

| 权重名称 | 匹配规则 | 分类结果 |
|---------|---------|---------|
| `self_attn.q_a_proj.weight` | `^self_attn\\..*` | attention |
| `self_attn.q_a_proj.weight_scale_inv` | `^self_attn\\..*` | attention |
| `self_attn.q_a_layernorm.weight` | `^self_attn\\..*` | attention |
| `mlp.gate.weight` | `^mlp\\.gate\\..*` | ffn.router |
| `mlp.gate.e_score_correction_bias` | `^mlp\\.gate\\..*` | ffn.router |
| `mlp.experts.0.gate_proj.weight` | `^mlp\\.experts\\.\\d+\\..*` | ffn.router_expert |
| `mlp.shared_experts.gate_proj.weight` | `^mlp\\.shared_experts\\..*` | ffn.shared_expert |
| `mlp.gate_proj.weight` | `^mlp\\.(gate_proj\|up_proj\|down_proj)\\..*` | ffn.dense_mlp |
| `enorm.weight` | `^enorm\\.weight$` | norm |
| `eh_proj.weight` | `^eh_proj\\.weight$` | others |

### GPT-OSS 120B 权重分类

| 权重名称 | 匹配规则 | 分类结果 |
|---------|---------|---------|
| `self_attn.q_proj.weight` | `^self_attn\\..*` | attention |
| `self_attn.q_proj.bias` | `^self_attn\\..*` | attention |
| `self_attn.sinks` | `^self_attn\\..*` | attention |
| `mlp.router.weight` | `^mlp\\.router\\..*` | ffn.router |
| `mlp.router.bias` | `^mlp\\.router\\..*` | ffn.router |
| `mlp.experts.gate_up_proj_blocks` | `^mlp\\.experts\\.(gate_up_proj\|down_proj).*` | ffn.router_expert |
| `mlp.experts.gate_up_proj_scales` | `^mlp\\.experts\\.(gate_up_proj\|down_proj).*` | ffn.router_expert |
| `mlp.experts.down_proj_blocks` | `^mlp\\.experts\\.(gate_up_proj\|down_proj).*` | ffn.router_expert |

## 实现要点

### 1. WeightClassifier 实现

```python
class WeightClassifier:
    def classify(self, weight_name: str) -> ClassificationResult:
        """
        分类权重名称

        优先级：
        1. 先匹配模型特定规则
        2. 再匹配通用规则
        3. 默认归类到 others
        """
        # 提取层号和专家编号
        layer_idx = self._extract_layer_idx(weight_name)
        expert_idx = self._extract_expert_idx(weight_name)

        # 尝试匹配规则
        for category in ['embedding', 'attention', 'ffn', 'norm', 'others']:
            if category in self.rules:
                result = self._match_category(weight_name, category, self.rules[category])
                if result:
                    return ClassificationResult(
                        category=category,
                        subcategory=result,
                        layer_idx=layer_idx,
                        expert_idx=expert_idx
                    )

        # 默认归类到 others
        return ClassificationResult(
            category='others',
            subcategory=weight_name,
            layer_idx=layer_idx
        )

    def _match_category(self, weight_name: str, category: str, rules: Dict) -> Optional[str]:
        """
        匹配某个类别的规则

        支持：
        1. 直接模式匹配（patterns）
        2. 排除模式（exclude_patterns）
        """
        if 'patterns' in rules:
            # 检查排除模式
            if 'exclude_patterns' in rules:
                for exclude_pattern in rules['exclude_patterns']:
                    if re.match(exclude_pattern, weight_name):
                        return None

            # 检查匹配模式
            for pattern in rules['patterns']:
                if re.match(pattern, weight_name):
                    return category

        # 如果有子类别（如 ffn.router, ffn.router_expert）
        for subcategory, subconfig in rules.items():
            if isinstance(subconfig, dict) and 'patterns' in subconfig:
                # 检查排除模式
                if 'exclude_patterns' in subconfig:
                    for exclude_pattern in subconfig['exclude_patterns']:
                        if re.match(exclude_pattern, weight_name):
                            continue

                # 检查匹配模式
                for pattern in subconfig['patterns']:
                    if re.match(pattern, weight_name):
                        return subcategory

        return None
```

### 2. 配置生成逻辑

```python
def generate_config_from_weights(repo_id: str, token: str = None) -> ModelConfig:
    """从 HuggingFace 生成配置文件"""

    # 1. 获取 metadata
    metadata = get_safetensors_metadata(repo_id, token=token)
    all_tensors = extract_all_tensors(metadata)

    # 2. 加载 config.json
    config_json = load_config_json(repo_id, token)

    # 3. 自动检测模型类型
    model_type = ModelDetector.detect_model_type(config_json, list(all_tensors.keys()))

    # 4. 创建分类器
    classifier = WeightClassifier(model_type=model_type)

    # 5. 分类所有权重（不去重）
    categorized_weights = defaultdict(lambda: defaultdict(list))
    for weight_name, weight_info in all_tensors.items():
        result = classifier.classify(weight_name)
        categorized_weights[result.category][result.subcategory].append({
            'name': weight_name,  # 保留完整名称
            'shape': list(weight_info.shape),
            'dtype': str(weight_info.dtype),
            'layer_idx': result.layer_idx,
            'expert_idx': result.expert_idx
        })

    # 6. 生成 YAML 配置
    yaml_config = build_yaml_config(categorized_weights, config_json, model_type)

    return yaml_config
```

### 3. 生成 YAML 时保留完整名称

```yaml
# 生成的配置文件示例
embedding:
  embed_tokens.weight: {shape: [129280, 7168], dtype: "BF16", layers: 1}
  model.embed_tokens.weight: {shape: [129280, 7168], dtype: "BF16", layers: 1}
  lm_head.weight: {shape: [129280, 7168], dtype: "BF16", layers: 1}
  shared_head.head.weight: {shape: [129280, 7168], dtype: "BF16", layers: 1}

attention:
  num_layers: 62
  type: "MLA"
  components:
    self_attn.q_a_proj.weight: {shape: [1536, 7168], dtype: "F8_E4M3", layers: 1}
    self_attn.q_a_proj.weight_scale_inv: {shape: [12, 56], dtype: "F32", layers: 1}
    self_attn.q_a_layernorm.weight: {shape: [1536], dtype: "BF16", layers: 1}
    # ... 其他 attention 权重
```

## 总结

### 优点
1. ✅ **简化规则**：基于前缀匹配，不需要为每个子权重单独写规则
2. ✅ **自动扩展**：新的子权重（如 bias, scale）自动支持
3. ✅ **易于维护**：规则清晰，容易理解和修改
4. ✅ **保留完整信息**：不去重，保留所有权重的完整名称

### 改进点
1. ✅ 使用基于前缀的分类策略
2. ✅ 添加排除模式避免误匹配
3. ✅ 保留完整权重名称（不去重）
4. ✅ 支持特殊模块（MTP, 量化 MoE）
