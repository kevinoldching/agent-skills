# Qwen/Qwen2.5-0.5B 权重元数据

**总权重数量**: 290


## Architecture Config (from config.json)

| 字段 | 值 |
|------|-----|
| model_type | qwen2 |
| hidden_size | 896 |
| num_hidden_layers | 24 |
| vocab_size | 151936 |
| num_attention_heads | 14 |
| num_key_value_heads | 2 |
| intermediate_size | 4864 |

## 按模块类型统计

| 模块类型 | 权重数量 |
|----------|----------|
| attention | 168 |
| ffn_dense | 72 |
| norm | 49 |
| embedding | 1 |

## 权重详细列表

### Embedding / Output Norm

- `model.embed_tokens.weight`  151936 x 896  (BF16)
- `model.norm.weight`  896  (BF16)

### Layer Weights (示例)

#### Layer 0

- `model.layers.0.input_layernorm.weight`  896  (BF16)  → **norm**
- `model.layers.0.mlp.down_proj.weight`  896 x 4864  (BF16)  → **ffn_dense**
- `model.layers.0.mlp.gate_proj.weight`  4864 x 896  (BF16)  → **ffn_dense**
- `model.layers.0.mlp.up_proj.weight`  4864 x 896  (BF16)  → **ffn_dense**
- `model.layers.0.post_attention_layernorm.weight`  896  (BF16)  → **norm**
- `model.layers.0.self_attn.k_proj.bias`  128  (BF16)  → **attention**
- `model.layers.0.self_attn.k_proj.weight`  128 x 896  (BF16)  → **attention**
- `model.layers.0.self_attn.o_proj.weight`  896 x 896  (BF16)  → **attention**
- `model.layers.0.self_attn.q_proj.bias`  896  (BF16)  → **attention**
- `model.layers.0.self_attn.q_proj.weight`  896 x 896  (BF16)  → **attention**
- `model.layers.0.self_attn.v_proj.bias`  128  (BF16)  → **attention**
- `model.layers.0.self_attn.v_proj.weight`  128 x 896  (BF16)  → **attention**

#### Layer 1

- `model.layers.1.input_layernorm.weight`  896  (BF16)  → **norm**
- `model.layers.1.mlp.down_proj.weight`  896 x 4864  (BF16)  → **ffn_dense**
- `model.layers.1.mlp.gate_proj.weight`  4864 x 896  (BF16)  → **ffn_dense**
- `model.layers.1.mlp.up_proj.weight`  4864 x 896  (BF16)  → **ffn_dense**
- `model.layers.1.post_attention_layernorm.weight`  896  (BF16)  → **norm**
- `model.layers.1.self_attn.k_proj.bias`  128  (BF16)  → **attention**
- `model.layers.1.self_attn.k_proj.weight`  128 x 896  (BF16)  → **attention**
- `model.layers.1.self_attn.o_proj.weight`  896 x 896  (BF16)  → **attention**
- `model.layers.1.self_attn.q_proj.bias`  896  (BF16)  → **attention**
- `model.layers.1.self_attn.q_proj.weight`  896 x 896  (BF16)  → **attention**
- `model.layers.1.self_attn.v_proj.bias`  128  (BF16)  → **attention**
- `model.layers.1.self_attn.v_proj.weight`  128 x 896  (BF16)  → **attention**

#### Layer 22

- `model.layers.22.input_layernorm.weight`  896  (BF16)  → **norm**
- `model.layers.22.mlp.down_proj.weight`  896 x 4864  (BF16)  → **ffn_dense**
- `model.layers.22.mlp.gate_proj.weight`  4864 x 896  (BF16)  → **ffn_dense**
- `model.layers.22.mlp.up_proj.weight`  4864 x 896  (BF16)  → **ffn_dense**
- `model.layers.22.post_attention_layernorm.weight`  896  (BF16)  → **norm**
- `model.layers.22.self_attn.k_proj.bias`  128  (BF16)  → **attention**
- `model.layers.22.self_attn.k_proj.weight`  128 x 896  (BF16)  → **attention**
- `model.layers.22.self_attn.o_proj.weight`  896 x 896  (BF16)  → **attention**
- `model.layers.22.self_attn.q_proj.bias`  896  (BF16)  → **attention**
- `model.layers.22.self_attn.q_proj.weight`  896 x 896  (BF16)  → **attention**
- `model.layers.22.self_attn.v_proj.bias`  128  (BF16)  → **attention**
- `model.layers.22.self_attn.v_proj.weight`  128 x 896  (BF16)  → **attention**

#### Layer 23

- `model.layers.23.input_layernorm.weight`  896  (BF16)  → **norm**
- `model.layers.23.mlp.down_proj.weight`  896 x 4864  (BF16)  → **ffn_dense**
- `model.layers.23.mlp.gate_proj.weight`  4864 x 896  (BF16)  → **ffn_dense**
- `model.layers.23.mlp.up_proj.weight`  4864 x 896  (BF16)  → **ffn_dense**
- `model.layers.23.post_attention_layernorm.weight`  896  (BF16)  → **norm**
- `model.layers.23.self_attn.k_proj.bias`  128  (BF16)  → **attention**
- `model.layers.23.self_attn.k_proj.weight`  128 x 896  (BF16)  → **attention**
- `model.layers.23.self_attn.o_proj.weight`  896 x 896  (BF16)  → **attention**
- `model.layers.23.self_attn.q_proj.bias`  896  (BF16)  → **attention**
- `model.layers.23.self_attn.q_proj.weight`  896 x 896  (BF16)  → **attention**
- `model.layers.23.self_attn.v_proj.bias`  128  (BF16)  → **attention**
- `model.layers.23.self_attn.v_proj.weight`  128 x 896  (BF16)  → **attention**
