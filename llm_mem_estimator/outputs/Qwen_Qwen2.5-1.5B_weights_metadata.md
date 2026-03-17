# Qwen/Qwen2.5-1.5B 权重元数据

**总权重数量**: 338


## Architecture Config (from config.json)

| 字段 | 值 |
|------|-----|
| model_type | qwen2 |
| hidden_size | 1536 |
| num_hidden_layers | 28 |
| vocab_size | 151936 |
| num_attention_heads | 12 |
| num_key_value_heads | 2 |
| intermediate_size | 8960 |

## 按模块类型统计

| 模块类型 | 权重数量 |
|----------|----------|
| attention | 196 |
| ffn_dense | 84 |
| norm | 57 |
| embedding | 1 |

## 权重详细列表

### Embedding / Output Norm

- `model.embed_tokens.weight`  151936 x 1536  (BF16)
- `model.norm.weight`  1536  (BF16)

### Layer Weights (示例)

#### Layer 0

- `model.layers.0.input_layernorm.weight`  1536  (BF16)  → **norm**
- `model.layers.0.mlp.down_proj.weight`  1536 x 8960  (BF16)  → **ffn_dense**
- `model.layers.0.mlp.gate_proj.weight`  8960 x 1536  (BF16)  → **ffn_dense**
- `model.layers.0.mlp.up_proj.weight`  8960 x 1536  (BF16)  → **ffn_dense**
- `model.layers.0.post_attention_layernorm.weight`  1536  (BF16)  → **norm**
- `model.layers.0.self_attn.k_proj.bias`  256  (BF16)  → **attention**
- `model.layers.0.self_attn.k_proj.weight`  256 x 1536  (BF16)  → **attention**
- `model.layers.0.self_attn.o_proj.weight`  1536 x 1536  (BF16)  → **attention**
- `model.layers.0.self_attn.q_proj.bias`  1536  (BF16)  → **attention**
- `model.layers.0.self_attn.q_proj.weight`  1536 x 1536  (BF16)  → **attention**
- `model.layers.0.self_attn.v_proj.bias`  256  (BF16)  → **attention**
- `model.layers.0.self_attn.v_proj.weight`  256 x 1536  (BF16)  → **attention**

#### Layer 1

- `model.layers.1.input_layernorm.weight`  1536  (BF16)  → **norm**
- `model.layers.1.mlp.down_proj.weight`  1536 x 8960  (BF16)  → **ffn_dense**
- `model.layers.1.mlp.gate_proj.weight`  8960 x 1536  (BF16)  → **ffn_dense**
- `model.layers.1.mlp.up_proj.weight`  8960 x 1536  (BF16)  → **ffn_dense**
- `model.layers.1.post_attention_layernorm.weight`  1536  (BF16)  → **norm**
- `model.layers.1.self_attn.k_proj.bias`  256  (BF16)  → **attention**
- `model.layers.1.self_attn.k_proj.weight`  256 x 1536  (BF16)  → **attention**
- `model.layers.1.self_attn.o_proj.weight`  1536 x 1536  (BF16)  → **attention**
- `model.layers.1.self_attn.q_proj.bias`  1536  (BF16)  → **attention**
- `model.layers.1.self_attn.q_proj.weight`  1536 x 1536  (BF16)  → **attention**
- `model.layers.1.self_attn.v_proj.bias`  256  (BF16)  → **attention**
- `model.layers.1.self_attn.v_proj.weight`  256 x 1536  (BF16)  → **attention**

#### Layer 26

- `model.layers.26.input_layernorm.weight`  1536  (BF16)  → **norm**
- `model.layers.26.mlp.down_proj.weight`  1536 x 8960  (BF16)  → **ffn_dense**
- `model.layers.26.mlp.gate_proj.weight`  8960 x 1536  (BF16)  → **ffn_dense**
- `model.layers.26.mlp.up_proj.weight`  8960 x 1536  (BF16)  → **ffn_dense**
- `model.layers.26.post_attention_layernorm.weight`  1536  (BF16)  → **norm**
- `model.layers.26.self_attn.k_proj.bias`  256  (BF16)  → **attention**
- `model.layers.26.self_attn.k_proj.weight`  256 x 1536  (BF16)  → **attention**
- `model.layers.26.self_attn.o_proj.weight`  1536 x 1536  (BF16)  → **attention**
- `model.layers.26.self_attn.q_proj.bias`  1536  (BF16)  → **attention**
- `model.layers.26.self_attn.q_proj.weight`  1536 x 1536  (BF16)  → **attention**
- `model.layers.26.self_attn.v_proj.bias`  256  (BF16)  → **attention**
- `model.layers.26.self_attn.v_proj.weight`  256 x 1536  (BF16)  → **attention**

#### Layer 27

- `model.layers.27.input_layernorm.weight`  1536  (BF16)  → **norm**
- `model.layers.27.mlp.down_proj.weight`  1536 x 8960  (BF16)  → **ffn_dense**
- `model.layers.27.mlp.gate_proj.weight`  8960 x 1536  (BF16)  → **ffn_dense**
- `model.layers.27.mlp.up_proj.weight`  8960 x 1536  (BF16)  → **ffn_dense**
- `model.layers.27.post_attention_layernorm.weight`  1536  (BF16)  → **norm**
- `model.layers.27.self_attn.k_proj.bias`  256  (BF16)  → **attention**
- `model.layers.27.self_attn.k_proj.weight`  256 x 1536  (BF16)  → **attention**
- `model.layers.27.self_attn.o_proj.weight`  1536 x 1536  (BF16)  → **attention**
- `model.layers.27.self_attn.q_proj.bias`  1536  (BF16)  → **attention**
- `model.layers.27.self_attn.q_proj.weight`  1536 x 1536  (BF16)  → **attention**
- `model.layers.27.self_attn.v_proj.bias`  256  (BF16)  → **attention**
- `model.layers.27.self_attn.v_proj.weight`  256 x 1536  (BF16)  → **attention**
