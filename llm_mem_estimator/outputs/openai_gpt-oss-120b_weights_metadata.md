# openai/gpt-oss-120b 权重元数据

**总权重数量**: 687


## Architecture Config (from config.json)

| 字段 | 值 |
|------|-----|
| model_type | gpt_oss |
| hidden_size | 2880 |
| num_hidden_layers | 36 |
| vocab_size | 201088 |
| num_attention_heads | 64 |
| num_key_value_heads | 8 |
| intermediate_size | 2880 |
| num_experts_per_tok | 4 |

## 按模块类型统计

| 模块类型 | 权重数量 |
|----------|----------|
| attention | 324 |
| ffn_moe | 288 |
| norm | 73 |
| embedding | 2 |

## 权重详细列表

### Embedding / Output Norm

- `lm_head.weight`  201088 x 2880  (BF16)
- `model.embed_tokens.weight`  201088 x 2880  (BF16)
- `model.norm.weight`  2880  (BF16)

### Layer Weights (示例)

#### Layer 0

- `model.layers.0.input_layernorm.weight`  2880  (BF16)  → **norm**
- `model.layers.0.mlp.experts.down_proj_bias`  128 x 2880  (BF16)  → **ffn_moe**
- `model.layers.0.mlp.experts.down_proj_blocks`  128 x 2880 x 90 x 16  (U8)  → **ffn_moe**
- `model.layers.0.mlp.experts.down_proj_scales`  128 x 2880 x 90  (U8)  → **ffn_moe**
- `model.layers.0.mlp.experts.gate_up_proj_bias`  128 x 5760  (BF16)  → **ffn_moe**
- `model.layers.0.mlp.experts.gate_up_proj_blocks`  128 x 5760 x 90 x 16  (U8)  → **ffn_moe**
- `model.layers.0.mlp.experts.gate_up_proj_scales`  128 x 5760 x 90  (U8)  → **ffn_moe**
- `model.layers.0.mlp.router.bias`  128  (BF16)  → **ffn_moe**
- `model.layers.0.mlp.router.weight`  128 x 2880  (BF16)  → **ffn_moe**
- `model.layers.0.post_attention_layernorm.weight`  2880  (BF16)  → **norm**
- `model.layers.0.self_attn.k_proj.bias`  512  (BF16)  → **attention**
- `model.layers.0.self_attn.k_proj.weight`  512 x 2880  (BF16)  → **attention**
- `model.layers.0.self_attn.o_proj.bias`  2880  (BF16)  → **attention**
- `model.layers.0.self_attn.o_proj.weight`  2880 x 4096  (BF16)  → **attention**
- `model.layers.0.self_attn.q_proj.bias`  4096  (BF16)  → **attention**
- `model.layers.0.self_attn.q_proj.weight`  4096 x 2880  (BF16)  → **attention**
- `model.layers.0.self_attn.sinks`  64  (BF16)  → **attention**
- `model.layers.0.self_attn.v_proj.bias`  512  (BF16)  → **attention**
- `model.layers.0.self_attn.v_proj.weight`  512 x 2880  (BF16)  → **attention**

#### Layer 1

- `model.layers.1.input_layernorm.weight`  2880  (BF16)  → **norm**
- `model.layers.1.mlp.experts.down_proj_bias`  128 x 2880  (BF16)  → **ffn_moe**
- `model.layers.1.mlp.experts.down_proj_blocks`  128 x 2880 x 90 x 16  (U8)  → **ffn_moe**
- `model.layers.1.mlp.experts.down_proj_scales`  128 x 2880 x 90  (U8)  → **ffn_moe**
- `model.layers.1.mlp.experts.gate_up_proj_bias`  128 x 5760  (BF16)  → **ffn_moe**
- `model.layers.1.mlp.experts.gate_up_proj_blocks`  128 x 5760 x 90 x 16  (U8)  → **ffn_moe**
- `model.layers.1.mlp.experts.gate_up_proj_scales`  128 x 5760 x 90  (U8)  → **ffn_moe**
- `model.layers.1.mlp.router.bias`  128  (BF16)  → **ffn_moe**
- `model.layers.1.mlp.router.weight`  128 x 2880  (BF16)  → **ffn_moe**
- `model.layers.1.post_attention_layernorm.weight`  2880  (BF16)  → **norm**
- `model.layers.1.self_attn.k_proj.bias`  512  (BF16)  → **attention**
- `model.layers.1.self_attn.k_proj.weight`  512 x 2880  (BF16)  → **attention**
- `model.layers.1.self_attn.o_proj.bias`  2880  (BF16)  → **attention**
- `model.layers.1.self_attn.o_proj.weight`  2880 x 4096  (BF16)  → **attention**
- `model.layers.1.self_attn.q_proj.bias`  4096  (BF16)  → **attention**
- `model.layers.1.self_attn.q_proj.weight`  4096 x 2880  (BF16)  → **attention**
- `model.layers.1.self_attn.sinks`  64  (BF16)  → **attention**
- `model.layers.1.self_attn.v_proj.bias`  512  (BF16)  → **attention**
- `model.layers.1.self_attn.v_proj.weight`  512 x 2880  (BF16)  → **attention**

#### Layer 34

- `model.layers.34.input_layernorm.weight`  2880  (BF16)  → **norm**
- `model.layers.34.mlp.experts.down_proj_bias`  128 x 2880  (BF16)  → **ffn_moe**
- `model.layers.34.mlp.experts.down_proj_blocks`  128 x 2880 x 90 x 16  (U8)  → **ffn_moe**
- `model.layers.34.mlp.experts.down_proj_scales`  128 x 2880 x 90  (U8)  → **ffn_moe**
- `model.layers.34.mlp.experts.gate_up_proj_bias`  128 x 5760  (BF16)  → **ffn_moe**
- `model.layers.34.mlp.experts.gate_up_proj_blocks`  128 x 5760 x 90 x 16  (U8)  → **ffn_moe**
- `model.layers.34.mlp.experts.gate_up_proj_scales`  128 x 5760 x 90  (U8)  → **ffn_moe**
- `model.layers.34.mlp.router.bias`  128  (BF16)  → **ffn_moe**
- `model.layers.34.mlp.router.weight`  128 x 2880  (BF16)  → **ffn_moe**
- `model.layers.34.post_attention_layernorm.weight`  2880  (BF16)  → **norm**
- `model.layers.34.self_attn.k_proj.bias`  512  (BF16)  → **attention**
- `model.layers.34.self_attn.k_proj.weight`  512 x 2880  (BF16)  → **attention**
- `model.layers.34.self_attn.o_proj.bias`  2880  (BF16)  → **attention**
- `model.layers.34.self_attn.o_proj.weight`  2880 x 4096  (BF16)  → **attention**
- `model.layers.34.self_attn.q_proj.bias`  4096  (BF16)  → **attention**
- `model.layers.34.self_attn.q_proj.weight`  4096 x 2880  (BF16)  → **attention**
- `model.layers.34.self_attn.sinks`  64  (BF16)  → **attention**
- `model.layers.34.self_attn.v_proj.bias`  512  (BF16)  → **attention**
- `model.layers.34.self_attn.v_proj.weight`  512 x 2880  (BF16)  → **attention**

#### Layer 35

- `model.layers.35.input_layernorm.weight`  2880  (BF16)  → **norm**
- `model.layers.35.mlp.experts.down_proj_bias`  128 x 2880  (BF16)  → **ffn_moe**
- `model.layers.35.mlp.experts.down_proj_blocks`  128 x 2880 x 90 x 16  (U8)  → **ffn_moe**
- `model.layers.35.mlp.experts.down_proj_scales`  128 x 2880 x 90  (U8)  → **ffn_moe**
- `model.layers.35.mlp.experts.gate_up_proj_bias`  128 x 5760  (BF16)  → **ffn_moe**
- `model.layers.35.mlp.experts.gate_up_proj_blocks`  128 x 5760 x 90 x 16  (U8)  → **ffn_moe**
- `model.layers.35.mlp.experts.gate_up_proj_scales`  128 x 5760 x 90  (U8)  → **ffn_moe**
- `model.layers.35.mlp.router.bias`  128  (BF16)  → **ffn_moe**
- `model.layers.35.mlp.router.weight`  128 x 2880  (BF16)  → **ffn_moe**
- `model.layers.35.post_attention_layernorm.weight`  2880  (BF16)  → **norm**
- `model.layers.35.self_attn.k_proj.bias`  512  (BF16)  → **attention**
- `model.layers.35.self_attn.k_proj.weight`  512 x 2880  (BF16)  → **attention**
- `model.layers.35.self_attn.o_proj.bias`  2880  (BF16)  → **attention**
- `model.layers.35.self_attn.o_proj.weight`  2880 x 4096  (BF16)  → **attention**
- `model.layers.35.self_attn.q_proj.bias`  4096  (BF16)  → **attention**
- `model.layers.35.self_attn.q_proj.weight`  4096 x 2880  (BF16)  → **attention**
- `model.layers.35.self_attn.sinks`  64  (BF16)  → **attention**
- `model.layers.35.self_attn.v_proj.bias`  512  (BF16)  → **attention**
- `model.layers.35.self_attn.v_proj.weight`  512 x 2880  (BF16)  → **attention**
