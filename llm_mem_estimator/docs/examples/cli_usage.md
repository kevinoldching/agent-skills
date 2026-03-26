---
python scripts/calculate_mem.py --chip Ascend-910B-64GB --config configs/models/Kimi-K2.5.yaml --find
-max-seq-len --prompt-len 65535 --gen-len 8192 --activation-peak 1.5 --ep 64 --tp 8
Note: --batch-size=1 with --find-max-seq-len, --prompt-len and --gen-len, searching for maximum batch_size...
---
# Kimi-K2.5 - Memory Estimation Report

## Model Information

- **Model Name**: Kimi-K2.5
- **Total Parameters**: 170.74B
- **Number of Layers**: 61
- **Attention Type**: mla
- **FFN Type**: moe
- **Normalization Type**: rmsnorm

## Configuration

- **Batch Size**: 8
- **Prompt Length**: 65535
- **Generated Length**: 8192
- **Tensor Parallel (TP)**: 8
- **Pipeline Parallel (PP)**: 1
- **Data Parallel (DP)**: 1
- **Context Parallel (CP)**: 1
- **Expert Parallel (EP)**: 64

## Hardware Information

- **Chip**: huawei/Ascend-910B-64GB
- **VRAM**: 64 GB
- **Bandwidth**: 1200 GB/s

## Memory Usage Summary

| Component | Memory (GB) | Percentage |
|-----------|-------------|------------|
| Weights | 13.69 | 24.5% |
| KV Cache | 38.60 | 69.2% |
| Activation | 1.50 | 2.7% |
| System Reserved | 2.00 | 3.6% |
| **Total** | **55.79** | **100.0%** |

**Computation Formulas:**

- KV Cache: `batch_size * seq_len * (kv_lora_rank + qk_rope_head_dim) * num_layers / cp_size`
- Activation: `batch_size * seq_len * hidden_size * num_experts_per_tok * recommended_capacity_factor / cp_size`
- recommended_capacity_factor: has_prefill=1.25, decode=12.5

**Calculation Example:**

```
# KV Cache formula: batch_size * seq_len * (kv_lora_rank + qk_rope_head_dim) * num_layers / cp_size
# Substituting: batch_size=8, seq_len=73727, tp_size=8, cp_size=1
# = 38.6010 GB

# Activation formula: batch_size * seq_len * hidden_size * num_experts_per_tok * recommended_capacity_factor / cp_size
# Substituting: batch_size=8, seq_len=73727, hidden_size=7168, num_experts_per_tok=8, cp_size=1
# = 8 * 73727 * 7168 * 8 * 0.02 * 2 / 1
# = 1.500000 GB
```

## Weights Breakdown by Module

**By Module Type:**

| Module Type | Memory (GB) | Percentage |
|-------------|-------------|------------|
| embedding | 0.55 | 4.0% |
| attention | 2.94 | 21.5% |
| ffn_moe | 8.61 | 62.9% |
| ffn_shared_expert | 0.62 | 4.5% |
| ffn_dense | 0.09 | 0.7% |
| norm | 0.00 | 0.0% |
| others | 0.88 | 6.4% |

**Detailed Breakdown:**

| Module Type | Weight Name | Shape | Layers | Memory (GB) | Percentage | Data Type | Parallel Strategy | World Size |
|-------------|-------------|-------|--------|-------------|------------|-----------|-------------------|------------|
| embedding | language_model.lm_head.weight | [163840,7168] | 1 | 0.27344 | 2.00% | BF16 | TP | 8 |
| embedding | language_model.model.embed_tokens.weight | [163840,7168] | 1 | 0.27344 | 2.00% | BF16 | TP | 8 |
| attention | language_model.self_attn.kv_a_layernorm.weight | [512] | 61 | 0.00006 | 0.00% | BF16 | replicated | 1 |
| attention | language_model.self_attn.kv_a_proj_with_mqa.weight | [576,7168] | 61 | 0.46912 | 3.43% | BF16 | replicated | 1 |
| attention | language_model.self_attn.kv_b_proj.weight | [16384,512] | 61 | 0.11914 | 0.87% | BF16 | TP | 8 |
| attention | language_model.self_attn.o_proj.weight | [7168,8192] | 61 | 0.83398 | 6.09% | BF16 | TP | 8 |
| attention | language_model.self_attn.q_a_layernorm.weight | [1536] | 61 | 0.00017 | 0.00% | BF16 | replicated | 1 |
| attention | language_model.self_attn.q_a_proj.weight | [1536,7168] | 61 | 1.25098 | 9.14% | BF16 | replicated | 1 |
| attention | language_model.self_attn.q_b_proj.weight | [12288,1536] | 61 | 0.26807 | 1.96% | BF16 | TP | 8 |
| ffn_moe | language_model.mlp.gate.e_score_correction_bias | [384] | 60 | 0.00009 | 0.00% | F32 | replicated | 1 |
| ffn_moe | language_model.mlp.experts.N.down_proj.weight_packed | [384,7168,256] | 60 | 2.46094 | 17.98% | I32 | EP | 64 |
| ffn_moe | language_model.mlp.experts.N.down_proj.weight_shape | [2] | 60 | 0.00000 | 0.00% | I32 | EP | 64 |
| ffn_moe | language_model.mlp.experts.N.gate_proj.weight_packed | [384,2048,896] | 60 | 2.46094 | 17.98% | I32 | EP | 64 |
| ffn_moe | language_model.mlp.experts.N.gate_proj.weight_shape | [2] | 60 | 0.00000 | 0.00% | I32 | EP | 64 |
| ffn_moe | language_model.mlp.experts.N.up_proj.weight_packed | [384,2048,896] | 60 | 2.46094 | 17.98% | I32 | EP | 64 |
| ffn_moe | language_model.mlp.experts.N.up_proj.weight_shape | [2] | 60 | 0.00000 | 0.00% | I32 | EP | 64 |
| ffn_moe | language_model.mlp.experts.N.down_proj.weight_scale | [384,7168,64] | 60 | 0.30762 | 2.25% | BF16 | EP | 64 |
| ffn_moe | language_model.mlp.experts.N.gate_proj.weight_scale | [384,2048,224] | 60 | 0.30762 | 2.25% | BF16 | EP | 64 |
| ffn_moe | language_model.mlp.experts.N.up_proj.weight_scale | [384,2048,224] | 60 | 0.30762 | 2.25% | BF16 | EP | 64 |
| ffn_moe | language_model.mlp.gate.weight | [384,7168] | 60 | 0.30762 | 2.25% | BF16 | replicated | 1 |
| ffn_shared_expert | language_model.mlp.shared_experts.down_proj.weight | [7168,2048] | 60 | 0.20508 | 1.50% | BF16 | TP | 8 |
| ffn_shared_expert | language_model.mlp.shared_experts.gate_proj.weight | [2048,7168] | 60 | 0.20508 | 1.50% | BF16 | TP | 8 |
| ffn_shared_expert | language_model.mlp.shared_experts.up_proj.weight | [2048,7168] | 60 | 0.20508 | 1.50% | BF16 | TP | 8 |
| ffn_dense | language_model.model.layers.N.mlp.down_proj.weight | [7168,18432] | 1 | 0.03076 | 0.22% | BF16 | TP | 8 |
| ffn_dense | language_model.model.layers.N.mlp.gate_proj.weight | [18432,7168] | 1 | 0.03076 | 0.22% | BF16 | TP | 8 |
| ffn_dense | language_model.model.layers.N.mlp.up_proj.weight | [18432,7168] | 1 | 0.03076 | 0.22% | BF16 | TP | 8 |
| norm | language_model.input_layernorm.weight | [7168] | 61 | 0.00081 | 0.01% | BF16 | replicated | 1 |
| norm | language_model.post_attention_layernorm.weight | [7168] | 61 | 0.00081 | 0.01% | BF16 | replicated | 1 |
| others | mm_projector.pre_norm.bias | [1152] | 1 | 0.00000 | 0.00% | BF16 | replicated | 1 |
| others | mm_projector.pre_norm.weight | [1152] | 1 | 0.00000 | 0.00% | BF16 | replicated | 1 |
| others | mm_projector.proj.N.bias | [4608] | 1 | 0.00001 | 0.00% | BF16 | replicated | 1 |
| others | mm_projector.proj.N.weight | [4608,4608] | 1 | 0.03955 | 0.29% | BF16 | replicated | 1 |
| others | mm_projector.proj.N.bias | [7168] | 1 | 0.00001 | 0.00% | BF16 | replicated | 1 |
| others | mm_projector.proj.N.weight | [7168,4608] | 1 | 0.06152 | 0.45% | BF16 | replicated | 1 |
| others | language_model.model.norm.weight | [7168] | 1 | 0.00001 | 0.00% | BF16 | replicated | 1 |
| others | vision_tower.encoder.blocks.N.mlp.fc0.bias | [4304] | 27 | 0.00022 | 0.00% | BF16 | replicated | 1 |
| others | vision_tower.encoder.blocks.N.mlp.fc0.weight | [4304,1152] | 27 | 0.24936 | 1.82% | BF16 | replicated | 1 |
| others | vision_tower.encoder.blocks.N.mlp.fc1.bias | [1152] | 27 | 0.00006 | 0.00% | BF16 | replicated | 1 |
| others | vision_tower.encoder.blocks.N.mlp.fc1.weight | [1152,4304] | 27 | 0.24936 | 1.82% | BF16 | replicated | 1 |
| others | vision_tower.encoder.blocks.N.norm0.bias | [1152] | 27 | 0.00006 | 0.00% | BF16 | replicated | 1 |
| others | vision_tower.encoder.blocks.N.norm0.weight | [1152] | 27 | 0.00006 | 0.00% | BF16 | replicated | 1 |
| others | vision_tower.encoder.blocks.N.norm1.bias | [1152] | 27 | 0.00006 | 0.00% | BF16 | replicated | 1 |
| others | vision_tower.encoder.blocks.N.norm1.weight | [1152] | 27 | 0.00006 | 0.00% | BF16 | replicated | 1 |
| others | vision_tower.encoder.blocks.N.wo.bias | [1152] | 27 | 0.00006 | 0.00% | BF16 | replicated | 1 |
| others | vision_tower.encoder.blocks.N.wo.weight | [1152,1152] | 27 | 0.06674 | 0.49% | BF16 | replicated | 1 |
| others | vision_tower.encoder.blocks.N.wqkv.bias | [3456] | 27 | 0.00017 | 0.00% | BF16 | replicated | 1 |
| others | vision_tower.encoder.blocks.N.wqkv.weight | [3456,1152] | 27 | 0.20023 | 1.46% | BF16 | replicated | 1 |
| others | vision_tower.encoder.final_layernorm.bias | [1152] | 1 | 0.00000 | 0.00% | BF16 | replicated | 1 |
| others | vision_tower.encoder.final_layernorm.weight | [1152] | 1 | 0.00000 | 0.00% | BF16 | replicated | 1 |
| others | vision_tower.patch_embed.pos_emb.weight | [64,64,1152] | 1 | 0.00879 | 0.06% | BF16 | replicated | 1 |
| others | vision_tower.patch_embed.proj.bias | [1152] | 1 | 0.00000 | 0.00% | BF16 | replicated | 1 |
| others | vision_tower.patch_embed.proj.weight | [1152,3,14,14] | 1 | 0.00126 | 0.01% | BF16 | replicated | 1 |
| **Total** | - | - | - | **13.68849** | **100.00%** | - | - | - |


## Result
- **Maximum batch size: 8**
- VRAM * gpu_util - Total = 64 * 90% - 55.79 = 57.60 - 55.79 = 1.81 GB
- Status: ✅ Fits