# gpt-oss-120b - Memory Estimation Report

## Model Information

- **Model Name**: gpt-oss-120b
- **Total Parameters**: 63.08B
- **Number of Layers**: 36
- **Quantization**: mxfp4
- **Attention Type**: gqa
- **FFN Type**: moe
- **Normalization Type**: rmsnorm

## Configuration

- **Batch Size**: 1
- **Sequence Length**: 2048
- **Tensor Parallel (TP)**: 8
- **Pipeline Parallel (PP)**: 1
- **Data Parallel (DP)**: 1
- **Context Parallel (CP)**: 1
- **Expert Parallel (EP)**: 8

## Memory Usage Summary

| Component | Memory (GB) | Percentage |
|-----------|-------------|------------|
| Weights | 7.62 | 79.0% |
| KV Cache | 0.02 | 0.2% |
| Activation | 0.01 | 0.1% |
| System Reserved | 2.00 | 20.7% |
| **Total** | **9.64** | **100.0%** |

**计算公式说明：**

- KV Cache: `2 * batch_size * seq_len * 512 * num_layers`
- Activation: `batch_size * seq_len * hidden_size * 4 * 1.25 * dtype_bytes`

## Weights Breakdown by Module

| Module Type | Weight Name | Shape | Layers | Memory (GB) | Percentage | Data Type | Parallel Strategy | World Size |
|-------------|-------------|-------|--------|-------------|------------|-----------|-------------------|------------|
| embedding | lm_head.weight | [201088,2880] | 1 | 0.1348 | 1.8% | BF16 | TP | 8 |
| embedding | model.embed_tokens.weight | [201088,2880] | 1 | 0.1348 | 1.8% | BF16 | TP | 8 |
| attention | self_attn.k_proj.bias | [512] | 36 | 0.0000 | 0.0% | BF16 | replicated | 1 |
| attention | self_attn.k_proj.weight | [512,2880] | 36 | 0.0124 | 0.2% | BF16 | TP | 8 |
| attention | self_attn.o_proj.bias | [2880] | 36 | 0.0002 | 0.0% | BF16 | replicated | 1 |
| attention | self_attn.o_proj.weight | [2880,4096] | 36 | 0.0989 | 1.3% | BF16 | TP | 8 |
| attention | self_attn.q_proj.bias | [4096] | 36 | 0.0003 | 0.0% | BF16 | replicated | 1 |
| attention | self_attn.q_proj.weight | [4096,2880] | 36 | 0.0989 | 1.3% | BF16 | TP | 8 |
| attention | self_attn.sinks | [64] | 36 | 0.0000 | 0.0% | BF16 | replicated | 1 |
| attention | self_attn.v_proj.bias | [512] | 36 | 0.0000 | 0.0% | BF16 | replicated | 1 |
| attention | self_attn.v_proj.weight | [512,2880] | 36 | 0.0124 | 0.2% | BF16 | TP | 8 |
| ffn_moe | mlp.experts.down_proj_bias | [128,2880] | 36 | 0.0031 | 0.0% | BF16 | EP | 8 |
| ffn_moe | mlp.experts.gate_up_proj_bias | [128,5760] | 36 | 0.0062 | 0.1% | BF16 | EP | 8 |
| ffn_moe | mlp.router.bias | [128] | 36 | 0.0000 | 0.0% | BF16 | replicated | 1 |
| ffn_moe | mlp.router.weight | [128,2880] | 36 | 0.0247 | 0.3% | BF16 | replicated | 1 |
| ffn_moe | mlp.experts.down_proj_blocks | [128,2880,90,16] | 36 | 2.2247 | 29.2% | U8 | EP | 8 |
| ffn_moe | mlp.experts.down_proj_scales | [128,2880,90] | 36 | 0.1390 | 1.8% | U8 | EP | 8 |
| ffn_moe | mlp.experts.gate_up_proj_blocks | [128,5760,90,16] | 36 | 4.4495 | 58.4% | U8 | EP | 8 |
| ffn_moe | mlp.experts.gate_up_proj_scales | [128,5760,90] | 36 | 0.2781 | 3.7% | U8 | EP | 8 |
| norm | post_attention_layernorm.weight | [2880] | 36 | 0.0002 | 0.0% | BF16 | replicated | 1 |
| norm | input_layernorm.weight | [2880] | 36 | 0.0002 | 0.0% | BF16 | replicated | 1 |
| norm | model.norm.weight | [2880] | 1 | 0.0000 | 0.0% | BF16 | replicated | 1 |
| **Total** | - | - | - | **7.6184** | **100.0%** | - | - | - |
