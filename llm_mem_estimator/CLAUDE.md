# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an LLM memory estimator tool designed to calculate GPU memory usage for large language models. It estimates memory consumption for model weights, KV Cache, and activation values, and predicts maximum supported sequence length and batch size for given hardware configurations.

**Current Status**: Early-stage project with detailed planning documentation. Implementation is pending.

## Architecture

### Core Design Principles

1. **Modular Architecture**: Attention, FFN, and Norm modules are independently implemented
2. **Configuration-Driven**: Each model is defined in a YAML config file (`configs/models/<model_name>.yaml`)
3. **Plugin-Style Registration**: Module types can be added through registration mechanism
4. **Extensible**: Supports adding custom fields to config files without breaking compatibility

### Supported Model Components

**Attention Types**:
- MHA (Multi-Head Attention)
- MQA (Multi-Query Attention)
- GQA (Grouped-Query Attention)
- SWA (Sliding Window Attention)
- MLA (Multi-Latent Attention) - DeepSeek-style with LoRA compression
- DSA (DeepSeek Attention)
- GDN (Generalized Delta Networks)

**FFN Types**:
- Standard FFN
- SwiGLU
- MoE (Mixture of Experts) with router and shared experts

**Normalization Types**:
- LayerNorm
- RMSNorm
- DeepNorm

### Parallelization Strategies

The tool accounts for memory distribution across:
- TP (Tensor Parallel): Column/row sharding
- PP (Pipeline Parallel): Layer sharding
- DP (Data Parallel): Batch replication
- EP (Expert Parallel): Expert sharding for MoE
- SP (Sequence Parallel): Sequence sharding
- CP (Context Parallel): Context sharding

## Memory Calculation Formulas

### Total Memory Formula
```
Total Memory = Weights + Activations (peak) + KV Cache + System Reserved + Other
```

### Component Breakdown

**1. Model Weights**:
```
Weight Memory = Σ(module_params × dtype_bytes) / TP / EP / DP
```

**2. Activations (Peak)**:
```
Activation Memory = peak_factor × batch_size × seq_len × hidden_size × num_layers × dtype_bytes / TP / CP
```
- Peak factor: typically 4-8 depending on model architecture

**3. KV Cache**:
```
KV Cache = 2 × batch_size × seq_len × kv_dim × num_layers × dtype_bytes / TP / CP
```
- `kv_dim` varies by attention type (e.g., MLA uses compressed LoRA rank)

**4. System Reserved**:
```
System Reserved = fixed_value (e.g., 2GB) OR total_vram × reserve_ratio (e.g., 5%)
```

### Data Type Sizes
- FP32: 4 bytes
- FP16/BF16: 2 bytes
- FP8/INT8: 1 byte
- INT4: 0.5 bytes

## Configuration File Structure

Model configs use YAML format with three main sections:

1. **model_identity**: Model name, total params, layers, quantization
2. **architecture_config**: Core parameters (hidden_size, num_layers, attention config, MoE config)
3. **modules**: Detailed module definitions with shapes, dtypes, and parallel strategies
4. **computation_rules**: Formulas for memory calculation

Example parallel strategies:
- `replicated`: Full replication, no sharding
- `tp_col`: Tensor parallel, column-wise sharding
- `tp_row`: Tensor parallel, row-wise sharding
- `expert_sharded`: Expert parallel for MoE
- `pp`: Pipeline parallel, layer-wise sharding

## Planned Directory Structure

```
llm_mem_estimator/
├── SKILL.md              # Skill interface definition (to be created)
├── CLAUDE.md             # This file
├── docs/
│   └── plan.md           # Detailed implementation plan (Chinese)
├── configs/
│   ├── models/           # Model YAML configs (to be created)
│   └── chips.json        # Hardware specs (to be created)
└── scripts/
    └── calculate_mem.py  # Core calculation script (to be created)
```

## Implementation Notes

### When Implementing calculate_mem.py

1. **No Parameter Inference**: The tool must NOT auto-infer any parameters. Users must provide a complete YAML config file or the tool should error.

2. **Modular Design**: Implement separate classes for:
   - Attention modules (MHA, MQA, GQA, SWA, MLA, DSA, GDN)
   - FFN modules (Standard, SwiGLU, MoE)
   - Norm modules (LayerNorm, RMSNorm, DeepNorm)

3. **Config Validation**: Validate that all required fields for the specified module types are present in the config.

4. **Parallel Strategy Handling**: Apply parallel strategy factors correctly when calculating per-device memory.

5. **Output Format**: Provide detailed breakdown showing:
   - Total memory usage (GB)
   - Component breakdown (weights, KV cache, activations)
   - Maximum sequence length and batch size estimation
   - Per-device memory for distributed setups

### Hardware Configuration

Create `configs/chips.json` with common chip specs:
- NVIDIA: H100 (80GB/141GB), A100 (80GB), RTX 4090, etc.
- Huawei Ascend: 910B (64GB/32GB)
- AMD: MI300X, MI350X
- Intel Gaudi: Gaudi2, Gaudi3

Each entry should include:
- `vram_gb`: Memory capacity
- `bandwidth_gb_s`: Memory bandwidth

## Related Skills

This skill is part of the agent-skills repository alongside:
- `context-length-estimator`: Estimates maximum context length for given hardware (similar but simpler)
- `code_review_batching`: PR code review tool for omni-npu project
- `design/rat`: Requirements analysis and reporting tool
- `gitee_ops`: Gitee operations skill

## Key Differences from context-length-estimator

While `context-length-estimator` provides quick estimates using simplified formulas, `llm_mem_estimator` is designed for:
- More accurate memory estimation with detailed module-level accounting
- Support for complex architectures (MLA, DSA, MoE with routing)
- Explicit parallelization strategy modeling
- Configuration-driven approach for reproducibility
- Industrial-grade specifications (e.g., DeepSeek-R1 with FP8 blockwise quantization)
