---
name: llm-memory-estimator
description: |
  LLM GPU Memory Estimator - Estimate GPU memory usage for Large Language Models.
  Use this skill whenever the user wants to:
  - Estimate GPU memory usage for LLM model weights, KV cache, and activations
  - Find maximum sequence length supported by specific hardware (e.g., H100-80GB)
  - Analyze memory breakdown for different parallel strategies (TP, PP, DP, CP, EP)
  - Handle Prefill/Decode (PD) separation scenarios
  - Generate YAML config from HuggingFace models
  - Compare memory usage across different chips, models, or parallel configurations

  This skill is for LLM inference memory estimation only. Do not use for training memory estimation.
---

# LLM Memory Estimator Skill

This skill helps estimate GPU memory usage for Large Language Models (LLMs) including weights, KV cache, and activations. It supports Prefill/Decode separation scenarios and various parallel strategies.

## Project Structure

```
llm_mem_estimator/
├── scripts/
│   └── calculate_mem.py      # Main CLI tool
├── configs/
│   ├── models/               # Model YAML configs
│   │   ├── gpt-oss-120b.yaml
│   │   ├── DeepSeek-V3.yaml
│   │   └── ...
│   ├── weight_mapping_rules.yaml
│   └── chips.json            # Hardware specs
├── llm_mem_estimator/
│   ├── memory_estimator.py   # Core estimation logic
│   ├── model_config.py       # Config handling
│   ├── model_detector.py     # HuggingFace model detection
│   └── report_generator.py   # Report generation
└── docs/spec/
    ├── calculate_mem_cli_spec.md
    ├── yaml_config_spec.md
    └── weight_mapping_rules_spec.md
```

## Core Capabilities

### 1. Memory Estimation

Calculate total GPU memory usage:
- **Model Weights**: Based on model parameters and dtype
- **KV Cache**: For attention key-value storage
- **Activations**: Temporary computation memory
- **System Reserved**: Fixed overhead (default 2GB)

### 2. Parallel Strategy Support

| Strategy | Description | CLI Flag |
|----------|-------------|----------|
| TP | Tensor Parallel (column/row sharding) | `--tp` |
| PP | Pipeline Parallel (layer sharding) | `--pp` |
| DP | Data Parallel (replication) | `--dp` |
| CP | Context Parallel (sequence sharding) | `--cp` |
| EP | Expert Parallel (MoE expert sharding) | `--ep` |

### 3. PD Separation Scenarios

**Prefill Stage**: Process input prompt
- Uses `has_prefill` factor (1.25)
- Activation scales with `prompt_len + gen_len`

**Decode Stage**: Generate output tokens
- Uses `decode` factor (12.5)
- Activation is fixed (seq_len=1)

### 4. Hardware Support

Supported chips in `configs/chips.json`:
- NVIDIA: H100-80GB, H100-141GB, A100-80GB, RTX-4090
- Huawei: Ascend-910B-64GB, Ascend-910B-32GB
- AMD: MI300X, MI350X
- Intel: Gaudi2, Gaudi3

## CLI Usage

### Basic Memory Estimation

```bash
# Using pre-defined config
python scripts/calculate_mem.py \
    --config configs/models/gpt-oss-120b.yaml \
    --batch-size 4 \
    --prompt-len 4096 \
    --gen-len 1024 \
    --tp 4

# From HuggingFace model
python scripts/calculate_mem.py \
    --model Qwen/Qwen2.5-1.5B \
    --batch-size 1 \
    --prompt-len 4096 \
    --gen-len 1024
```

### Find Maximum Sequence Length

```bash
# Find max gen_len (Decode scenario, fixed prompt_len)
python scripts/calculate_mem.py \
    --config configs/models/gpt-oss-120b.yaml \
    --chip H100-80GB \
    --find-max-seq-len \
    --prompt-len 4096 \
    --tp 8

# Find max prompt_len (Prefill scenario, fixed gen_len)
python scripts/calculate_mem.py \
    --config configs/models/gpt-oss-120b.yaml \
    --chip H100-80GB \
    --find-max-seq-len \
    --gen-len 1 \
    --tp 8
```

### Generate Model Config

```bash
# Generate YAML config from HuggingFace model
python scripts/calculate_mem.py --model deepseek-ai/DeepSeek-V3 --generate-config

# Save to custom location
python scripts/calculate_mem.py --model Qwen/Qwen2.5-0.5B --generate-config --output-config ./my_model.yaml
```

## Parameter Combinations

| Scenario | `--find-max-seq-len` | `--gen-len` | `--prompt-len` | Behavior |
|----------|:---:|:---:|:---:|---------|
| 1 | ✗ | ✓ | ✓ | Normal estimation with decode factor |
| 2 | ✗ | ✓ | ✗ | Error: `--prompt-len required` |
| 3 | ✗ | ✗ | ✓ | Error: `--gen-len required` |
| 4 | ✗ | ✗ | ✗ | Error: both required |
| 5 | ✓ | ✗ | ✗ | Search max gen_len (default prompt_len=4096) |
| 6 | ✓ | ✗ | ✓ | Search max gen_len with specified prompt_len |
| 7 | ✓ | ✓ | ✗ | Search max prompt_len with specified gen_len |
| 8 | ✓ | ✓ | ✓ | Warning, treated as scenario 1 |

## Supported Data Types

- `fp32`: 4 bytes
- `fp16` / `bf16`: 2 bytes
- `fp8` / `int8`: 1 byte
- `int4`: 0.5 bytes

## When to Use This Skill

Use this skill for:
1. **Capacity Planning**: Determine if a model fits on given hardware
2. **Optimization**: Compare different parallel strategies
3. **PD Separation**: Analyze Prefill/Decode memory requirements
4. **Model Comparison**: Compare memory usage across different models
5. **Config Generation**: Create YAML configs for new models

## Key Formulas

**Total Memory**:
```
Total = Weights + KV Cache + Activations + System Reserved
```

**KV Cache** (per GPU):
```
2 * batch_size * seq_len * num_kv_heads * head_dim * num_layers / (tp * cp)
```

**Activation** (Decode, seq_len=1):
```
batch_size * seq_len * hidden_size * num_experts * decode_factor / cp
```

**Activation** (Prefill):
```
batch_size * seq_len * hidden_size * num_experts * has_prefill_factor / cp
```

## Workflow

1. **Identify Input Source**:
   - YAML config file exists → Use `--config`
   - HuggingFace model → Use `--model`
   - Local weights → Use `--local`

2. **Determine Goal**:
   - Fixed sequence length → Provide both `--prompt-len` and `--gen-len`
   - Find max gen_len (Decode) → Use `--find-max-seq-len` with `--prompt-len`
   - Find max prompt_len (Prefill) → Use `--find-max-seq-len` with `--gen-len`

3. **Specify Hardware** (for max sequence search):
   - Use `--chip` with chip name (e.g., `H100-80GB` or `nvidia/H100-80GB`)

4. **Set Parallel Strategy**:
   - Adjust `--tp`, `--pp`, `--dp`, `--cp`, `--ep` as needed

5. **Run and Analyze**:
   - Review memory breakdown in the report
   - Check calculation steps for verification
