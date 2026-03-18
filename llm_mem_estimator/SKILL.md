---
name: llm-memory-estimator
description: |
  LLM GPU 显存估算工具 - 估算大语言模型的 GPU 显存占用。
  当用户想要以下操作时使用此 Skill：
  - 估算 LLM 模型权重、KV Cache、激活值的 GPU 显存占用
  - 查找特定硬件（如 H100-80GB）支持的最大序列长度
  - 分析不同并行策略（TP、PP、DP、CP、EP）的显存分布
  - 处理 Prefill/Decode (PD) 分离场景
  - 从 HuggingFace 模型生成 YAML 配置
  - 比较不同芯片、模型或并行配置的显存占用

  此 Skill 仅用于 LLM 推理显存估算，不适用于训练。
---

# LLM 显存估算工具

此工具帮助估算大语言模型（LLMs）的 GPU 显存占用，包括模型权重、KV Cache 和激活值。支持 Prefill/Decode 分离场景和各种并行策略。

## 项目结构

```
llm_mem_estimator/
├── scripts/
│   └── calculate_mem.py      # 主 CLI 工具
├── configs/
│   ├── models/               # 模型 YAML 配置
│   │   ├── gpt-oss-120b.yaml
│   │   ├── DeepSeek-V3.yaml
│   │   └── ...
│   ├── weight_mapping_rules.yaml
│   └── chips.json            # 硬件规格
├── llm_mem_estimator/
│   ├── memory_estimator.py   # 核心估算逻辑
│   ├── model_config.py       # 配置处理
│   ├── model_detector.py     # HuggingFace 模型检测
│   └── report_generator.py   # 报告生成
└── docs/spec/
    ├── calculate_mem_cli_spec.md
    ├── yaml_config_spec.md
    └── weight_mapping_rules_spec.md
```

## 核心功能

### 1. 显存估算

计算总 GPU 显存占用：
- **模型权重**：基于模型参数量和数据类型
- **KV Cache**：注意力机制的 key-value 存储
- **激活值**：临时计算内存
- **系统预留**：固定开销（默认 2GB）

### 2. 并行策略支持

| 策略 | 说明 | CLI 参数 |
|------|------|----------|
| TP | Tensor Parallel（列/行切分） | `--tp` |
| PP | Pipeline Parallel（层切分） | `--pp` |
| DP | Data Parallel（复制） | `--dp` |
| CP | Context Parallel（序列切分） | `--cp` |
| EP | Expert Parallel（MoE 专家切分） | `--ep` |

### 3. PD 分离场景

**Prefill 阶段**：处理输入 prompt
- 使用 `has_prefill` 系数（1.25）
- 激活值随 `prompt_len + gen_len` 缩放

**Decode 阶段**：生成输出 token
- 使用 `decode` 系数（12.5）
- 激活值固定（seq_len=1）

### 4. 硬件支持

`configs/chips.json` 支持的芯片：
- NVIDIA: H100-80GB, H100-141GB, A100-80GB, RTX-4090
- 华为: Ascend-910B-64GB, Ascend-910B-32GB
- AMD: MI300X, MI350X
- Intel: Gaudi2, Gaudi3

## CLI 使用方法

### 基本显存估算

```bash
# 使用预定义配置
python scripts/calculate_mem.py \
    --config configs/models/gpt-oss-120b.yaml \
    --batch-size 4 \
    --prompt-len 4096 \
    --gen-len 1024 \
    --tp 4

# 从 HuggingFace 模型
python scripts/calculate_mem.py \
    --model Qwen/Qwen2.5-1.5B \
    --batch-size 1 \
    --prompt-len 4096 \
    --gen-len 1024
```

### 查找最大序列长度

```bash
# 查找最大 gen_len（Decode 场景，固定 prompt_len）
python scripts/calculate_mem.py \
    --config configs/models/gpt-oss-120b.yaml \
    --chip H100-80GB \
    --find-max-seq-len \
    --prompt-len 4096 \
    --tp 8

# 查找最大 prompt_len（Prefill 场景，固定 gen_len）
python scripts/calculate_mem.py \
    --config configs/models/gpt-oss-120b.yaml \
    --chip H100-80GB \
    --find-max-seq-len \
    --gen-len 1 \
    --tp 8
```

### 生成模型配置

```bash
# 从 HuggingFace 模型生成 YAML 配置
python scripts/calculate_mem.py --model deepseek-ai/DeepSeek-V3 --generate-config

# 保存到自定义位置
python scripts/calculate_mem.py --model Qwen/Qwen2.5-0.5B --generate-config --output-config ./my_model.yaml
```

## 参数组合

| 场景 | `--find-max-seq-len` | `--gen-len` | `--prompt-len` | 行为 |
|------|:---:|:---:|:---:|------|
| 1 | ✗ | ✓ | ✓ | 使用 decode 系数进行常规估算 |
| 2 | ✗ | ✓ | ✗ | 错误：需要 `--prompt-len` |
| 3 | ✗ | ✗ | ✓ | 错误：需要 `--gen-len` |
| 4 | ✗ | ✗ | ✗ | 错误：两者都需要 |
| 5 | ✓ | ✗ | ✗ | 搜索最大 gen_len（默认 prompt_len=4096） |
| 6 | ✓ | ✗ | ✓ | 搜索指定 prompt_len 下的最大 gen_len |
| 7 | ✓ | ✓ | ✗ | 搜索指定 gen_len 下的最大 prompt_len |
| 8 | ✓ | ✓ | ✓ | 警告，按场景 1 处理 |

## 支持的数据类型

- `fp32`: 4 字节
- `fp16` / `bf16`: 2 字节
- `fp8` / `int8`: 1 字节
- `int4`: 0.5 字节

## 使用场景

此 Skill 适用于：
1. **容量规划**：判断模型是否适合给定硬件
2. **优化**：比较不同并行策略
3. **PD 分离**：分析 Prefill/Decode 显存需求
4. **模型对比**：比较不同模型的显存占用
5. **配置生成**：为新模型创建 YAML 配置

## 关键公式

**总显存**：
```
Total = Weights + KV Cache + Activations + System Reserved
```

**KV Cache**（每 GPU）：
```
2 * batch_size * seq_len * num_kv_heads * head_dim * num_layers / (tp * cp)
```

**激活值**（Decode，seq_len=1）：
```
batch_size * seq_len * hidden_size * num_experts * decode_factor / cp
```

**激活值**（Prefill）：
```
batch_size * seq_len * hidden_size * num_experts * has_prefill_factor / cp
```

## 工作流程

1. **确定输入来源**：
   - 已有 YAML 配置文件 → 使用 `--config`
   - HuggingFace 模型 → 使用 `--model`
   - 本地权重 → 使用 `--local`

2. **确定目标**：
   - 固定序列长度 → 同时提供 `--prompt-len` 和 `--gen-len`
   - 查找最大 gen_len（Decode）→ 使用 `--find-max-seq-len` 配合 `--prompt-len`
   - 查找最大 prompt_len（Prefill）→ 使用 `--find-max-seq-len` 配合 `--gen-len`

3. **指定硬件**（用于最大序列搜索）：
   - 使用 `--chip` 指定芯片名称（如 `H100-80GB` 或 `nvidia/H100-80GB`）

4. **设置并行策略**：
   - 根据需要调整 `--tp`、`--pp`、`--dp`、`--cp`、`--ep`

5. **运行和分析**：
   - 查看报告中的显存分布
   - 检查计算步骤以验证结果

## 参考文档

- [CLI 详细说明](./docs/spec/calculate_mem_cli_spec.md)
- [YAML 配置规范](./docs/spec/yaml_config_spec.md)
- [权重映射规则](./docs/spec/weight_mapping_rules_spec.md)
