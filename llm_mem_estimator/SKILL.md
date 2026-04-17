---
name: llm-mem-estimator
description: |
  LLM GPU 显存估算工具 - 估算大语言模型的 GPU 显存占用。
  当用户想要以下操作时使用此 Skill：
  - 估算 LLM 模型权重、KV Cache、激活值的 GPU 显存占用
  - 查找特定硬件（如 H100-80GB、Ascend-910B-64GB）支持的最大序列长度或最大 batch_size
  - 分析不同并行策略（TP、PP、DP、CP、EP）的显存分布
  - 处理 Prefill/Decode (PD) 分离场景
  - 从 HuggingFace 模型生成 YAML 配置文件
  - 从远程服务器通过 SFTP 读取模型权重（无需下载）
  - 比较不同芯片、模型或并行配置的显存占用
  - 固定 activation 峰值（--activation-peak）进行估算

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
│   ├── models/                # 模型 YAML 配置
│   │   ├── gpt-oss-120b.yaml
│   │   ├── DeepSeek-V3.yaml
│   │   ├── Kimi-K2.5.yaml
│   │   └── ...
│   ├── weight_mapping_rules.yaml
│   └── chips.json             # 硬件规格
├── llm_mem_estimator/
│   ├── memory_estimator.py    # 核心估算逻辑
│   ├── model_config.py        # 配置处理
│   ├── model_detector.py      # HuggingFace 模型检测
│   └── report_generator.py   # 报告生成
├── tests/
│   └── test_models.yaml       # 测试模型列表
└── docs/spec/
    ├── calculate_mem_cli_spec.md
    ├── yaml_config_spec.md
    └── weight_mapping_rules_spec.md
```

## 核心功能

### 1. 显存估算

计算总 GPU 显存占用：
- **模型权重**：基于模型参数量和数据类型，支持 TP/PP/DP/EP 并行切分
- **KV Cache**：注意力机制的 key-value 存储，支持 MLA/MQA/GQA/SWA 等架构
- **激活值**：临时计算内存，使用 has_prefill(1.25) 或 decode(12.5) 系数
- **系统预留**：固定开销（默认 2GB）

### 2. 并行策略支持

| 策略 | 说明 | CLI 参数 |
|------|------|----------|
| TP | Tensor Parallel（列/行切分） | `--tp` |
| PP | Pipeline Parallel（层切分） | `--pp` |
| DP | Data Parallel（复制） | `--dp` |
| CP | Context Parallel（序列切分） | `--cp` |
| EP | Expert Parallel（MoE 专家切分） | `--ep` |

### 3. TP 变体（TP Variants）

TP 变体允许不同的权重组使用不同的 TP size，提供更灵活的并行策略配置。

#### TP 变体类型

| 变体 | 说明 | CLI 参数 |
|------|------|----------|
| TP_O_PROJ | o_proj 权重的 TP size | `--tp-o-proj` |
| TP_MLP | FFN 权重的 TP size | `--tp-mlp` |
| TP_SHARED_EXPERT | 共享专家的 TP size | `--tp-shared-expert` |
| TP_EMBEDDING | embedding 的 TP size | `--tp-embedding` |

#### 配置位置

TP 变体定义在 `configs/weight_mapping_rules.yaml` 的 `tp_variants` 区块：

```yaml
tp_variants:
  TP_O_PROJ: 8      # o_proj.weight 的默认 TP size
  TP_MLP: 8         # FFN 权重的默认 TP size
  TP_SHARED_EXPERT: 8  # 共享专家的默认 TP size
  TP_EMBEDDING: 8   # embedding 的默认 TP size
```

#### 使用方式

**使用 YAML 默认值**：
```bash
python scripts/calculate_mem.py --config configs/models/Kimi-K2.5.yaml --tp 4
```

**覆盖 TP 变体**：
```bash
python scripts/calculate_mem.py --config configs/models/Kimi-K2.5.yaml --tp 4 --tp-o-proj 4 --tp-mlp 8 --tp-shared-expert 4 --tp-embedding 2
```

#### 与并行策略表结合

| 配置 | 说明 |
|------|------|
| TP=8 | 全局 TP size |
| TP=8, --tp-o-proj 4 | o_proj 使用 TP=4，MLP 使用 TP=8 |
| TP=8, --tp-embedding 1 | embedding 使用 TP=1（replicated） |

### 4. 并行策略约束（续）

使用并行策略时必须满足以下约束：

#### 约束公式
```
TP × DP = EP = 总卡数
```

#### 约束规则
| 规则 | 说明 |
|------|------|
| TP × DP = 总卡数 | Tensor Parallel 乘以 Data Parallel 等于总卡数 |
| EP = 总卡数 | Expert Parallel 必须等于总卡数 |
| EP = TP × DP | EP 与 TP×DP 必须相等 |
| EP ≤ MoE experts | Expert Parallel 不能超过 MoE 专家数量 |
| TP ≤ 单机卡数 | Tensor Parallel 不能超过单机 GPU 数量（如果已知） |

#### 示例
| 配置 | 有效？ | 说明 |
|------|--------|------|
| TP=8, DP=1, EP=8 | ✓ | 8×1=8, EP=8 |
| TP=8, DP=2, EP=16 | ✓ | 8×2=16, EP=16 |
| TP=8, DP=1, EP=16 | ✗ | 8×1=8 ≠ 16 |
| TP=4, DP=4, EP=16 | ✓ | 4×4=16, EP=16 |

#### 典型配置示例
```
# 单机 8 卡配置（TP=8）
TP=8, DP=1, EP=8

# 16 卡配置（TP=8 × DP=2）
TP=8, DP=2, EP=16

# 128 卡配置（TP=8 × DP=16）
TP=8, DP=16, EP=128
```

#### Skill 工作流程

当用户指定并行策略时，skill 应该：

1. **验证约束**：检查 `TP × DP` 是否等于 `EP`
2. **修正建议**：如果不满足，给出修正建议
3. **计算总卡数**：使用 `TP × DP` 或 `EP`（两者应相等）作为总卡数
4. **分配验证**：检查芯片数量是否足够支持总卡数

#### 报告中的显示

在显存估算报告中，应明确显示：
- 总卡数：`Total GPUs = TP × DP = EP = 8 × 2 = 16`
- 每张卡的显存：`Memory per GPU = Total Memory / Total GPUs`

### 5. PD 分离场景

**Prefill 阶段**：处理输入 prompt
- 使用 `has_prefill` 系数（1.25）
- 激活值随 `prompt_len + gen_len` 缩放

**Decode 阶段**：生成输出 token
- 使用 `decode` 系数（12.5）
- 激活值固定（seq_len=1）

**固定 activation 峰值**：使用 `--activation-peak` 直接指定激活值

### 6. 硬件支持

`configs/chips.json` 支持的芯片：
- NVIDIA: H100-80GB, H100-141GB, A100-80GB, A100-40GB, RTX-4090, RTX-3090
- 华为: Ascend-910B-64GB, Ascend-910B-32GB

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

# 从远程服务器（SFTP 远程读取）
python scripts/calculate_mem.py \
    --remote ubuntu@192.168.1.100:/data/models/Qwen2.5-0.5B \
    --batch-size 1 \
    --prompt-len 4096 \
    --gen-len 1024 \
    --tp 4

### PD分离场景（--stage 参数）

```bash
# 混部场景（不指定 --stage，走 hybrid 层）
python scripts/calculate_mem.py --model Kimi-K2.5 --tp 4

# PD分离场景
python scripts/calculate_mem.py --model Kimi-K2.5 --tp 4 --stage prefill
python scripts/calculate_mem.py --model Kimi-K2.5 --tp 4 --stage decode
```

### 查找最大序列长度 / 最大 batch_size

```bash
# Scene 5: 搜索最大 gen_len（Decode 场景，固定 prompt_len）
python scripts/calculate_mem.py \
    --config configs/models/gpt-oss-120b.yaml \
    --chip H100-80GB \
    --find-max-seq-len \
    --prompt-len 4096 \
    --tp 8

# Scene 6: 指定 prompt_len，搜索最大 gen_len
python scripts/calculate_mem.py \
    --config configs/models/gpt-oss-120b.yaml \
    --chip H100-80GB \
    --find-max-seq-len \
    --prompt-len 4096 \
    --tp 8

# Scene 7: 指定 gen_len，搜索最大 prompt_len（Prefill 场景）
python scripts/calculate_mem.py \
    --config configs/models/gpt-oss-120b.yaml \
    --chip H100-80GB \
    --find-max-seq-len \
    --gen-len 1 \
    --tp 8

# Scene 8: batch_size≠1 时直接估算显示 Fits/Exceeds
python scripts/calculate_mem.py \
    --config configs/models/gpt-oss-120b.yaml \
    --chip Ascend-910B-64GB \
    --batch-size 2 \
    --prompt-len 2048 \
    --gen-len 2048 \
    --find-max-seq-len

# Scene 8: batch_size=1 时搜索最大 batch_size
python scripts/calculate_mem.py \
    --config configs/models/gpt-oss-120b.yaml \
    --chip Ascend-910B-64GB \
    --batch-size 1 \
    --prompt-len 2048 \
    --gen-len 2048 \
    --find-max-seq-len
```

### 生成模型配置

```bash
# 从 HuggingFace 模型生成 YAML 配置
python scripts/calculate_mem.py --model deepseek-ai/DeepSeek-V3 --generate-config

# 保存到自定义位置
python scripts/calculate_mem.py --model Qwen/Qwen2.5-0.5B --generate-config --output-config ./my_model.yaml

# 从本地权重目录加载权重，生成 YAML 配置
python scripts/calculate_mem.py --local /path/models/Qwen2.5-0.5B --generate-config
```

### 使用固定 activation 峰值

```bash
# 指定 activation 峰值固定为 10GB，用剩余显存搜索最大 batch_size
python scripts/calculate_mem.py \
    --config configs/models/gpt-oss-120b.yaml \
    --chip Ascend-910B-64GB \
    --batch-size 1 \
    --prompt-len 2048 \
    --gen-len 2048 \
    --activation-peak 10 \
    --find-max-seq-len
```

## 场景说明

| 场景 | `--find-max-seq-len` | `--gen-len` | `--prompt-len` | 处理逻辑 |
|-----|:---:|:---:|:---:|---------|
| 1 | ✗ | ✓ | ✓ | 正常估算，使用 **has_prefill** factor (1.25) |
| 2 | ✗ | ✓ | ✗ | **报错**: `--prompt-len is required` |
| 3 | ✗ | ✗ | ✓ | **报错**: `--gen-len is required` |
| 4 | ✗ | ✗ | ✗ | **报错**: `--prompt-len and --gen-len are required` |
| 5 | ✓ | ✗ | ✗ | prompt_len=默认值(4096)，搜索 max gen_len，使用 **decode** factor |
| 6 | ✓ | ✗ | ✓ | prompt_len=用户指定，搜索 max gen_len，使用 **decode** factor |
| 7 | ✓ | ✓ | ✗ | gen_len=用户指定，搜索 max prompt_len，使用 **has_prefill** factor |
| 8 | ✓ | ✓ | ✓ | batch_size≠1: 正常估算 + 显示 Fits/Exceeds; batch_size=1: **搜索 max batch_size** |

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

## 工作流程

1. **确定输入来源**：
   - 已有 YAML 配置文件 → 使用 `--config`
   - HuggingFace 模型 → 使用 `--model`
   - 本地权重 → 使用 `--local`
   - 远程服务器权重 → 使用 `--remote user@host:/path/to/model`

2. **确定目标**：
   - 固定序列长度 → 同时提供 `--prompt-len` 和 `--gen-len`
   - 查找最大 gen_len（Decode）→ 使用 `--find-max-seq-len` 配合 `--prompt-len`
   - 查找最大 prompt_len（Prefill）→ 使用 `--find-max-seq-len` 配合 `--gen-len`
   - 查找最大 batch_size → 使用 `--find-max-seq-len` 配合 `--prompt-len` 和 `--gen-len`，batch_size=1

3. **指定硬件**（用于最大序列搜索）：
   - 使用 `--chip` 指定芯片名称（如 `H100-80GB` 或 `nvidia/H100-80GB`）

4. **设置并行策略**：
   - 根据需要调整 `--tp`、`--pp`、`--dp`、`--cp`、`--ep`、`--tp-o-proj`、`--tp-mlp`、`--tp-shared-expert`、`--tp-embedding` 等tp_variant参数

5. **可选：固定 activation 峰值**：
   - 使用 `--activation-peak` 直接指定激活值（单位：GB）

6. **运行和分析**：
   - 查看报告中的显存分布
   - 检查是否 Fits/Exceeds

## 支持的模型

通过 `--generate-config` 支持从 HuggingFace 自动生成配置：
- Qwen 系列（Qwen2.5-0.5B, Qwen2.5-1.5B 等）
- DeepSeek 系列（DeepSeek-V3, DeepSeek-V3.1, DeepSeek-V3.2）
- Kimi 系列（Kimi-K2.5）
- MiniMax 系列（MiniMax-M2.5）
- GLM 系列（GLM-5）
- GPT-OSS 系列（gpt-oss-120b, gpt-oss-20b）

支持的注意力架构：MHA, MQA, GQA, SWA, MLA (DeepSeek), DSA, GDN

支持的 FFN 类型：Standard FFN, SwiGLU, MoE (支持 EP)

## 调试信息

权重元数据缓存位置：`~/.cache/llm_mem_estimator/metadata_cache/`

查看缓存的权重数据：
```bash
cat ~/.cache/llm_mem_estimator/metadata_cache/deepseek-ai--DeepSeek-V3_weights.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(list(d.keys())[:5])"
```

## 参考文档

- [CLI 详细说明](./docs/spec/calculate_mem_cli_spec.md)
- [YAML 配置规范](./docs/spec/yaml_config_spec.md)
- [权重映射规则](./docs/spec/weight_mapping_rules_spec.md)
