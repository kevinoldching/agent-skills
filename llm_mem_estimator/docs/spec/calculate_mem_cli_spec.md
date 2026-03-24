# calculate_mem.py 使用说明

LLM Memory Estimator CLI 工具，用于估算大语言模型的 GPU 显存占用。

## 基本用法

```bash
python scripts/calculate_mem.py [选项]
```

## 输入选项（必选一项）

| 选项 | 说明 | 示例 |
|------|------|------|
| `--config` | 指定模型 YAML 配置文件路径 | `--config configs/models/gpt-oss-120b.yaml` |
| `--model` | HuggingFace 模型名称 | `--model Qwen/Qwen2.5-0.5B` |
| `--local` | 本地模型权重路径 | `--local ./models/llama-weights` |

## 配置生成选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--generate-config` | 从模型生成 YAML 配置并保存 | 不生成 |
| `--output-config` | 指定生成配置的输出路径 | `configs/models/<model_name>.yaml` |

### 示例：生成配置

```bash
# 从 HuggingFace 模型生成配置
python scripts/calculate_mem.py --model Qwen/Qwen2.5-0.5B --generate-config

# 指定输出路径
python scripts/calculate_mem.py --model Qwen/Qwen2.5-0.5B --generate-config --output-config ./my_config.yaml
```

## 估算参数

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--batch-size` | 批处理大小 | 1 |
| `--prompt-len` | 输入提示词长度 | 4096 |
| `--gen-len` | 生成输出长度 | 1024 |
| `--kv-dtype` | KV Cache 数据类型 | fp16 |
| `--activation-dtype` | 激活值数据类型 | fp16 |
| `--activation-peak` | 固定 activation 峰值 (GB)，覆盖公式计算 | None |

> **注意**：
> - 对于 Prefill/Decode 分离部署场景：Activation 根据场景使用不同因子：
>   - **Decode 阶段**（固定 seq_len=1）：使用 decode 因子 (12.5)
>   - **Prefill 阶段**（seq_len = prompt_len + gen_len）：使用 has_prefill 因子 (1.25)
> - 使用 `--activation-peak` 时：用户直接指定 activation 峰值，搜索函数用剩余显存计算

### 场景说明

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

支持的数据类型：`fp32`, `fp16`, `bf16`, `fp8`, `int8`, `int4`

## 并行配置

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--tp` | Tensor Parallel (TP) 并行度 | 1 |
| `--pp` | Pipeline Parallel (PP) 并行度 | 1 |
| `--dp` | Data Parallel (DP) 并行度 | 1 |
| `--cp` | Context Parallel (CP) 并行度 | 1 |
| `--ep` | Expert Parallel (EP) 并行度 | 1 |
| `--stage` | PD分离阶段 | None（混部/通用并行配置） |

`--stage` 选项：
- 不指定：`parallel_defaults.hybrid`（混部/通用场景）
- `prefill`：使用 `parallel_defaults.prefill`
- `decode`：使用 `parallel_defaults.decode`

### 示例：多卡并行估算

```bash
# 8卡 Tensor Parallel
python scripts/calculate_mem.py --config configs/models/gpt-oss-120b.yaml --tp 8

# 2路并行 (TP=2, PP=2)
python scripts/calculate_mem.py --config configs/models/gpt-oss-120b.yaml --tp 2 --pp 2
```

### 示例：PD分离场景

```bash
# 混部场景（不指定 --stage）
python scripts/calculate_mem.py --model Kimi-K2.5 --tp 4

# PD分离场景
python scripts/calculate_mem.py --model Kimi-K2.5 --tp 4 --stage prefill
python scripts/calculate_mem.py --model Kimi-K2.5 --tp 4 --stage decode
```

## 硬件配置

| 选项 | 说明 |
|------|------|
| `--chip` | 芯片名称 (如 `H100-80GB` 或 `nvidia/H100-80GB`) |
| `--find-max-seq-len` | 根据芯片显存查找最大支持序列长度。<br>• 仅指定 `--gen-len` 时：搜索 max prompt_len (Prefill 场景)<br>• 仅指定 `--prompt-len` 时：搜索 max gen_len (Decode 场景)<br>• 两者都指定：正常估算 |
| `--system-reserved` | 系统保留显存 (GB) | 2.0 |

### 支持的芯片

在 `configs/chips.json` 中定义了支持的芯片，包括：
- `H100-80GB` / `H100-141GB` (nvidia)
- `A100-80GB` / `A100-40GB` (nvidia)
- `RTX-4090` / `RTX-3090` (nvidia)
- `Ascend-910B-64GB` / `Ascend-910B-32GB` (huawei)

支持两种命名格式：
- 简短格式：`H100-80GB`（推荐）
- 完整格式：`nvidia/H100-80GB`

### 示例：查找最大序列长度

```bash
# 查找在 H100-80GB 上的最大序列长度
python scripts/calculate_mem.py --config configs/models/gpt-oss-120b.yaml \
    --chip H100-80GB --find-max-seq-len --batch-size 1
```

## 输出选项

| 选项 | 说明 |
|------|------|
| `--output` | 将报告保存到指定文件 |

### 示例：保存报告

```bash
python scripts/calculate_mem.py --config configs/models/gpt-oss-120b.yaml \
    --batch-size 8 --prompt-len 4096 --gen-len 1024 --tp 8 --output memory_report.txt
```

## 使用示例

### 示例 1：使用预定义配置文件估算

```bash
python scripts/calculate_mem.py \
    --config configs/models/gpt-oss-120b.yaml \
    --batch-size 4 \
    --prompt-len 4096 \
    --gen-len 1024 \
    --tp 4
```

输出示例：
```
==================================================
LLM Memory Estimation Report
==================================================
Model: gpt-oss-120b
Configuration:
  Batch Size: 4
  Sequence Length: 2048
  TP: 4 | PP: 1 | DP: 1 | CP: 1

Memory Breakdown (per GPU):
  Weights:           32.50 GB
  KV Cache:          16.00 GB
  Activations:        8.25 GB
  System Reserved:    2.00 GB
  ─────────────────────────────
  Total:             58.75 GB
==================================================
```

### 示例 2：从 HuggingFace 模型直接估算

```bash
python scripts/calculate_mem.py \
    --model Qwen/Qwen2.5-1.5B \
    --batch-size 1 \
    --prompt-len 4096 \
    --gen-len 1024
```

### 示例 3：查找硬件最大支持生成长度 (Decode 场景)

```bash
python scripts/calculate_mem.py \
    --config configs/models/gpt-oss-120b.yaml \
    --chip nvidia/H100-80GB \
    --find-max-seq-len \
    --batch-size 1 \
    --prompt-len 4096 \
    --tp 8

# 输出: Maximum generated length: 16,384
```

### 示例 4：查找硬件最大支持 Prompt 长度 (Prefill 场景)

```bash
# 固定 gen_len=1，搜索最大 prompt_len
python scripts/calculate_mem.py \
    --config configs/models/gpt-oss-120b.yaml \
    --chip nvidia/H100-80GB \
    --find-max-seq-len \
    --gen-len 1 \
    --batch-size 1 \
    --tp 8

# 输出: Maximum prompt length: xxx
```

### 示例 5：使用不同数据类型的量化

```bash
# 使用 FP8 量化
python scripts/calculate_mem.py \
    --config configs/models/gpt-oss-120b.yaml \
    --kv-dtype fp8 \
    --activation-dtype fp8
```

### 示例 6：Scene 8 - batch_size ≠ 1 (直接估算)

```bash
# 指定 batch_size=2，直接估算并显示是否满足显存
python scripts/calculate_mem.py \
    --config configs/models/gpt-oss-120b.yaml \
    --chip Ascend-910B-64GB \
    --batch-size 2 \
    --prompt-len 2048 \
    --gen-len 2048 \
    --find-max-seq-len

# 输出: Status: ✅ Fits 或 ❌ Exceeds
```

### 示例 7：Scene 8 - batch_size = 1 (搜索最大 batch_size)

```bash
# 指定 batch_size=1，搜索最大支持的 batch_size
python scripts/calculate_mem.py \
    --config configs/models/gpt-oss-120b.yaml \
    --chip Ascend-910B-64GB \
    --batch-size 1 \
    --prompt-len 2048 \
    --gen-len 2048 \
    --find-max-seq-len

# 输出: Maximum batch size: 210
```

### 示例 8：使用固定 activation 峰值

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

## 完整选项列表

```
usage: calculate_mem.py [-h] (--config CONFIG | --model MODEL | --local LOCAL)
                       [--generate-config] [--output-config OUTPUT_CONFIG]
                       [--batch-size BATCH_SIZE] [--prompt-len PROMPT_LEN] [--gen-len GEN_LEN]
                       [--kv-dtype KV_DTYPE] [--activation-dtype ACTIVATION_DTYPE]
                       [--activation-peak ACTIVATION_PEAK]
                       [--tp TP] [--pp PP] [--dp DP] [--cp CP] [--ep EP]
                       [--chip CHIP] [--find-max-seq-len]
                       [--system-reserved SYSTEM_RESERVED]
                       [--output OUTPUT]
```

## 注意事项

1. **输入优先级**：`--config` > `--model` > `--local`，三者互斥
2. **查找最大序列长度时**：必须指定 `--chip` 参数
3. **并行度影响显存**：TP/PP/DP/CP 会影响每张卡的显存占用
4. **系统保留显存**：默认保留 2GB，可根据实际情况调整
