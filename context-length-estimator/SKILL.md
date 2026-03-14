---
name: context-length-estimator
description: 大模型Context长度估算工具。当用户想要估算在特定硬件上能支持多长的上下文长度时触发，包括计算模型权重、KV Cache、激活值的显存占用，以及不同芯片的显存规格对比。支持从模型名称自动获取参数或手动输入参数。
---

# Context Length Estimator

这个Skill用于估算在不同芯片上运行大模型时能支持的最大上下文长度。

## 使用场景

当用户询问以下问题时使用此Skill：
- "Llama-70B在H100上能跑多长的context?"
- "我的4090显卡能跑多大的模型?"
- "估算一下这个模型需要多少显存"
- "计算某个芯片能支持多少token"
- "模型参数XXB在XXGB显存上能跑多长序列"
- 任何关于大模型显存占用和context长度的估算问题

## 输入方式

### 方式一: 使用模型名称 (推荐)

直接指定模型名称和芯片名称:

```bash
python scripts/calculate_context.py -m llama-70b -c h100-80gb
```

支持的模型包括:
- Llama系列: llama-7b, llama-13b, llama-70b, llama-405b, llama3-8b, llama3-70b, llama3.1-405b, llama3.3-70b, llama4-17b等
- Qwen系列: qwen-7b, qwen-14b, qwen-72b, qwen2-57b-a14b, qwen2.5-7b~72b, qwen3-8b~72b, qwen3-max等
- MoE模型: mixtral-8x7b, mixtral-8x22b, deepseek-v2, deepseek-v3, deepseek-v2.5, qwen3-235b等
- 其他: mistral-7b, yi-34b, glm-4-9b, glm-4-130b, baichuan-13b, minimax-m2等

支持的芯片包括:
- NVIDIA: h100-80gb, h100-141gb, h20, b100, b200, a100-80gb, l40s, rtx-4090, rtx-5090等
- 华为昇腾: ascend-910b-64gb, ascend-910b-32gb等
- Intel Gaudi: gaudi2, gaudi3等
- AMD: mi300x, mi350x, mi250x等

### 方式二: 手动指定参数

当模型不在数据库中时，可以手动指定:

```bash
python scripts/calculate_context.py \
  --params 70 \
  --hidden 8192 \
  --layers 80 \
  --chip h100-80gb
```

## 参数说明

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| --model | -m | 模型名称 | 无 |
| --params | -p | 模型参数量(B) | 无 |
| --hidden | 无 | 隐藏层大小 | 4096 |
| --layers | -l | Transformer层数 | 32 |
| --heads | 无 | 注意力头数 | 32 |
| --head-dim | 无 | 注意力头维度 | 128 |
| --chip | -c | 芯片名称 | 无 |
| --vram | 无 | 显存大小(GB) | 无 |
| --quant | -q | 模型权重量化: fp32/fp16/fp8/int8/int4 | fp16 |
| --kv-quant | -k | KV Cache量化: fp32/fp16/fp8/int8/int4 | fp16 |
| --num-experts | -e | MoE模型要加载的专家数量 (默认全部) | 全部 |
| --batch | -b | 批大小 | 1 |
| --overhead | 无 | 系统开销(GB)，与--gpu-util二选一 | 2.0 |
| --gpu-util | -g | GPU显存利用率 (vLLM的gpu_utilize参数) | 0.9 |
| --kv-ratio | 无 | KV Cache显存占比 | 0.5 |
| --output | -o | 输出格式: text/json | text |

## 输出示例

运行后会输出详细的估算报告，包括:

1. **硬件配置**: 芯片型号、显存大小、可用显存
2. **模型配置**: 模型参数、量化方式、批大小
3. **内存占用详情**: 模型权重、KV Cache、激活值
4. **估算结果**: 最大token数、字符数、等效文本量

## 计算原理

### 显存占用构成

1. **模型权重**: `参数总量 × 量化字节数` (可通过 --quant 指定)
2. **KV Cache**: `2 × batch_size × seq_len × hidden_size × num_layers × KV量化字节数` (可通过 --kv-quant 指定)
3. **激活值**: `4 × batch_size × seq_len × hidden_size × num_layers × 量化字节数`

**KV Cache 量化示例**:
- FP16: `2 × B × S × H × L × 2B`
- INT8: `2 × B × S × H × L × 1B`
- INT4: `2 × B × S × H × L × 0.5B`

### 最大序列长度公式

```
max_seq_len = (可用显存 - 模型权重) / (每token KV Cache + 每token激活)
```

### 可用显存计算

有两种方式计算可用显存：

1. **vLLM 风格** (`--gpu-util`): `可用显存 = 显存总量 × gpu_utilization`
   - 默认 `gpu_utilization=0.9`，表示保留 10% 显存给系统/框架开销
   - 这是 vLLM、TensorRT-LLM 等推理框架的默认行为

2. **传统风格** (`--overhead`): `可用显存 = 显存总量 - 系统开销`
   - 当 `gpu_util=1.0` 时自动切换为此模式
   - 默认系统开销 2GB

## 示例命令

```bash
# Llama-70B 在 H100 80GB 上 (FP16模型权重 + FP16 KV Cache)
python scripts/calculate_context.py -m llama-70b -c h100-80gb

# Llama-70B 在 H100 80GB 上 (INT8模型权重 + INT8 KV Cache)
python scripts/calculate_context.py -m llama-70b -c h100-80gb -q int8 -k int8

# Llama-70B 在 H100 80GB 上 (FP16权重 + INT4 KV Cache) - 延长context
python scripts/calculate_context.py -m llama-70b -c h100-80gb -q fp16 -k int4

# Mixtral-8x7B (MoE模型) 在 A100 80GB 上
python scripts/calculate_context.py -m mixtral-8x7b -c a100-80gb

# Mixtral-8x7B 在 RTX 4090 (24GB) 上，只加载4个专家
python scripts/calculate_context.py -m mixtral-8x7b -c rtx-4090 -q int8 -e 4

# DeepSeek-V3 在 H100 141GB 上，加载64个专家
python scripts/calculate_context.py -m deepseek-v3 -c h100-141gb -q int4 -k int4 -e 64

# Qwen3-72B 在 H100 80GB 上 (FP8 量化)
python scripts/calculate_context.py -m qwen3-72b -c h100-80gb -q fp8 -k fp8

# Qwen3-Max 在 B100 (192GB) 上，加载64个专家
python scripts/calculate_context.py -m qwen3-max -c b100 -q fp8 -k fp8 -e 64

# Llama3-8B 在 H100 80GB 上 (FP8 极限优化)
python scripts/calculate_context.py -m llama3-8b -c h100-80gb -q fp8 -k fp8

# vLLM 风格：gpu_utilization=0.9 (默认，保留10%显存)
python scripts/calculate_context.py -m llama3-8b -c h100-80gb -g 0.9

# vLLM 风格：gpu_utilization=0.95 (更激进)
python scripts/calculate_context.py -m llama3-8b -c h100-80gb -g 0.95

# 传统风格：固定系统开销2GB (gpu_util=1.0)
python scripts/calculate_context.py -m llama3-8b -c h100-80gb -g 1.0 --overhead 2

# Qwen-72B 在 Ascend 910B 64GB 上
python scripts/calculate_context.py -m qwen-72b -c ascend-910b-64gb

# RTX 4090 上能跑的最大模型
python scripts/calculate_context.py -c rtx-4090 --params 30

# 列出所有支持的模型
python scripts/calculate_context.py --list-models

# 列出所有支持的芯片
python scripts/calculate_context.py --list-chips

# JSON格式输出 (便于程序处理)
python scripts/calculate_context.py -m llama-70b -c h100-80gb -o json
```

## 参考文档

- `references/model_params.md` - 常见模型参数参考
- `references/chip_specs.md` - 芯片显存规格参考
