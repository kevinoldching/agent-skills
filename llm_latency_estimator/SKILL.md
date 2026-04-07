---
name: llm-roofline-analysis
description: |
  基于华为 Ascend 芯片的 vLLM 大模型整网 Roofline 性能分析工具。
  当用户想要以下操作时使用此 Skill：
  - 分析 LLM 模型的计算强度（Roofline 分析）
  - 估算 Prefill/Decode 阶段推理时延
  - 针对 Ascend 芯片给出性能优化建议
  - 分析模型各模块是 compute-bound 还是 memory-bound

  支持输入源：vLLM 源码路径、HuggingFace 模型名、用户指定 config.json。
  此 Skill 仅用于推理性能分析，不适用于训练。
---

# LLM Roofline 性能分析工具

## 使用方式

用户提供以下输入：
- **模型来源**: vLLM 源码路径 / HuggingFace 模型名 / config.json
- **参数**: prompt_len, gen_len, chip（芯片型号或自定义参数）

Skill 会输出：
- 模型各模块的 Roofline 分析（计算强度、瓶颈类型）
- Prefill / Decode 阶段时延估算
- 针对硬件平台的性能优化建议

## 核心流程

### 1. 模型结构解析

AI 读取 vLLM 源码结构和 HuggingFace config.json，识别：
- 模块组织: Embedding / Attention / FFN / Output
- FFN 类型: Standard / SwiGLU / MoE
- 注意力类型: MHA / GQA / MLA / Mamba（从 config.json 字段判断）

### 2. 算子 FLOPs 动态推导

**不要写死公式**。AI 根据注意力类型，从 `configs/attention_patterns.yaml` 获取计算特征模式，代入 config 中的 shape 参数动态计算。

AI 任务：
- 从 config.json 提取 hidden_size, num_attention_heads, num_key_value_heads, intermediate_size
- 从源码提取 MatMul weight shape
- 根据注意力类型计算 FLOPs 和 Bytes
- 融合算子: AI 推导 + WebSearch 验证

### 3. Roofline 分析

公式（严格使用 max，非相加）：
```
T_module = max(FLOPs / PeakFLOPs, Bytes / Bandwidth)
```

AI Balance = PeakFLOPs / Bandwidth

瓶颈判定：
- I > AI Balance → Compute-Bound
- I < AI Balance → Memory-Bound

### 4. 时延估算

```
T_prefill = Σ_layers max(FLOPs_layer(prompt_len) / PeakFLOPs, Bytes_layer / Bandwidth)

T_attn_decode(seq_len) = max(FLOPs_attn(seq_len) / PeakFLOPs, Bytes_kvcache(seq_len) / Bandwidth)
T_ffn_decode = max(FLOPs_ffn / PeakFLOPs, Bytes_ffn_weights / Bandwidth)
T_decode(seq_len) = Σ_layers (T_attn_layer(seq_len) + T_ffn_layer)

T_total = T_prefill + gen_len × T_decode
```

### 5. 优化建议生成

基于瓶颈分析，AI 输出针对性建议：
- Memory-Bound 模块: 量化、KV cache 压缩、Flash Attention、算子融合
- Compute-Bound 模块: Tensor Core 优化、精度选择、CANN 融合算子

### 6. 输出格式

报告包含：
1. **Roofline 分析**: Mermaid xychart 图表 + 模块分析表
2. **时延估算**: Prefill / Decode / 总时延分解
3. **优化建议**: 分模块针对性建议

## 配置文件

### chips.json
芯片参数（峰值算力 TFLOPS、带宽 GB/s），支持华为 Ascend 和 NVIDIA GPU。

### attention_patterns.yaml
注意力计算特征模式模板，供 AI 动态调用。

## 脚本

仅 `scripts/download_hf_config.py` 用于下载 HuggingFace config.json。

## 输出示例

见设计文档 `docs/superpowers/specs/2026-04-07-llm-roofline-analysis-design.md` 第 4 节。
