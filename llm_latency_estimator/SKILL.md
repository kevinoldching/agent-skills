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

**必须读取用户提供的源码**。使用 Read 工具读取 vLLM 模型目录下的源码文件：

```
必须执行的操作：
1. 使用 Read 工具读取用户提供 vLLM 源码路径下的所有 .py 文件
2. 识别模型结构：遍历模型目录，列出所有模块文件
3. 读取 HuggingFace config.json（用户指定路径，或用 scripts/download_hf_config.py 下载）
4. 从源码中提取每个 MatMul 层的 weight shape（如 hidden_dim, intermediate_size）
5. 识别 FFN 类型：检查 FFN 模块代码，判断是 Standard / SwiGLU / MoE
6. 识别注意力类型：检查 attention 模块代码，判断是 MHA / GQA / MLA / Mamba
```

识别内容：
- 模块组织: Embedding / Attention / FFN / Output
- FFN 类型: Standard / SwiGLU / MoE
- 注意力类型: MHA / GQA / MLA / Mamba（从源码和 config.json 判断）

### 2. 算子 FLOPs 动态推导

**必须从源码提取 shape 参数，不要写死公式**。

```
必须执行的操作：
1. 使用 Read 工具读取 vLLM 源码中的 model.py / attention.py / ffn.py
2. 从源码中找到所有 Linear/MatMul 层的 in_features 和 out_features
3. 从 config.json 提取：hidden_size, num_attention_heads, num_key_value_heads, intermediate_size
4. 根据识别的注意力类型，从 configs/attention_patterns.yaml 获取计算特征模式
5. AI 代入实际 shape 值，计算每个模块的 FLOPs 和内存访问量（Bytes）
```

**融合算子处理**：
- 读取源码时，识别融合的 LayerNorm + SiLU 等模式
- AI 推导等效 FLOPs，必要时用 WebSearch 验证融合算子在 Ascend 上的性能

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

使用 Read 工具读取 `configs/chips.json`，获取芯片参数：
- 峰值算力 `peak_fp32_tflops`
- 带宽 `bandwidth_gb_s`

如果用户指定了芯片名称（如 Ascend-910B-64GB），从 chips.json 中查找对应参数。如果用户提供了自定义参数（peak_flops, bandwidth），直接使用。

### attention_patterns.yaml

使用 Read 工具读取 `configs/attention_patterns.yaml`，获取注意力计算特征模式模板。根据识别的注意力类型（MHA/GQA/MLA/Mamba），选择对应的 flops_per_token 和 memory_per_token 公式，代入实际参数计算。

## 脚本

仅 `scripts/download_hf_config.py` 用于下载 HuggingFace config.json。

## 输出示例

见设计文档 `docs/superpowers/specs/2026-04-07-llm-roofline-analysis-design.md` 第 4 节。
