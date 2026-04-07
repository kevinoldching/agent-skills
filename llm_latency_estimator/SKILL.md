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
6. 识别注意力类型：检查 attention 模块代码，判断是 MHA / GQA / MLA / Mamba或其它类型
```

### 2. 算子 FLOPs 动态推导

**必须从源码分析计算模式，不要依赖预置公式**。

```
必须执行的操作：
1. 使用 Read 工具读取 vLLM 源码中的 attention 实现（如 attention.py, flash_attention.py）
2. 分析 attention 计算流程：
   - 找出 QKV projection 的 shape (hidden_dim → d_k, d_v)
   - 找出 attention score 计算方式 (MatMul 还是其他)
   - 确定 KV cache 的 shape 和访问模式
3. 分析 FFN 计算流程：
   - 找出 up_proj, gate_proj, down_proj 的 shape
   - 判断是否有 SiLU/Swish 激活
4. AI 根据实际代码推导出该模型的 FLOPs 公式
5. AI 根据 KV cache 形状计算内存访问量（Bytes）
```

**AI 应自行分析的内容**：
- Attention: QKV projection FLOPs = 2 × in_dim × out_dim × 3 (Q, K, V 分别计算)
- Attention Score: 根据实际实现（MatMul 或其他）推导
- FFN SwiGLU: Up + Gate + Down = 3 × hidden_dim × intermediate_size
- FFN Standard: Up + Down = 2 × hidden_dim × intermediate_size

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

**⚠️ 警告：以下公式中的 FLOPs 和 Bytes 值必须来自第2步的源码分析结果，不得直接从 config.json 计算。**

完成第2步后，AI 已推导出：
- `FLOPs_attn(seq_len)` — Attention 模块的 FLOPs（与序列长度相关）
- `Bytes_kvcache(seq_len)` — Attention 模块的 KV cache 内存访问量
- `FLOPs_ffn` — FFN 模块的 FLOPs
- `Bytes_ffn_weights` — FFN 模块的权重内存访问量

将这些值代入以下公式：

```
# 单模块耗时（严格使用 max，非相加）
T_module = max(FLOPs_module / PeakFLOPs, Bytes_module / Bandwidth)

# Prefill 阶段（所有 Transformer 层串行）
T_prefill = Σ_layers T_module(prompt_len)

# Decode 阶段（逐 token 生成，seq_len 随已生成长度变化）
T_decode = Σ_layers (T_attn_layer + T_ffn_layer)

# 端到端总时延
T_total = T_prefill + gen_len × T_decode
```

**数据来源检查清单**：
- [ ] `FLOPs_attn` 是否来自源码中 MatMul shape 分析？
- [ ] `Bytes_kvcache` 是否来自源码中 KV cache 访问模式分析？
- [ ] `FLOPs_ffn` 是否来自源码中 FFN shape 分析？
- [ ] `Bytes_ffn_weights` 是否来自源码中权重 size 分析？

### 5. 优化建议生成

基于瓶颈分析，AI 输出针对性建议：
- Memory-Bound 模块: 量化、KV cache 压缩、Flash Attention、算子融合等
- Compute-Bound 模块: Tensor Core 优化、精度选择、CANN 融合算子等

### 6. 输出格式

报告包含：
1. **Roofline 分析**: Mermaid xychart 图表 + 模块分析表
2. **时延估算**: Prefill / Decode / 总时延分解
3. **优化建议**: 分模块针对性建议

## 配置文件

### chips.json

使用 Read 工具读取 `references/chips.json`，获取芯片参数：
- 峰值算力 `peak_fp16_tflops`
- 带宽 `bandwidth_gb_s`

如果用户指定了芯片名称（如 Ascend-910B-64GB），从 chips.json 中查找对应参数。如果用户提供了自定义参数（peak_flops, bandwidth），直接使用。

## 脚本

仅 `scripts/download_hf_config.py` 用于下载 HuggingFace config.json。

## 输出示例

输出格式模板见 `references/output_format.md`，包含：
1. **Roofline 分析**: 模块分析表 + ASCII 图表
2. **时延估算**: Prefill / Decode / 总时延分解
3. **优化建议**: 分模块针对性建议
