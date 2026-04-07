# LLM Roofline 性能分析工具

## 1. 概述

基于华为 Ascend 芯片的 vLLM 大模型整网 Roofline 性能分析工具。通过解析模型源码和配置，自动计算模型各模块的计算强度（Computational Intensity），对比硬件的算力与带宽，输出 Roofline 分析结果及端到端推理时延估算。

### 核心能力

- **Roofline 分析**: 识别模型各模块是 compute-bound 还是 memory-bound
- **时延估算**: 估算 Prefill 阶段和 Decode 阶段推理时延
- **多输入支持**: 支持 vLLM 源码路径、HuggingFace 模型名、用户指定 config.json

### AI 主导原则

- 源码解析、算子推导、公式计算、瓶颈分析全部由 **AI 主导**
- 脚本仅在必要时使用（如 HF 模型下载）
- 芯片性能参数以 JSON 配置为主，AI 读取使用

---

## 2. 输入规格

### 2.1 模型来源

| 输入类型 | 说明 | 示例 |
|---------|------|------|
| vLLM 源码路径 | 本地 vLLM 模型实现目录 | `/path/to/vllm/models/llama/` |
| HuggingFace 模型名 | HF Hub 上的模型名 | `Qwen/Qwen2.5-72B-Instruct` |
| 用户指定 config.json | 用户提供的模型配置文件 | `./my_model_config.json` |

### 2.2 用户需提供的参数

| 参数 | 说明 | 必填 |
|------|------|------|
| `prompt_len` | 输入 prompt 长度（token 数） | 是 |
| `gen_len` | 生成序列长度（token 数） | 是 |
| `chip` | 芯片型号，或提供自定义芯片参数 | 是 |

### 2.3 芯片参数（预置 + 自定义）

**预置芯片** (`configs/chips.json`):

```json
{
  "huawei": {
    "Ascend-910B-64GB": {
      "peak_fp32_flops_tflops": 256,
      "bandwidth_gb_s": 1200,
      "vram_gb": 64,
      "cards_per_node": 8
    },
    "Ascend-910C-64GB": {
      "peak_fp32_flops_tflops": 300,
      "bandwidth_gb_s": 1200,
      "vram_gb": 64,
      "cards_per_node": 16
    }
  },
  "nvidia": {
    "H100-80GB": {
      "peak_fp32_flops_tflops": 989,
      "bandwidth_gb_s": 3350,
      "vram_gb": 80,
      "cards_per_node": 8
    }
  }
}
```

**用户自定义芯片参数**（通过命令行或对话提供）:

- `peak_flops`: 峰值算力 (TFLOPS)
- `bandwidth`: 带宽 (GB/s)
- `vram`: 显存 (GB)
- `cards_per_node`: 单机卡数

---

## 3. 分析流程

### 3.1 模型结构解析（AI 主导）

AI 读取 vLLM 源码结构和 HuggingFace config.json，识别以下模块：

```
Model
├── Embedding
├── Transformer Layers (×N)
│   ├── Input LayerNorm
│   ├── Self Attention
│   │   ├── QKV Proj (MatMul)
│   │   ├── Core Attention (QK^T + Softmax + Score)
│   │   └── O Proj (MatMul)
│   ├── Post Attention LayerNorm
│   └── FFN
│       ├── Up Proj (MatMul)
│       ├── Gate Proj (MatMul)
│       └── Down Proj (MatMul)
├── Final LayerNorm
└── Output Head
```

**AI 任务**:
- 遍历 vLLM 源码目录，识别模块文件
- 提取每个 MatMul 的 weight shape
- 识别 FFN 类型（Standard / SwiGLU / MoE）
- 识别注意力机制类型（MHA / MQA / GQA / MLA）

### 3.2 算子 FLOPs 解析（AI 主导）

#### 标准算子 FLOPs 公式

| 算子 | FLOPs 公式 | 说明 |
|------|-----------|------|
| MatMul (QxK) | 2 × M × N × K | M=输出行, N=输出列, K=输入维度 |
| Attention QK^T | 2 × d² × s | d=hidden_dim, s=seq_len |
| Attention Score | 2 × d × s | Softmax 后加权求和 |
| FFN SwiGLU | 3 × d × ffn_dim | Up + Gate + Down 三个 MatMul |
| FFN Standard | 2 × d × ffn_dim | Up + Down 两个 MatMul |
| MoE Expert | 每个 Expert 独立计算后路由求和 | 动态路由 |

#### 融合算子处理

对于融合算子（如融合的 LayerNorm + SiLU），AI 需要：

1. **推导计算模式**: 分析源码中的融合逻辑，推导等效 FLOPs
2. **WebSearch 验证**: 搜索该融合算子在 Ascend 上的实际性能数据
3. **Fallback**: 若无法确定，使用简化模型估算

**AI 任务**:
- 从源码提取每个算子的具体 shape 参数
- 计算每个模块的总 FLOPs
- 识别融合算子，推导计算模式
- 必要时通过 WebSearch 验证融合算子性能

### 3.3 Roofline 分析（AI 主导）

#### 计算强度

```
I = FLOPs / 数据传输量 (Byte)
```

其中数据传输量包括：
- weight 访存: 权重字节数
- activation 访存: 激活值字节数
- KV cache 访存: KV cache 字节数（仅 Prefill 阶段）

#### 硬件 AI Balance

```
AI Balance = PeakFLOPs / Bandwidth = FLOPs/Byte
```

#### 瓶颈判定

| 条件 | 瓶颈类型 | 说明 |
|------|---------|------|
| I > AI Balance | Compute-Bound | 算力不足，带宽过剩 |
| I < AI Balance | Memory-Bound | 带宽不足，算力过剩 |
| I = AI Balance | Balanced | 算力与带宽匹配 |

**AI 任务**:
- 计算每个模块的计算强度 I
- 对比硬件 AI Balance
- 标注各模块的瓶颈类型
- 输出 Roofline 分析结果

### 3.4 时延估算（AI 主导）

#### Prefill 阶段时延

```
T_prefill = Σ (模块FLOPs[i] / 峰值算力) + Σ (数据传输量[i] / 带宽)
```

其中数据传输量包括 weight 访存和 activation 访存。

#### Decode 阶段时延

Decode 阶段逐 token 生成，每个 token 的时延包括：
- Attention 计算（与已生成 token 长度相关）
- FFN 前向计算
- KV cache 更新

```
T_decode = Σ (AttentionFLOPs(seq_len) / 峰值算力) +
           Σ (FFNFLOPs / 峰值算力) +
           Σ (KVCache 访存 / 带宽)
```

#### 端到端时延

```
T_total = T_prefill + gen_len × T_decode
```

**AI 任务**:
- 读取芯片配置（峰值算力、带宽）
- 根据 prompt_len 和 gen_len 计算各阶段时延
- 输出分阶段时延 + 端到端总时延

---

## 4. 输出规格

### 4.1 Roofline 分析报告

```markdown
## 模型: [模型名]
## 芯片: [芯片型号]

### 模块级分析

| 模块 | FLOPs (T) | 数据传输 (GB) | 计算强度 | 瓶颈类型 |
|------|-----------|--------------|---------|---------|
| Embedding | 0.01 | 0.5 | 0.02 | Memory-Bound |
| Attention | 2.5 | 1.2 | 2.08 | Compute-Bound |
| FFN | 5.3 | 2.1 | 2.52 | Compute-Bound |
| Output | 0.02 | 0.01 | 2.0 | Balanced |

### Roofline 图示

```
FLOPs/Byte
    │
    │         /─────── Compute-Bound 区间
    │        /
  200├───────
    │      │
    │      │  Memory-Bound 区间
    │      │
    └─────────────────────→ I (计算强度)
        20    100    200
```

### 整体评估

- 模型总计算强度: XXX FLOPs/Byte
- 硬件 AI Balance: XXX FLOPs/Byte
- 瓶颈分布: XX% Compute-Bound, XX% Memory-Bound
```

### 4.2 时延估算报告

```markdown
## 时延估算

### 硬件参数
- 芯片: Ascend-910B-64GB
- 峰值算力: 256 TFLOPS
- 带宽: 1200 GB/s

### 输入参数
- prompt_len: 4096
- gen_len: 1024

### 阶段时延

| 阶段 | 时延 |
|------|------|
| Prefill | 125 ms |
| Decode (单 token) | 2.3 ms |
| 端到端总时延 | 2487 ms |

### 时延分解

Prefill:
  - Embedding: 5 ms
  - Attention: 45 ms
  - FFN: 70 ms
  - Output: 5 ms

Decode (per token):
  - Attention: 0.8 ms
  - FFN: 1.2 ms
  - Output: 0.3 ms
```

---

## 5. 项目结构

```
llm_latency_estimator/
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-04-07-llm-roofline-analysis-design.md
├── configs/
│   ├── chips.json              # 预置芯片性能参数
│   └── ops_flops_formulas.yaml  # 标准算子 FLOPs 公式模板
├── llm_latency_estimator/
│   ├── __init__.py
│   └── report_generator.py      # 报告生成
├── scripts/
│   └── download_hf_config.py    # 仅用于下载 HF config.json
└── SKILL.md
```

---

## 6. 模块职责

| 模块 | 职责 | AI/脚本 |
|------|------|--------|
| 模型结构解析 | 读取 vLLM 源码，识别模块组织 | **AI** |
| HuggingFace config 解析 | 解析 HF config.json 参数 | **AI** |
| HF 模型下载 | 下载 HF 模型配置 | **脚本** |
| 算子 FLOPs 计算 | 分析源码提取 shape，计算 FLOPs | **AI** |
| 融合算子分析 | 推导融合逻辑，WebSearch 验证 | **AI** |
| Roofline 分析 | 计算强度 vs AI Balance | **AI** |
| 时延估算 | 根据公式计算各阶段时延 | **AI** |
| 报告生成 | 输出结构化分析报告 | **AI** |

---

## 7. 数据流

```dot
digraph dataflow {
    rankdir=LR;
    node [shape=box];

    input [label="用户输入\n(vLLM路径 / HF模型名 / config.json)"];
    model_parse [label="AI: 模型结构解析\n识别 Embedding/Attention/FFN/Output"];
    hf_fetch [label="脚本: HF config 下载\nAI: 解析 config.json"];
    op_flops [label="AI: 算子 FLOPs 解析\n提取 shape + 计算公式"];
    fused_ops [label="AI: 融合算子推导\n+ WebSearch 验证"];
    roofline [label="AI: Roofline 分析\n计算强度 vs AI Balance"];
    latency [label="AI: 时延估算\nPrefill + Decode + 总时延"];
    report [label="AI: 报告生成\nRoofline 图 + 时延分解"];

    input -> model_parse;
    input -> hf_fetch;
    model_parse -> op_flops;
    hf_fetch -> op_flops;
    op_flops -> fused_ops;
    fused_ops -> roofline;
    roofline -> latency;
    latency -> report;
}
```

---

## 8. 关键技术决策

### 8.1 为什么 AI 主导

- vLLM 模型结构多样，自动解析源码比预定义模板更通用
- 融合算子形式多变，需要 AI 结合源码推导和搜索验证
- Roofline 分析需要理解计算图，AI 能更好地识别瓶颈根因

### 8.2 脚本最小化

仅在以下场景使用脚本：
- `huggingface-cli download` / `curl` 下载 HF 模型
- 批量文件操作（必要时）

### 8.3 芯片配置 JSON 化

芯片参数（峰值算力、带宽）以 JSON 存储，AI 直接读取使用，便于扩展新芯片。

---

## 9. 约束与限制

- 本工具仅适用于 **推理场景**，不适用于训练
- 时延估算是 **理论估算**，实际性能受kernel实现、并行策略等因素影响
- 融合算子的推导依赖 AI 分析能力，可能存在误差
- 需要用户提供正确的模型来源路径或 HuggingFace 模型名

---

## 10. 后续步骤

1. 实现模型结构解析模块
2. 实现算子 FLOPs 解析模块
3. 实现 Roofline 分析模块
4. 实现时延估算模块
5. 完善报告生成
6. 测试验证
