# LLM Roofline 性能分析工具 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现基于 Ascend 芯片的 vLLM 大模型 Roofline 性能分析 Skill，支持多输入源、AI 动态推导、Mermaid 可视化。

**Architecture:** 这是一个以 AI 推理为核心的 Skill，核心逻辑由 SKILL.md 定义，配置文件存储芯片参数和注意力模式模板，脚本仅用于 HF 模型下载。AI 完成模型解析、FLOPs 推导、Roofline 分析、时延估算和优化建议生成。

**Tech Stack:** Python（仅脚本）、YAML/JSON（配置）、SKILL.md（Skill 定义）

---

## File Structure

```
llm_latency_estimator/
├── SKILL.md                              # 核心 Skill 定义（AI 行为指南）
├── configs/
│   ├── chips.json                       # 预置芯片性能参数
│   └── attention_patterns.yaml          # 注意力计算特征模式模板
├── scripts/
│   └── download_hf_config.py            # 下载 HF config.json
├── llm_latency_estimator/
│   └── __init__.py                      # 包初始化
└── docs/superpowers/
    ├── specs/
    │   └── 2026-04-07-llm-roofline-analysis-design.md  # 设计文档
    └── plans/
        └── 2026-04-07-llm-roofline-analysis-implementation-plan.md  # 本计划
```

**Note:** 不同于 llm_mem_estimator 的 CLI 工具模式，此 Skill 以 SKILL.md 为核心，AI 直接读取配置和源码进行推理，脚本最小化。

---

## Task 1: 创建 SKILL.md

**Files:**
- Create: `llm_latency_estimator/SKILL.md`

SKILL.md 是此工具的核心，定义 AI 如何进行 Roofline 分析。参考 llm_mem_estimator/SKILL.md 的格式。

- [ ] **Step 1: 创建 SKILL.md 文件**

```markdown
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
```

- [ ] **Step 2: 验证 SKILL.md 格式**

检查 frontmatter 完整（name, description），无语法错误。

Run: `python3 -c "import yaml; yaml.safe_load(open('llm_latency_estimator/SKILL.md').read().split('---')[2])" if needed`
Expected: YAML 解析成功

- [ ] **Step 3: Commit**

```bash
git add llm_latency_estimator/SKILL.md
git commit -m "feat: add SKILL.md for roofline analysis"
```

---

## Task 2: 创建 chips.json

**Files:**
- Create: `llm_latency_estimator/configs/chips.json`

扩展 llm_mem_estimator 的芯片配置，增加峰值算力参数（用于 Roofline 分析）。

- [ ] **Step 1: 创建 configs 目录和 chips.json**

```json
{
  "huawei": {
    "Ascend-910B-64GB": {
      "vendor": "huawei",
      "model": "Ascend-910B",
      "vram_gb": 64,
      "peak_fp32_tflops": 256,
      "bandwidth_gb_s": 1200,
      "cards_per_node": 8
    },
    "Ascend-910C-64GB": {
      "vendor": "huawei",
      "model": "Ascend-910C",
      "vram_gb": 64,
      "peak_fp32_tflops": 300,
      "bandwidth_gb_s": 1200,
      "cards_per_node": 16
    },
    "Ascend-910C-32GB": {
      "vendor": "huawei",
      "model": "Ascend-910C",
      "vram_gb": 32,
      "peak_fp32_tflops": 300,
      "bandwidth_gb_s": 1200,
      "cards_per_node": 16
    }
  },
  "nvidia": {
    "H100-80GB": {
      "vendor": "nvidia",
      "model": "H100",
      "vram_gb": 80,
      "peak_fp32_tflops": 989,
      "bandwidth_gb_s": 3350,
      "cards_per_node": 8
    },
    "A100-80GB": {
      "vendor": "nvidia",
      "model": "A100",
      "vram_gb": 80,
      "peak_fp32_tflops": 312,
      "bandwidth_gb_s": 2039,
      "cards_per_node": 8
    }
  }
}
```

- [ ] **Step 2: 验证 JSON 格式**

Run: `python3 -c "import json; print(json.load(open('llm_latency_estimator/configs/chips.json')).keys())"`
Expected: `dict_keys(['huawei', 'nvidia'])`

- [ ] **Step 3: Commit**

```bash
git add llm_latency_estimator/configs/chips.json
git commit -m "feat: add chips.json with peak flops and bandwidth"
```

---

## Task 3: 创建 attention_patterns.yaml

**Files:**
- Create: `llm_latency_estimator/configs/attention_patterns.yaml`

存储注意力计算的特征模式，供 AI 动态调用。

- [ ] **Step 1: 创建 attention_patterns.yaml**

```yaml
# 注意力机制计算特征模式模板
# AI 根据注意力类型代入 config 中的实际参数进行计算
# 不要在此文件中写死具体数值

attention_patterns:
  MHA:
    # Multi-Head Attention (Standard)
    description: "标准多头注意力，K/V 与 Q 同维度，shape = [batch, n_heads, seq_len, d_head]"
    kv_cache_shape: "[batch, n_heads, seq_len, d_head]"
    flops_per_token: |
      QKV_proj: 2 × hidden_dim × hidden_dim × 3
      Attention_score: 2 × n_heads × d_head × seq_len
      Softmax: 3 × n_heads × seq_len
      Score_weighted_sum: 2 × n_heads × d_head × seq_len
      Output_proj: 2 × hidden_dim × hidden_dim
    memory_per_token: |
      KV_cache_read: 2 × n_heads × seq_len × d_head × bytes_per_element
      KV_cache_write: 2 × n_heads × d_head × bytes_per_element

  GQA:
    # Grouped Query Attention
    description: "分组查询注意力，n_kv_heads < n_q_heads，K/V heads 数量减少"
    kv_cache_shape: "[batch, n_kv_heads, seq_len, d_head]"
    kv_ratio: "n_kv_heads / n_q_heads (通常 1/4 到 1/8)"
    flops_per_token: |
      Q_proj: 2 × hidden_dim × hidden_dim
      KV_proj: 2 × hidden_dim × (n_kv_heads × d_head)
      Attention_score (with KV sharing): 2 × n_kv_heads × d_head × seq_len
      Score_weighted_sum: 2 × n_kv_heads × d_head × seq_len
      Output_proj: 2 × hidden_dim × hidden_dim
    memory_per_token: |
      KV_cache_read: 2 × n_kv_heads × seq_len × d_head × bytes_per_element (reduced vs MHA)
      KV_cache_write: 2 × n_kv_heads × d_head × bytes_per_element

  MLA:
    # Multi-head Latent Attention (DeepSeek V3)
    description: "低秩压缩注意力，KV cache 压缩到 d_compressed 维度"
    kv_cache_shape: "[batch, seq_len, d_compressed]"
    compression_ratio: "d_compressed / (2 × n_heads × d_head)"
    flops_per_token: |
      Q_proj: 2 × hidden_dim × d_qa
      KV_proj (compressed): 2 × hidden_dim × d_compressed
      Attention_score: 2 × d_compressed × seq_len
      Output_proj: 2 × hidden_dim × d_compressed
    memory_per_token: |
      KV_cache_read: d_compressed × seq_len × bytes_per_element (significant reduction)
      KV_cache_write: d_compressed × bytes_per_element

  Mamba:
    # State Space Model (Mamba)
    description: "状态空间模型，非标准 attention，需要分析 SSM 算子实现"
    kv_cache_shape: "[batch, state_dim, seq_len] (non-standard)"
    note: "需 AI 从源码推导 FLOPs，依赖于 state_dim 和 seq_len"
    flops_per_token: |
      SSM_forward: O(state_dim × seq_len)
      具体 FLOPs 需分析 vLLM 中 Mamba 实现源码
```

ffn_patterns:
  # FFN 类型模式
  standard:
    description: "标准 FFN: Up_proj + Down_proj"
    flops: "2 × hidden_dim × intermediate_size"

  swiglu:
    description: "SwiGLU FFN: Gate_proj + SiLU(Up_proj) ⊙ Down_proj"
    flops: "3 × hidden_dim × intermediate_size"

  moe:
    description: "MoE FFN: 多个 Expert 分别计算后加权求和"
    flops: "2 × hidden_dim × intermediate_size × n_experts × top_k (实际只有 top_k Expert 计算)"
```

- [ ] **Step 2: 验证 YAML 格式**

Run: `python3 -c "import yaml; print(list(yaml.safe_load(open('llm_latency_estimator/configs/attention_patterns.yaml'))['attention_patterns'].keys()))"`
Expected: `['MHA', 'GQA', 'MLA', 'Mamba']`

- [ ] **Step 3: Commit**

```bash
git add llm_latency_estimator/configs/attention_patterns.yaml
git commit -m "feat: add attention_patterns.yaml with MHA/GQA/MLA/Mamba templates"
```

---

## Task 4: 创建 scripts/download_hf_config.py

**Files:**
- Create: `llm_latency_estimator/scripts/download_hf_config.py`

仅用于下载 HuggingFace模型的 config.json，保持脚本最小化。

- [ ] **Step 1: 创建目录和脚本**

```python
#!/usr/bin/env python3
"""
Download HuggingFace model config.json for LLM Roofline Analysis.

Usage:
    python scripts/download_hf_config.py Qwen/Qwen2.5-72B-Instruct
    python scripts/download_hf_config.py Qwen/Qwen2.5-72B-Instruct --output /path/to/save
"""

import argparse
import json
import os
import sys
from pathlib import Path

def download_config(model_name: str, output_path: str | None = None) -> dict:
    """Download config.json from HuggingFace model hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Error: huggingface_hub not installed. Install with: pip install huggingface_hub")
        sys.exit(1)

    config_path = hf_hub_download(repo_id=model_name, filename="config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    if output_path:
        output_file = Path(output_path) / "config.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to {output_file}")
    else:
        print(json.dumps(config, indent=2))

    return config

def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace model config.json")
    parser.add_argument("model", help="HuggingFace model name (e.g., Qwen/Qwen2.5-72B-Instruct)")
    parser.add_argument("--output", "-o", help="Output directory to save config.json")
    args = parser.parse_args()

    download_config(args.model, args.output)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证脚本语法**

Run: `python3 -m py_compile llm_latency_estimator/scripts/download_hf_config.py`
Expected: 无输出（成功）

- [ ] **Step 3: Commit**

```bash
git add llm_latency_estimator/scripts/download_hf_config.py
git commit -m "feat: add download_hf_config.py for HF model config"
```

---

## Task 5: 创建 llm_latency_estimator/__init__.py

**Files:**
- Create: `llm_latency_estimator/llm_latency_estimator/__init__.py`

包初始化文件，保持最小化。

- [ ] **Step 1: 创建 __init__.py**

```python
"""
LLM Roofline Analysis Tool

AI-driven performance analysis for LLM inference on Ascend chips.
Generates Roofline analysis, latency estimates, and optimization suggestions.
"""

__version__ = "0.1.0"
```

- [ ] **Step 2: Commit**

```bash
git add llm_latency_estimator/llm_latency_estimator/__init__.py
git commit -m "feat: add package init for llm_latency_estimator"
```

---

## Task 6: 整体验证

- [ ] **Step 1: 验证项目结构**

Run: `find llm_latency_estimator -type f | sort`
Expected:
```
llm_latency_estimator/
llm_latency_estimator/SKILL.md
llm_latency_estimator/configs/
llm_latency_estimator/configs/chips.json
llm_latency_estimator/configs/attention_patterns.yaml
llm_latency_estimator/llm_latency_estimator/
llm_latency_estimator/llm_latency_estimator/__init__.py
llm_latency_estimator/scripts/
llm_latency_estimator/scripts/download_hf_config.py
```

- [ ] **Step 2: 验证所有配置文件可解析**

Run: `python3 -c "import yaml, json; yaml.safe_load(open('llm_latency_estimator/configs/attention_patterns.yaml')); json.load(open('llm_latency_estimator/configs/chips.json')); print('All configs OK')"`
Expected: `All configs OK`

- [ ] **Step 3: Commit 验证**

```bash
git add -A
git commit -m "chore: verify project structure completeness"
```

---

## 后续步骤（AI 主导，无需预先实现）

Skill 完成后，AI 可直接用于分析：

1. **用户提供 vLLM 源码路径 + HuggingFace 模型名**
2. **AI 读取 chips.json 获取芯片参数**
3. **AI 读取 attention_patterns.yaml 获取计算模式**
4. **AI 从 config.json 提取参数，动态计算 FLOPs 和 Bytes**
5. **AI 执行 Roofline 分析，生成 Mermaid 图表**
6. **AI 计算时延，输出优化建议**

无需额外实现代码，AI 依据 SKILL.md 的指引即可完成完整分析流程。

---

## Self-Review Checklist

### Spec Coverage
- [x] 模型结构解析（SKILL.md Section 1）
- [x] 注意力类型识别（SKILL.md Section 1 + attention_patterns.yaml）
- [x] 算子 FLOPs 动态推导（SKILL.md Section 2 + attention_patterns.yaml）
- [x] Roofline 分析 max() 公式（SKILL.md Section 3）
- [x] 时延估算公式（SKILL.md Section 4）
- [x] 优化建议生成（SKILL.md Section 5）
- [x] Mermaid xychart 输出（SKILL.md Section 6）
- [x] 多输入源支持（SKILL.md description）
- [x] chips.json 预置芯片 + 自定义支持（SKILL.md + configs/chips.json）
- [x] HF 下载脚本（scripts/download_hf_config.py）

### Placeholder Scan
- [x] 无 "TBD"、"TODO" 占位符
- [x] 所有步骤包含实际代码和命令
- [x] 所有文件路径为确定值

### Type Consistency
- [x] chips.json 字段: peak_fp32_tflops, bandwidth_gb_s（统一后缀）
- [x] attention_patterns.yaml: attention_patterns 和 ffn_patterns 分开定义
- [x] SKILL.md 中的公式与设计文档一致（max 非相加）
