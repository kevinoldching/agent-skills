# LLM Roofline 性能分析工具 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现基于 Ascend 芯片的 vLLM 大模型 Roofline 性能分析 Skill，支持多输入源、AI 动态推导源码计算模式、Mermaid 可视化。

**Architecture:** 这是一个以 AI 推理为核心的 Skill，核心逻辑由 SKILL.md 定义。AI 直接读取 vLLM 源码分析 FLOPs 模式，chips.json 存储芯片参数，脚本仅用于 HF 模型下载。

**Tech Stack:** Python（仅脚本）、JSON（配置）、SKILL.md（Skill 定义）

---

## File Structure

```
llm_latency_estimator/
├── SKILL.md                              # 核心 Skill 定义（AI 行为指南）
├── configs/
│   └── chips.json                       # 预置芯片性能参数
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

**Note:** 此 Skill 以 SKILL.md 为核心，AI 直接读取 vLLM 源码分析计算模式，不依赖预置公式模板。脚本最小化。

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

## Task 3: 创建 scripts/download_hf_config.py

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
import sys
from pathlib import Path

def download_config(model_name: str, output_path: str | None = None) -> dict | None:
    """Download config.json from HuggingFace model hub. Returns None on failure."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Error: huggingface_hub not installed. Install with: pip install huggingface_hub")
        return None

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

    result = download_config(args.model, args.output)
    if result is None:
        sys.exit(1)

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

## Task 4: 创建 llm_latency_estimator/__init__.py

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

## Task 5: 整体验证

- [ ] **Step 1: 验证项目结构**

Run: `find llm_latency_estimator -type f ! -path '*/__pycache__/*' ! -path '*/.git/*' ! -path '*/docs/*' | sort`
Expected:
```
llm_latency_estimator/
llm_latency_estimator/SKILL.md
llm_latency_estimator/configs/
llm_latency_estimator/configs/chips.json
llm_latency_estimator/llm_latency_estimator/
llm_latency_estimator/llm_latency_estimator/__init__.py
llm_latency_estimator/scripts/
llm_latency_estimator/scripts/download_hf_config.py
```

- [ ] **Step 2: 验证 JSON 配置可解析**

Run: `python3 -c "import json; json.load(open('llm_latency_estimator/configs/chips.json')); print('chips.json OK')"`
Expected: `chips.json OK`

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
3. **AI 读取 vLLM 源码，动态分析 FLOPs 计算模式**
4. **AI 从 config.json 提取参数，动态计算 FLOPs 和 Bytes**
5. **AI 执行 Roofline 分析，生成 Mermaid 图表**
6. **AI 计算时延，输出优化建议**

无需额外实现代码，AI 依据 SKILL.md 的指引读取源码即可完成完整分析流程。

---

## Self-Review Checklist

### Spec Coverage
- [x] 模型结构解析（SKILL.md Section 1）
- [x] 算子 FLOPs 动态推导（SKILL.md Section 2，AI 从源码分析，不依赖预置模板）
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
- [x] SKILL.md 中的公式与设计文档一致（max 非相加）
