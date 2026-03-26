# LLM Architecture Generator Implementation Plan (2026-03-26 Design)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the 2026-03-26 design spec: new HuggingFace downloader with repo scanning, Level 1/Level 2 detail views with left-right/top-down layouts, shape inference from model.py, and natural language invocation.

**Architecture:** Two-phase implementation: (1) Create download_model.py script tool, (2) Rewrite SKILL.md with new AI instructions for parsing, analysis, and Mermaid generation.

**Tech Stack:** Python (download_model.py), Bash/PS1 (render scripts), Mermaid syntax, HuggingFace hub API

---

## File Structure

```
llm_arch_generator/
├── SKILL.md                              # Rewrite: new AI instructions (main entry)
├── scripts/
│   ├── download_model.py                 # Create: HuggingFace downloader with repo scanning
│   └── render_mermaid.sh                 # Existing: unchanged
└── templates/                             # Existing: unchanged
    └── {family}/common.yaml
```

---

## Task 1: Create download_model.py

**Files:**
- Create: `scripts/download_model.py`

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Download config.json and model.py from HuggingFace with caching.

model.py can be anywhere in the repo tree. This script uses list_repo_files()
to scan the entire repository (all subdirectories) and locate the modeling file.
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

CACHE_DIR = Path.home() / ".cache" / "llm_arch_generator"

def get_cache_path(model_id: str, filename: str) -> Path:
    """Get local cache path for a downloaded file."""
    safe_id = model_id.replace('/', '_').replace('-', '_')
    return CACHE_DIR / safe_id / filename

def find_modeling_file(model_id: str) -> str | None:
    """
    Find the modeling file by scanning the entire repository.

    Uses list_repo_files() to get all files recursively, then matches:
      - model.py
      - modeling.py
      - modeling_<anything>.py

    This handles any directory layout (src/, inference/, flash_attention/, etc.).
    Returns the full path in the repo if found, None otherwise.
    """
    try:
        all_files = list_repo_files(model_id)
    except Exception:
        return None

    modeling_patterns = [
        'model.py',
        'modeling.py',
    ]

    for f in all_files:
        filename = os.path.basename(f)
        if filename in modeling_patterns:
            return f
        if filename.startswith('modeling_') and filename.endswith('.py'):
            return f

    return None

def download_model(
    model_id: str,
    output_dir: str = None,
    use_cache: bool = True
) -> tuple[str, str | None]:
    """
    Download config.json and model.py from HuggingFace.

    Caches to ~/.cache/llm_arch_generator/{model_id}/
    Clear cache by deleting that directory.
    """
    if output_dir is None:
        out_path = CACHE_DIR / model_id.replace('/', '_').replace('-', '_')
    else:
        out_path = Path(output_dir)

    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Download config.json
    config_cache = get_cache_path(model_id, "config.json")
    if use_cache and config_cache.exists():
        import shutil
        dest = out_path / "config.json"
        shutil.copy(config_cache, dest)
        config_path = str(dest)
    else:
        config_path = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            local_dir=str(out_path)
        )
        config_cache.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(config_path, str(config_cache))

    # 2. Find and download modeling file
    modeling_filename = find_modeling_file(model_id)

    model_path = None
    if modeling_filename:
        model_cache = get_cache_path(model_id, modeling_filename.replace('/', '_'))
        if use_cache and model_cache.exists():
            import shutil
            dest = out_path / os.path.basename(modeling_filename)
            shutil.copy(model_cache, dest)
            model_path = str(dest)
        else:
            try:
                model_path = hf_hub_download(
                    repo_id=model_id,
                    filename=modeling_filename,
                    local_dir=str(out_path)
                )
                model_cache.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy(model_path, str(model_cache))
            except Exception:
                model_path = None

    return config_path, model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model files from HuggingFace")
    parser.add_argument("model_id", help="e.g., meta-llama/Llama-3-8b")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: cache)")
    parser.add_argument("--no-cache", action="store_true", help="Bypass cache")
    args = parser.parse_args()

    config, model = download_model(
        args.model_id,
        args.output_dir,
        use_cache=not args.no_cache
    )
    print(f"config.json: {config}")
    print(f"modeling_*.py: {model}")
```

- [ ] **Step 2: Test the script**

Run: `python scripts/download_model.py meta-llama/Llama-3-8b --no-cache`
Expected: Downloads config.json and model.py to cache, prints paths

- [ ] **Step 3: Test caching**

Run: `python scripts/download_model.py meta-llama/Llama-3-8b`
Expected: Uses cached files, prints paths without re-downloading

- [ ] **Step 4: Commit**

```bash
git add scripts/download_model.py
git commit -m "feat: add HuggingFace downloader with repo scanning for model.py"
```

---

## Task 2: Rewrite SKILL.md - New AI Instructions

**Files:**
- Modify: `SKILL.md` (complete rewrite)

**Summary of changes from old SKILL.md:**
- Add natural language interpretation
- Add `-v` / `-vv` detail level flags
- Level 1: left-right (graph LR) with dashed transformer block
- Level 2: top-down (graph TD) with module expansion via `==>`
- Shape inference from model.py weight definitions
- Residual detection from forward() method analysis
- Auto-completion based on model family conventions

- [ ] **Step 1: Write new SKILL.md with failing tests**

The "test" here is attempting to use the skill against the new spec. Write the complete new SKILL.md:

```markdown
---
name: llm-arch-generator
description: Generate professional LLM model architecture diagrams from HuggingFace models or YAML configs. Supports PNG/SVG/Mermaid output formats.
---

# LLM Architecture Generator

## Overview

A Claude Code skill that generates professional model architecture diagrams from HuggingFace models, local model files, or user-defined configurations. Supports multiple output formats (PNG/SVG/Mermaid) and two detail levels: collapsed (-v) and expanded (-vv).

## Invocation Syntax

### Standard Invocation

```
/llm-arch-generator <model> [-v|-vv] [--format png,svg,mmd] [--output /path/to/dir]
```

### Natural Language Invocation

When users describe what they want in natural language, the skill interprets:

| User says | Interpreted as |
|-----------|----------------|
| "Draw / generate / plot the architecture of {model}" | Standard generation with `-vv` |
| "Simple / high-level / macro / collapsed view" | `-v` |
| "Detailed / expanded / with projections" | `-vv` |
| "Save to {path}" | `--output /path` |
| "PNG / SVG / Mermaid format" | `--format` |

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model` | HuggingFace ID, local path, or YAML file | Required |
| `-v` | Level 1: collapsed block structure with residual connections | — |
| `-vv` | Level 2: expanded module internals (default) | Default |
| `--format` | Output formats (comma-separated) | png,svg,mmd |
| `--output` | Output directory | Current directory |

### Examples

```bash
# Default (-vv): expanded view
/llm-arch-generator KimiML/kimi-k2-5

# Level 1 (-v): collapsed with residual connections
/llm-arch-generator meta-llama/Llama-3-8b -v

# Level 2 (-vv): explicit expanded view
/llm-arch-generator Qwen/Qwen2-7B -vv

# With output format
/llm-arch-generator Qwen/Qwen2-7B --format png --output ./diagrams

# Natural language equivalents
/llm-arch-generator Draw a detailed architecture diagram for Kimi-K2.5
/llm-arch-generator Generate a simple macro view of LLaMA-3
/llm-arch-generator Plot the architecture of Qwen2-7B and save to ./qwen_arch
```

---

## Detail Levels

### `-v` (Collapsed, Level 1)

Shows individual **Attention**, **FFN** (or **MoE**, **MLP**) modules visible in the flow, with a dashed box grouping them labeled `× N layers`. Residual connections are shown at this level.

**Key:** Level 1 directly shows Attention/FFN/MoE/MLP modules in the flow (not a black-box "Stack"), with a dashed border indicating they repeat N times.

### `-vv` (Expanded, Level 2)

Top-down layout. Level 1 shows one transformer layer path (left/top). Complex modules (Attention, MoE/FFN) expand to detailed views (right/bottom) via `==>` arrows.

---

## Information Extraction

### HuggingFace Model Processing

When a HuggingFace model ID is provided:

1. **Download files** using `scripts/download_model.py`:
   ```bash
   python scripts/download_model.py <model_id> --output-dir /tmp/model_download
   ```
2. **Parse config.json**: Extract H, I, num_heads, kv_heads, layers
3. **Read model.py** (if available): Build module tree, trace forward path

### Shape Inference: config.json + model.py Combined

Shape inference **must combine** both sources — not config.json alone.

**From config.json:**
- `hidden_size` (H)
- `num_hidden_layers`
- `intermediate_size` (I)
- `num_attention_heads`
- `num_key_value_heads` (for GQA)
- `head_dim` = H / num_attention_heads

**From model.py:**

model.py contains **full tensor definitions** that enable precise shape calculation:

```python
# Example from modeling_llama.py
class LlamaAttention(nn.Module):
    def __init__(self, config):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # These are the actual tensor shapes defined in code:
        self.q_proj = nn.Linear(H, H)           # [H, H]
        self.k_proj = nn.Linear(H, kvH * head_dim)  # [H, kvH * head_dim]
        self.v_proj = nn.Linear(H, kvH * head_dim)  # [H, kvH * head_dim]
        self.o_proj = nn.Linear(H, H)            # [H, H]
```

**Correct shape annotations:**

| Layer | Shape |
|-------|-------|
| Q_proj weight | `[H, H]` or `[H, num_heads × head_dim]` |
| K_proj weight | `[H, num_key_value_heads × head_dim]` |
| V_proj weight | `[H, num_key_value_heads × head_dim]` |
| O_proj weight | `[H, num_heads × head_dim]` |
| Attention output (after softmax) | `[B, num_heads, S, head_dim]` |
| FFN gate/up | `[H, intermediate_size]` |
| FFN down | `[intermediate_size, H]` |

### Residual Connection Detection from model.py

Residual connections must be derived from **actual code analysis**, not assumptions:

**Pre-norm pattern (LLaMA, Qwen, Kimi):**

```python
# Identified from model.py forward():
input = layer_norm(input)
output = attention(input)
input = input + output          # residual add here
output = mlp(input)
input = input + output          # residual add here
input = layer_norm(input)
```

**Post-norm pattern (GLM, some GPT variants):**

```python
# Identified from model.py forward():
output = attention(input)
input = norm(input + output)    # residual then norm
```

AI must read the actual `forward()` method and identify:
- Which tensor flows into which module
- Where `add` / `+` / `subtract` operations occur
- Which tensors are added together (residual source and destination)
- Conditional branches (training vs inference paths)

---

## Mermaid Syntax Generation

### Level 1: Block Structure with Residual Connections (graph LR)

```mermaid
graph LR
    E["Embedding"] --> LN1["RMSNorm"]
    LN1 --> Input["Input"]

    subgraph TB["Transformer Block × 32"]
        direction LR
        style TB dashed
        Input --> LN_a["RMSNorm"]
        LN_a --> Attn["Attention"]
        Input -.->|add| Add1["Add"]
        Attn --> Add1

        Add1 --> LN_b["RMSNorm"]
        LN_b --> FFN["FFN"]
        Add1 -.->|add| Add2["Add"]
        FFN --> Add2
        Add2 --> Out["Output"]
    end

    Out --> LN2["RMSNorm"]
    LN2 --> LM["LM Head"]
```

**Key:** `subgraph TB` with `style TB dashed` shows the repeated modules with a dashed border. Attention and FFN are visible inside — NOT hidden in a black-box node.

### Level 2: Module Expansion (graph TD)

Level 1 structure (top-down), with complex modules (MoE, Attention) expanded on the right via `==>` relationship arrows.

```mermaid
graph TD
    %% === 颜色定义 ===
    classDef attention fill:#e1f5ff,stroke:#01579b,stroke-width:2px;
    classDef moe fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef shared_expert fill:#b2dfdb,stroke:#00695c,stroke-width:2px;
    classDef ffn fill:#fff4e1,stroke:#333,stroke-width:2px;
    classDef norm fill:#f1f8e9,stroke:#33691e,stroke-width:1px;
    classDef input_stage fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;
    classDef output_stage fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;

    %% === 输入层 (Level 1) ===
    subgraph Input_Stage ["输入层"]
        direction TB
        Tokens["Token IDs"] --> Embed["Embedding"]
    end

    %% === Layer N 结构 (Level 1) ===
    subgraph Transformer_Layer ["Transformer Layer N"]
        direction TB

        layer_in((h_l)):::norm --> ln1["RMSNorm"]:::norm
        ln1 --> attn_module["MLA / Attention"]:::attention

        attn_module --> add1((+)):::norm
        layer_in -.-> |Residual 1| add1

        add1 --> ln2["RMSNorm"]:::norm
        ln2 --> moe_module["DeepSeekMoE / FFN"]:::moe

        moe_module --> add2((+)):::norm
        add1 -.-> |Residual 2| add2

        add2 --> layer_out((h_l+1)):::norm
    end

    %% === MoE 展开 (Level 2) ===
    subgraph MoE_Detail ["MoE 展开"]
        direction TB

        router["Sigmoid Router"]:::moe

        subgraph Expert_Pool ["专家池 (Shared + 8 Routed)"]
            shared["Shared Expert"]:::shared_expert
            routed_1["Routed Expert 1"]:::moe
            routed_2["Routed Expert 2"]:::moe
            routed_x["..." ]:::moe
            routed_n["Routed Expert N"]:::moe
        end

        router --> |Top-8| routed_1
        router --> |Top-8| routed_2
        router --> |Top-8| routed_n
        shared -.-> |always add| MoE_out
        routed_1 -.-> |if selected| MoE_out
        routed_2 -.-> |if selected| MoE_out
        routed_n -.-> |if selected| MoE_out
    end

    subgraph Attention_Detail ["Attention 展开"]
        direction TB

        attn_in["Input<br/>[B,S,H]"] --> q_proj["Q_proj<br/>H×H"]:::attention
        attn_in --> k_proj["K_proj<br/>H×kvH·head"]:::attention
        attn_in --> v_proj["V_proj<br/>H×kvH·head"]:::attention

        q_proj --> softmax["Softmax<br/>Q·Kᵀ/√d"]:::attention
        k_proj --> softmax

        softmax --> o_proj["O_proj<br/>hH×H"]:::attention
        o_proj --> attn_out["Output<br/>[B,S,H]"]:::attention
    end

    %% === 输出层 ===
    subgraph Output_Stage ["输出层"]
        direction TB
        final_norm["Final RMSNorm"]:::norm
        Head["LM Head"]:::output_stage
    end

    %% === 全局连接 ===
    Embed --> layer_in
    layer_out --> final_norm
    final_norm --> Head

    %% === 展开关系 (Level 1 ==> Level 2) ===
    moe_module ==> router
    attn_module ==> attn_in
```

### Color Conventions

Mermaid `classDef` style definitions:

```mermaid
classDef attention fill:#e1f5ff,stroke:#01579b,stroke-width:2px;
classDef moe fill:#fff3e0,stroke:#e65100,stroke-width:2px;
classDef shared_expert fill:#b2dfdb,stroke:#00695c,stroke-width:2px;
classDef ffn fill:#fff4e1,stroke:#333,stroke-width:2px;
classDef norm fill:#f1f8e9,stroke:#33691e,stroke-width:1px;
classDef input_stage fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;
classDef output_stage fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;
```

| Module Type | Fill | Border | Usage |
|-------------|------|--------|-------|
| Attention | #e1f5ff | #01579b | MLA, Q/K/V/O projections, Softmax |
| MoE | #fff3e0 | #e65100 | Router, Routed Experts |
| Shared Expert | #b2dfdb | #00695c | Shared Expert (always active) |
| FFN / MLP | #fff4e1 | #333 | gate/up/down_proj |
| Norm | #f1f8e9 | #33691e | RMSNorm, LayerNorm |
| Input/Output | #f3e5f5 | #4a148c | Embedding, LM Head |
| Residual | dashed | #999 | `-.->` arrows |
| Expand relation | solid bold | — | `==>` arrows (Level 1 → Level 2) |

---

## AI-Generated Components

| Component | Responsibility |
|-----------|----------------|
| **Parser** | AI reads config.json, extracts H, I, num_heads, kv_heads, layers |
| **Model Analyzer** | AI reads model.py, builds module tree, traces forward path |
| **Residual Detector** | AI reads forward() method, identifies add/shortcut operations |
| **Shape Calculator** | AI computes shapes from model.py weight definitions × config.json params |
| **Mermaid Generator** | AI generates left-right syntax per detail level |
| **Auto-completion** | AI fills missing info based on model family conventions |

---

## Workflow

```
1. User invokes /llm-arch-generator <model> [options]
            │
            ▼
2. Parse invocation (standard CLI or natural language)
            │
            ▼
3. (If HuggingFace) Script: download_model.py
   - config.json → parse H, I, num_heads, kv_heads, layers
   - model.py → scan repo with list_repo_files() to locate, then download
            │
            ▼
4. AI: Read model.py
   - Build module tree (Attention, FFN/MoE/MLP, projections)
   - Trace forward() path and residual connections
            │
            ▼
5. AI: Calculate shapes
   - Weight shapes from model.py (Linear layers)
   - Activation shapes from config.json (H, I, head_dim)
   - Propagation: [B, S, H] through each op
            │
            ▼
6. AI: Generate Mermaid syntax
   - Left-right layout (graph LR) for -v
   - Top-down layout (graph TD) with module expansion for -vv
   - Respects -v/-vv detail level
            │
            ▼
7. Write {model_name}_arch.mmd
            │
            ▼
8. (If --format includes png/svg) Script: render_mermaid.sh → PNG/SVG
            │
            ▼
9. Output files to {output_dir}/
```

### Fallback Path

If model.py is not available (only权重 files):
- AI infers structure from config.json + model family template
- Shape calculations use family conventions (H, I, head_dim relationships)
- Residual patterns use family defaults (pre-norm for LLaMA, post-norm for GLM)
- Note: precision reduced, model.py analysis preferred

---

## Input Handling

### HuggingFace Model ID

1. Download `config.json` and `model.py` using `scripts/download_model.py`
2. Cache to `~/.cache/llm_arch_generator/{model_id}/`
3. Parse config.json to extract model parameters
4. Match against known model family templates

### Local File Path

1. If path points to a directory, read `config.json` from that directory
2. If path points to a `.yaml` file, treat as YAML config

### YAML Config Direct Input

```yaml
model_name: my-custom-model
hidden_size: 768
num_layers: 24
intermediate_size: 3072
num_attention_heads: 12
num_key_value_heads: 32
activation: silu
norm: rmsnorm
```

---

## Template Matching

### Supported Model Families

| Family | Key Characteristics |
|--------|-------------------|
| LLaMA | RMSNorm, SiLU, pre-norm |
| Mistral | LLaMA derivative, GQA support |
| Qwen | LLaMA derivative, SiLU |
| GLM | Post-norm residual |
| MoE (DeepSeek) | Shared Expert + Routed Experts |

### Auto-Completion Rules

| Information | Source | Fallback |
|-------------|--------|----------|
| hidden_size | config.json direct read | Required |
| num_layers | config.json direct read | Required |
| intermediate_size | config.json direct read | 4 × hidden_size |
| num_attention_heads | config.json or calculation | hidden_size / head_dim |
| num_key_value_heads | config.json | num_attention_heads (no GQA) |
| Norm type | Model type knowledge | RMSNorm |
| Activation function | config.json or model type | SiLU for LLaMA, GELU for GPT |
| Residual connection type | Model type knowledge | pre-norm |

---

## Output Files

```
{output_dir}/
├── {model_name}_arch.png    # Rendered raster image
├── {model_name}_arch.svg    # Rendered vector image
└── {model_name}_arch.mmd    # Mermaid source (always generated)
```

---

## Mermaid CLI Rendering

### Prerequisites

- Node.js installed
- `@mermaid-js/mermaid-cli` installed (via npm or npx)

### Linux/macOS

```bash
# Global install
npm install -g @mermaid-js/mermaid-cli

# Or use via npx
npx @mermaid-js/mermaid-cli mmdc --version
```

### Windows

Use `scripts/render_mermaid.ps1`:

```powershell
.\render_mermaid.ps1 -Input "diagram.mmd" -OutputPng "diagram.png"
```

### Manual Rendering

```bash
# Using mermaid-cli directly
mmdc -i model_diagram.mmd -o model_diagram.png -b transparent -w 1920
mmdc -i model_diagram.mmd -o model_diagram.svg -b transparent -w 1920 -f svg
```

---

## File Structure

```
llm_arch_generator/
├── SKILL.md                         # AI instructions (main entry)
├── scripts/
│   ├── download_model.py            # HuggingFace file downloader (with repo scanning)
│   ├── render_mermaid.sh            # Bash renderer (Linux/macOS)
│   └── render_mermaid.ps1          # PowerShell renderer (Windows)
└── templates/                       # Model family templates
    ├── llama/common.yaml
    ├── mistral/common.yaml
    ├── qwen/common.yaml
    ├── glm/common.yaml
    └── ...
```

---

## Summary: AI vs Script Responsibilities

| Task | Responsibility |
|------|----------------|
| Parse config.json | **AI** |
| Locate model.py in repo (scan with list_repo_files) | **Script** |
| Read model.py | **AI** (directly reads file) |
| Analyze module hierarchy | **AI** |
| Trace forward path | **AI** |
| Detect residual connections | **AI** (from model.py forward() analysis) |
| Calculate tensor shapes | **AI** (from model.py weight definitions × config.json params) |
| Generate Mermaid syntax | **AI** |
| Interpret natural language | **AI** |
| Download HuggingFace files | **Script** (Python) |
| Render PNG/SVG | **Script** (Bash + mermaid-cli) |
| Auto-fill missing parameters | **AI** (from model family knowledge) |
```

- [ ] **Step 2: Verify SKILL.md is syntactically valid**

The SKILL.md is a markdown file, so validation is manual review. Read the file back and check:
- All mermaid code blocks are properly fenced
- All tables are properly formatted
- All links and references are correct

- [ ] **Step 3: Commit**

```bash
git add SKILL.md
git commit -m "feat: rewrite SKILL.md with Level 1/2 detail views and natural language support"
```

---

## Task 3: Test End-to-End

**Files:**
- Test: HuggingFace model (e.g., meta-llama/Llama-3-8b)

- [ ] **Step 1: Test Level 1 (-v) output**

Invoke the skill with `-v` flag on a known model, verify:
- graph LR layout used
- Dashed transformer block with × N label
- Attention and FFN visible inside the block
- Residual connections shown with `-.->`

- [ ] **Step 2: Test Level 2 (-vv) output**

Invoke the skill with `-vv` flag on a known model, verify:
- graph TD layout used
- Module expansion with `==>` arrows
- Attention expands to Q/K/V/O projections with shapes
- MoE expands to router + expert pool

- [ ] **Step 3: Test natural language interpretation**

Test interpretations:
- "Draw architecture of Llama-3" → equivalent to `-vv`
- "Simple view of Llama-3" → equivalent to `-v`
- "Save to ./output" → equivalent to `--output ./output`

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "test: add end-to-end tests for Level 1/Level 2 diagrams"
```

---

## Task 4: Create test template for MoE models

**Files:**
- Modify: `templates/deepseek/common.yaml` (if exists)
- Create: `templates/deepseek/deepseek-v3.yaml` (if not exists)

- [ ] **Step 1: Verify MoE template exists**

Check if `templates/deepseek/deepseek-v3.yaml` exists and supports:
- Shared Expert + Routed Experts structure
- Router component
- Expert pool rendering

- [ ] **Step 2: Update if needed**

If template is missing or incomplete, add support for:
- MoE block type with shared_expert and routed_experts
- Router projection
- Top-K selection visualization

- [ ] **Step 3: Commit**

```bash
git add templates/deepseek/
git commit -m "feat: add MoE template support for DeepSeek-V3 architecture"
```

---

## Execution Summary

| Task | Files | Key Changes |
|------|-------|-------------|
| 1. download_model.py | scripts/download_model.py (new) | HuggingFace downloader with list_repo_files scanning |
| 2. SKILL.md rewrite | SKILL.md (rewrite) | Level 1/2 views, LR/TD layouts, natural language |
| 3. End-to-end test | — | Verify diagrams render correctly |
| 4. MoE template | templates/deepseek/*.yaml | MoE architecture support |
