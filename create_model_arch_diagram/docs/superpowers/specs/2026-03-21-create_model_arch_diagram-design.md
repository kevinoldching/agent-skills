# Model Architecture Diagram Skill - Design Specification

## Overview

A Claude Code skill that generates professional model architecture diagrams from open-source models or user-defined configurations. Supports multiple output formats (PNG/SVG/Mermaid) for use in documentation, papers, and presentations.

## Architecture

```
User Input (HuggingFace ID / Local Path / YAML Config)
         │
         ▼
┌─────────────────────────┐
│    Parser Module         │
│  - HuggingFace Parser    │
│  - Local File Parser    │
│  - YAML Config Parser   │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│   Template Engine        │
│  - Model Type Detection  │
│  - Template Selection    │
│  - Parameter Filling     │
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│   Mermaid Generator      │
│  - Generate Mermaid Syntax│
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│   Renderer Module       │
│  - PNG Rendering        │
│  - SVG Rendering        │
│  - Mermaid Output       │
└─────────────────────────┘
```

## Input Sources

| Source | Handling |
|--------|----------|
| HuggingFace Model ID | Download config.json from HuggingFace Hub, cache to `templates/{family}/{model_name}.yaml` |
| Local Model Path | Read local config.json |
| YAML Config File | User-defined model structure (fully customizable) |

## Template System

### Template Storage

```
templates/
├── llama/
│   ├── common.yaml         # Generic LLaMA structure rules
│   └── llama-3-8b.yaml     # Generated instance config
├── mistral/
│   └── mistral-7b.yaml
├── qwen/
│   └── qwen2.yaml
├── glm/
│   └── glm4.yaml
├── baichuan/
│   └── baichuan3.yaml
├── mimo/
│   └── mimo.yaml
├── kimi/
│   └── kimi.yaml
├── minimax/
│   └── minimax.yaml
└── gpt-oss/
    └── gpt-oss.yaml
```

### Template Definition Format

Each template YAML defines only the differentiated parts. AI auto-fills the rest.

```yaml
# llama-3.yaml example
model_type: llama3
family: llama

block:
  - type: attention
    components:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
  - type: ffn
    components:
      - gate_proj
      - up_proj
      - down_proj

stack:
  num_layers_key: num_hidden_layers
  pattern: [block] × N

residual: pre-norm
```

### AI Auto-Completion

From config.json parameters + model type knowledge:

| Information | Source |
|-------------|--------|
| hidden_size, num_layers, intermediate_size | config.json (direct read) |
| num_heads, head_dim | config.json or calculation (hidden_size / num_heads) |
| Norm type (RMSNorm / LayerNorm) | Model type knowledge base |
| Activation function (SiLU / GELU) | config.json or model type conventions |
| Residual connection type (pre-norm / post-norm) | Model type knowledge base |
| Module connections | Model type knowledge base |

### Shape Calculation

- Attention weights: `[hidden_size, hidden_size]`
- FFN first layer: `[hidden_size, intermediate_size]`
- Input/Output: annotated as `hidden_size`

## Supported Model Templates

Initial template library includes:
- LLaMA series (LLaMA-2, LLaMA-3)
- Mistral series
- Qwen series
- GLM series
- Baichuan series
- Mimo series
- Kimi series
- MiniMax series
- GPT-OSS series

## Output Formats

| Format | Generation |
|--------|------------|
| PNG | Mermaid CLI rendering |
| SVG | Mermaid CLI rendering |
| Mermaid Syntax | Direct text output (user can copy and modify) |

### Output File Naming

```
{output_dir}/
├── {model_name}_arch.png
├── {model_name}_arch.svg
└── {model_name}_arch.mmd   # Mermaid source
```

### Output Directory

- User-specified via `--output /path/to/dir`
- Default: Claude Code current working directory

User's config files (YAML) are stored in skill's template directory:
```
templates/{family}/{model_name}.yaml
```

## User-Defined Model Config Format

### Full Specification:

```yaml
model_name: my-custom-model

blocks:
  - name: encoder
    type: transformer_block
    layers: 12
    hidden_size: 768
    intermediate_size: 3072
    num_heads: 12

  - name: decoder
    type: transformer_block
    layers: 12
    hidden_size: 768

connections:
  - from: encoder
    to: decoder

norm: rmsnorm
activation: silu
```

### Minimal Specification (AI fills defaults):

```yaml
model_name: my-custom-model
hidden_size: 768
num_layers: 24
intermediate_size: 3072
num_heads: 12
activation: silu
norm: rmsnorm
```

## Skill Invocation Interface

```
/create_model_arch_diagram <model_id_or_path> [--format png,svg,mmd] [--output /path/to/dir]

# Example:
/create_model_arch_diagram meta-llama/Llama-3-8b --format png,svg
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model` | HuggingFace ID / local path / YAML config | Required |
| `--format` | Output formats (comma-separated) | png,svg,mmd |
| `--output` | Output directory | Current working directory |

## Rendering Solution

Tool: `@mermaid-js/mermaid-cli` (npm package)

- Official Mermaid CLI tool
- Widely adopted in the ecosystem
- Good PNG/SVG quality
- Requires Node.js environment

## Workflow Summary

```
1. User input model ID / path / YAML
2. Download config.json (if HuggingFace)
3. Parse config + match template
4. Generate / update template YAML in skill's templates directory
5. Fill Mermaid template with parameters
6. Render to PNG/SVG (if requested)
7. Output files to user-specified directory
```

## File Structure

```
create_model_arch_diagram/
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-03-21-create_model_arch_diagram-design.md
├── templates/                    # Model templates (knowledge base)
│   ├── llama/
│   ├── mistral/
│   ├── qwen/
│   ├── glm/
│   ├── baichuan/
│   ├── mimo/
│   ├── kimi/
│   ├── minimax/
│   └── gpt-oss/
├── src/
│   ├── parser/                   # Input parsing
│   ├── template_engine/          # Template handling
│   ├── mermaid_generator/        # Mermaid syntax generation
│   └── renderer/                 # PNG/SVG rendering
└── skill.yaml                    # Skill definition
```
