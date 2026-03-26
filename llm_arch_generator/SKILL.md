---
name: llm-arch-generator
description: Use when user asks to draw, plot, generate, or visualize LLM model architecture, or wants to understand how a model (like LLaMA, MoE, DeepSeek, GPT) is structured internally. Also use for comparing model architectures, explaining attention/FFN/MoE modules, or generating architecture diagrams for documentation.
---

# LLM Architecture Generator

## Invocation

```
/llm-arch-generator <model> [-v|-vv] [--format png,svg,mmd] [--output /path/to/dir]
```

**Natural language mapping:**

| User says | Interpreted as |
|-----------|---------------|
| "Draw/plot/generate architecture of {model}" | `-vv` (expanded) |
| "Simple/high-level/macro/collapsed view" | `-v` (collapsed) |
| "Detailed/expanded/with projections" | `-vv` |
| "Save to {path}" | `--output /path` |

**Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model` | HuggingFace ID, local path, or YAML config | Required |
| `-v` | Level 1: collapsed blocks with residual connections (graph LR) | — |
| `-vv` | Level 2: expanded internals with projections (graph TD) | Default |
| `--format` | Output formats | png,svg,mmd |
| `--output` | Output directory | CWD |

### Examples

```bash
# Default: expanded view for DeepSeek V3 (MoE architecture)
/llm-arch-generator deepseek-ai/DeepSeek-V3-Base -vv

# Collapsed view for simple models
/llm-arch-generator gpt2 -v

# With PNG output
/llm-arch-generator Qwen/Qwen2-7B --format png --output ./diagrams

# Natural language
/llm-arch-generator Draw the architecture of LLaMA-3 and show me the projections
```

---

## Detail Levels

### `-v` (Level 1: Collapsed)

- `graph LR` layout
- Dashed `subgraph` border for repeated transformer blocks
- Attention/FFN/MoE visible inside, NOT black-boxed
- Residual `-.->` arrows shown explicitly

### `-vv` (Level 2: Expanded)

- `graph TD` layout
- Complex modules expand via `==>` arrows
- Projection layers visible (Q/K/V/O, gate/up/down, router/experts)

---

## Workflow

1. **Resolve model ID**: If user provides a model name (e.g., "Kimi-K2.5", "LLaMA-3"), search for the HuggingFace model ID first
   - Use web search to find the official HuggingFace repository
   - Common patterns: `moonshotai/Kimi-K2.5`, `meta-llama/Llama-3-8b`, `openai-community/gpt2`
   - If multiple matches exist, use the official/original model
2. **Download**: Once resolved, download via `scripts/download_model.py`
   - Scans repo with `list_repo_files()` to find `modeling_*.py`
   - Caches to `~/.cache/llm_arch_generator/{model_id}/`
3. **Read model.py**: Build module tree, trace `forward()` path
3. **Detect residual connections** from actual code analysis
4. **Calculate shapes**: Combine `config.json` params + `model.py` weight definitions
5. **Generate Mermaid**: Level 1 (`graph LR`) or Level 2 (`graph TD`)
6. **Render**: PNG/SVG via `scripts/render_mermaid.sh`

**Fallback**: If no `model.py`, infer from config.json + model family template.

**Error handling:**
- Model name not recognized → Search web for HuggingFace ID, ask user to confirm if ambiguous
- HuggingFace download fails → Use cached files if available, or try alternative approach
- `modeling_*.py` not found → Still generate diagram from config.json + template, note precision reduced
- Rendering fails (missing Chrome) → Still generate `.mmd` file, user can render manually

---

## Color Conventions

```mermaid
classDef attention fill:#e1f5ff,stroke:#01579b,stroke-width:2px
classDef moe fill:#fff3e0,stroke:#e65100,stroke-width:2px
classDef shared_expert fill:#b2dfdb,stroke:#00695c,stroke-width:2px
classDef ffn fill:#fff4e1,stroke:#333,stroke-width:2px
classDef norm fill:#f1f8e9,stroke:#33691e,stroke-width:1px
classDef input_stage fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
classDef output_stage fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
```

| Module | Fill | Border |
|--------|------|--------|
| Attention | #e1f5ff | #01579b |
| MoE | #fff3e0 | #e65100 |
| Shared Expert | #b2dfdb | #00695c |
| FFN/MLP | #fff4e1 | #333 |
| Norm | #f1f8e9 | #33691e |
| Input/Output | #f3e5f5 | #4a148c |
| Residual | dashed | #999 |
| Expand `==>` | solid bold | — |

---

## MoE Rendering

For MoE models (DeepSeek V3, Mixtral, etc.):

**Key components to render:**
- **Router**: Sigmoid/MLP that selects top-K experts
- **Shared Expert**: Always active, rendered with distinct color
- **Routed Experts**: Selected by router, shown as pool

**Example structure:**
```
router --> |Top-8| routed_1
router --> |Top-8| routed_2
shared -.-> |always add| MoE_out
routed_1 -.-> |if selected| MoE_out
```

**DeepSeek V3 specific:**
- First 3 layers are dense, then MoE layers start
- Uses MLA (Multi-head Latent Attention) instead of standard attention
- 256 routed experts + 1 shared expert, top-8 selection

---

## Output Files

```
{output_dir}/
├── {model_name}_arch.png
├── {model_name}_arch.svg
└── {model_name}_arch.mmd  (always generated)
```

---

## Reference

**Full details** (including complete mermaid syntax examples, shape inference methodology, residual detection patterns, and model family conventions): `docs/superpowers/specs/2026-03-26-llm_arch_generator-design.md`

**When to read the spec:**
- Need complete mermaid diagram examples → read spec lines 185-307
- Implementing shape inference → read spec lines 92-153
- Understanding residual patterns → read spec lines 155-181
- Template matching → read spec lines 449-506
