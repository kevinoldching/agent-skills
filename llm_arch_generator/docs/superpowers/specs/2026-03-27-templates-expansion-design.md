# Templates Expansion Design

## Context

The `llm_arch_generator/templates/` directory contains model family templates used by the AI to infer model architecture when HuggingFace analysis is unavailable. Currently:
- 10 model families exist, but most only have `common.yaml` (scaffold only)
- `SKILL.md` references `./templates/` at line 90 but has **no code** to actually load or match templates
- The `template:` inheritance field in existing yaml files is **dead code** — never parsed

## Goal

1. Expand `templates/` to cover all major models across the 10 families
2. Make `SKILL.md` actually reference `templates/` so AI can use them
3. Validate each `common.yaml` is precise enough for its family variants

---

## Templates Data Model

Each model is described by a YAML file with this schema:

```yaml
model_type: <identifier>          # e.g., qwen3, deepseek_v2, llama3
family: <family-name>             # e.g., qwen, deepseek, llama
model_name: <human-name>          # e.g., Qwen3-32B

source: <huggingface-id>          # e.g., Qwen/Qwen3-32B

config:
  hidden_size: <int>
  num_hidden_layers: <int>
  intermediate_size: <int>        # dense FFN intermediate size
  num_attention_heads: <int>
  num_key_value_heads: <int>
  vocab_size: <int>
  head_dim: <int>                 # optional, can be derived
  activation: <silu|gelu|relu>    # optional, defaults to silu
  rms_norm_eps: <float>            # optional
  rope_theta: <float>              # optional
  max_position_embeddings: <int>  # optional

  # Attention variants (one of):
  attention_type: <standard|gqa|mla|sliding_window>
  sliding_window: <int>            # if sliding_window attention
  kv_lora_rank: <int>              # if MLA
  q_lora_rank: <int>               # if MLA
  qk_rope_head_dim: <int>          # if MLA
  qk_nope_head_dim: <int>          # if MLA
  v_head_dim: <int>                # if MLA

  # MoE (if applicable)
  ffn_type: <dense|moe>            # default: dense
  n_routed_experts: <int>          # if MoE
  num_experts_per_tok: <int>       # if MoE
  moe_intermediate_size: <int>     # if MoE
  n_shared_experts: <int>          # if MoE (default: 1)
  first_k_dense_replace: <int>    # if MoE (layers before MoE starts)

  # Vision (if multimodal)
  vision_hidden_size: <int>
  vision_num_layers: <int>
  vision_patch_size: <int>

template: <family>/common.yaml     # reference to common.yaml
```

### `common.yaml` Schema

```yaml
model_type: <family-model-type>
family: <family-name>

block:
  - type: <attention|dense_ffn|moe_ffn>
    components:
      - <component-names>

stack:
  num_layers_key: <string>         # e.g., num_hidden_layers
  pattern: <string>                # e.g., "[block] × N"

residual: <pre-norm|post-norm>
norm: <rmsnorm|layernorm>
activation: <silu|gelu|relu>

input: embed_tokens
output: lm_head

kv_heads_key: <string>            # optional, e.g., num_key_value_heads

# Attention type declaration
attention_impl: <standard|gqa|mla|sliding_window>  # family-default

# MoE declaration
moe:
  n_routed_experts_key: <string>
  num_experts_per_tok_key: <string>
  moe_intermediate_size_key: <string>
  n_shared_experts_key: <string>  # optional
  first_k_dense_replace_key: <string>  # optional

# Vision (if family supports multimodal)
vision:
  encoder_type: <siglip|clip|other>
  projector_type: <linear|mlp>
```

---

## Models to Add

### deepseek/

| File | Model | Key Config |
|------|-------|------------|
| `common.yaml` | — | MoE scaffold, MLA, pre-norm |
| `deepseek-v3.yaml` | **already exists** | 61L, 7168H, 256 experts, first_k_dense_replace=3 |
| `deepseek-v3.2.yaml` | **NEW** | same as V3, 685B params |
| `deepseek-v3.1.yaml` | **NEW** | variant |
| `deepseek-v2.5.yaml` | **NEW** | 60L, 5120H, 160 experts, n_shared_experts=2, first_k_dense_replace=1 |
| `deepseek-v2.yaml` | **NEW** | base for V2.5 |
| `deepseek-r1.yaml` | **NEW** | reasoning, same arch as V3 |

### qwen/

| File | Model | Key Config |
|------|-------|------------|
| `common.yaml` | — | needs update: add GQA kv_heads_key, sliding_window |
| `qwen3-32b.yaml` | **NEW** | 64L, 5120H, GQA (8 KV heads), no MoE |
| `qwen3-8b.yaml` | **NEW** | smaller variant |
| `qwen3-4b.yaml` | **NEW** | |
| `qwen3-1.7b.yaml` | **NEW** | |
| `qwen3-0.6b.yaml` | **NEW** | |
| `qwen3-vl-8b.yaml` | **NEW** | + vision encoder |
| `qwen3.5-397b-a17b.yaml` | **NEW** | MoE, 512 experts, 60L, multimodal |
| `qwen3.5-122b-a10b.yaml` | **NEW** | MoE variant |
| `qwen3.5-35b-a3b.yaml` | **NEW** | MoE variant |
| `qwen3.5-27b.yaml` | **NEW** | dense |
| `qwen2.5-vl-7b.yaml` | **NEW** | vision-language |

### llama/

| File | Model | Key Config |
|------|-------|------------|
| `common.yaml` | — | pre-norm, rmsnorm, silu |
| `llama-2-7b.yaml` | **already exists** | 32L, 4096H |
| `llama-3-8b.yaml` | **already exists** | GQA (8 KV heads) |
| `llama-3.1-8b.yaml` | **NEW** | 32L, GQA |
| `llama-3.2-1b.yaml` | **NEW** | small variant |
| `llama-3.2-3b.yaml` | **NEW** | |
| `llama-3.2-11b.yaml` | **NEW** | |
| `llama-3.3-70b.yaml` | **NEW** | large variant |

### kimi/

| File | Model | Key Config |
|------|-------|------------|
| `common.yaml` | — | MLA, MoE, pre-norm (update to reflect K2.x) |
| `kimi-k2.5.yaml` | **NEW** | 61L, 7168H, 384 experts, MLA, multimodal |
| `kimi-k2.0.yaml` | **NEW** | earlier version |

### minimax/

| File | Model | Key Config |
|------|-------|------------|
| `common.yaml` | — | MoE, pre-norm, needs update |
| `minimax-m2.yaml` | **NEW** | 62L, 3072H, 256 experts, GQA |
| `minimax-m2.5.yaml` | **NEW** | |

### glm/

| File | Model | Key Config |
|------|-------|------------|
| `common.yaml` | — | post-norm, layernorm, gelu — **correct** |
| `glm-4-9b.yaml` | **NEW** | 40L, 4096H, GQA |
| `glm-4-28b.yaml` | **NEW** | 62L |
| `glm-4v.yaml` | **NEW** | + vision |
| `glm-4.7b.yaml` | **already exists** | |
| `glm-5.yaml` | **already exists** | |

### baichuan/

| File | Model | Key Config |
|------|-------|------------|
| `common.yaml` | — | pre-norm, rmsnorm, silu — **correct** |
| `baichuan-13b.yaml` | **NEW** | 40L, 4096H |
| `baichuan2-7b.yaml` | **NEW** | |
| `baichuan-m3.yaml` | **NEW** | MoE variant |

### gpt-oss/

| File | Model | Key Config |
|------|-------|------------|
| `common.yaml` | — | post-norm, layernorm, gelu — **correct** |
| `gpt-oss-20b.yaml` | **NEW** | 48L |
| `gpt-oss-120b.yaml` | **already exists** | |

### mistral/

| File | Model | Key Config |
|------|-------|------------|
| `common.yaml` | — | pre-norm, rmsnorm, sliding_window — **correct** |
| `mistral-7b-v0.1.yaml` | **NEW** | sliding_window=4096 |
| `mistral-7b-v0.3.yaml` | **NEW** | |
| `mistral-nemo.yaml` | **NEW** | 12B |
| `mistral-large.yaml` | **NEW** | |

### mimo/ (Xiaomi MiMo)

| File | Model | Key Config |
|------|-------|------------|
| `common.yaml` | — | MoE, pre-norm, sigmoid routing |
| `mimo-v2-flash.yaml` | **NEW** | 48L, 4096H, 256 experts, sigmoid routing, sliding_window=128 |
| `mimo-vl-7b.yaml` | **NEW** | vision-language |

---

## SKILL.md Changes

### Step 0 — Data Source Options

Two paths only (Built-in YAML = templates/):

```
1. HuggingFace — Download + analyze model.py (network, most accurate)
2. Built-in Templates — Match model name to templates/ (no network, family-level accuracy)
```

Note: `llm_mem_estimator/configs/models/` is unrelated to this skill — it belongs to the llm_mem_estimator skill.

### Branch B — Template Inference

Line 90: Change "Branch B — YAML Config" to "Branch B — Built-in Templates":

```
**Branch B — Built-in Templates:**
Match model name/family to templates/{family}/:
- Read family common.yaml to get block structure and residual pattern
- Read model yaml to get H, layers, attention type, MoE config
- Infer module internals (Q/K/V/O projections, gate/up/down_proj)
- For variants not in templates: use common.yaml + family conventions
```

### Add Template Matching Logic

When user selects "Template Inference" (or HuggingFace fails):
1. Extract model family from model name (e.g., "Qwen3-32B" → "qwen")
2. Try exact match: `templates/{family}/{model-name}.yaml`
3. Fall back to fuzzy: scan `templates/*/common.yaml` for matching `model_type`
4. Load matched yaml, merge with common.yaml

---

## common.yaml Validation Checklist

For each family, common.yaml must accurately describe:

- [ ] `residual` pattern (pre-norm vs post-norm)
- [ ] `norm` type (rmsnorm vs layernorm)
- [ ] `activation` (silu vs gelu)
- [ ] `attention_impl` matches family default (standard/GQA/MLA/sliding_window)
- [ ] `kv_heads_key` declared if non-standard
- [ ] `moe:` section declared with correct keys if family uses MoE
- [ ] Vision sections for multimodal families (kimi, qwen, glm)

---

## Files to Create/Modify

| Action | File |
|--------|------|
| CREATE | `templates/deepseek/deepseek-v3.2.yaml` |
| CREATE | `templates/deepseek/deepseek-v3.1.yaml` |
| CREATE | `templates/deepseek/deepseek-v2.5.yaml` |
| CREATE | `templates/deepseek/deepseek-v2.yaml` |
| CREATE | `templates/deepseek/deepseek-r1.yaml` |
| UPDATE | `templates/deepseek/common.yaml` — add DeepSeek-V2 MoE keys |
| CREATE | `templates/qwen/qwen3-32b.yaml` |
| CREATE | `templates/qwen/qwen3-8b.yaml` |
| CREATE | `templates/qwen/qwen3-4b.yaml` |
| CREATE | `templates/qwen/qwen3-1.7b.yaml` |
| CREATE | `templates/qwen/qwen3-0.6b.yaml` |
| CREATE | `templates/qwen/qwen3-vl-8b.yaml` |
| CREATE | `templates/qwen/qwen3.5-397b-a17b.yaml` |
| CREATE | `templates/qwen/qwen3.5-122b-a10b.yaml` |
| CREATE | `templates/qwen/qwen3.5-35b-a3b.yaml` |
| CREATE | `templates/qwen/qwen3.5-27b.yaml` |
| CREATE | `templates/qwen/qwen2.5-vl-7b.yaml` |
| UPDATE | `templates/qwen/common.yaml` — add kv_heads_key, sliding_window |
| CREATE | `templates/llama/llama-3.1-8b.yaml` |
| CREATE | `templates/llama/llama-3.2-1b.yaml` |
| CREATE | `templates/llama/llama-3.2-3b.yaml` |
| CREATE | `templates/llama/llama-3.2-11b.yaml` |
| CREATE | `templates/llama/llama-3.3-70b.yaml` |
| UPDATE | `templates/llama/common.yaml` — add kv_heads_key for GQA |
| CREATE | `templates/kimi/kimi-k2.5.yaml` |
| CREATE | `templates/kimi/kimi-k2.0.yaml` |
| UPDATE | `templates/kimi/common.yaml` — update for MLA+MoE K2.x |
| CREATE | `templates/minimax/minimax-m2.yaml` |
| CREATE | `templates/minimax/minimax-m2.5.yaml` |
| UPDATE | `templates/minimax/common.yaml` — update for MoE/GQA |
| CREATE | `templates/glm/glm-4-9b.yaml` |
| CREATE | `templates/glm/glm-4-28b.yaml` |
| CREATE | `templates/glm/glm-4v.yaml` |
| CREATE | `templates/baichuan/baichuan-13b.yaml` |
| CREATE | `templates/baichuan/baichuan2-7b.yaml` |
| CREATE | `templates/baichuan/baichuan-m3.yaml` |
| CREATE | `templates/mistral/mistral-7b-v0.1.yaml` |
| CREATE | `templates/mistral/mistral-7b-v0.3.yaml` |
| CREATE | `templates/mistral/mistral-nemo.yaml` |
| CREATE | `templates/mistral/mistral-large.yaml` |
| UPDATE | `templates/mistral/common.yaml` — add sliding_window attention |
| CREATE | `templates/mimo/common.yaml` |
| CREATE | `templates/mimo/mimo-v2-flash.yaml` |
| CREATE | `templates/mimo/mimo-vl-7b.yaml` |
| UPDATE | `SKILL.md` Step 0 and Branch B |
