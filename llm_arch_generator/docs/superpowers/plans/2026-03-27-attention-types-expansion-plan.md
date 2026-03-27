# Attention Types Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand llm_arch_generator's attention type support to cover DSA, SWA, Hybrid Parallel/Alternating/Chain modes, with explicit attention_type declarations in every template YAML.

**Architecture:** Three-layer approach — (1) Schema: YAML `attention_type` / `attention_hybrid_types` fields; (2) Mermaid generation: AI infers structure from config; (3) Verification: regex-based structural checks in verify_mermaid.py.

**Tech Stack:** Python 3, YAML templates, Mermaid diagrams, regex-based validation.

---

## File Map

### Scripts
- `scripts/verify_mermaid.py` — connectivity checker, needs regex rules replacing hardcoded whitelist

### SKILL.md
- `SKILL.md` — mermaid generation logic needs attention_hybrid_types branches + "Expected false positives" update

### Template YAMLs (56 files across 10 families)
| Family | Files | Attention Type |
|--------|-------|----------------|
| `templates/deepseek/` | common.yaml, deepseek-v3.yaml, deepseek-v3.1.yaml, deepseek-v3.2.yaml, deepseek-v2.yaml, deepseek-v2.5.yaml, deepseek-r1.yaml | MLA / DSA |
| `templates/kimi/` | common.yaml, kimi-k2.0.yaml, kimi-k2.5.yaml | MLA |
| `templates/qwen/` | common.yaml, + 11 model files | GQA |
| `templates/llama/` | common.yaml, + 7 model files | standard / GQA |
| `templates/glm/` | common.yaml, glm-4-9b.yaml, glm-4-28b.yaml, glm-4v.yaml, glm-5.yaml | GQA / DSA |
| `templates/baichuan/` | common.yaml, baichuan-13b.yaml, baichuan2-7b.yaml, baichuan-m3.yaml | standard / GQA |
| `templates/mistral/` | common.yaml, + 4 model files | SWA |
| `templates/minimax/` | common.yaml, minimax-m2.yaml, minimax-m2.5.yaml | GQA |
| `templates/mimo/` | common.yaml, mimo-v2-flash.yaml, mimo-vl-7b.yaml | SWA |
| `templates/gpt-oss/` | common.yaml, gpt-oss-120b.yaml, gpt-oss-20b.yaml | Hybrid Alternating |

### MMD diagrams (verify after generation)
- `deepseek_v3_2_arch.mmd` (create new — existing `deepseek_v3_arch.mmd` is for V3 MLA, V3.2 uses DSA)
- `gpt_oss_120b_arch.mmd`, `gpt_oss_20b_arch.mmd` (verify hybrid alternating structure)

---

## Task 1: Update verify_mermaid.py — regex-based extensibility

**Files:**
- Modify: `scripts/verify_mermaid.py` (find and replace TERMINAL_NODES and ALWAYS_ALONE_NODES sets)

- [ ] **Step 1: Read current verify_mermaid.py to find TERMINAL_NODES and ALWAYS_ALONE_NODES**

Run: `grep -n 'TERMINAL_NODES\|ALWAYS_ALONE_NODES' scripts/verify_mermaid.py | head -20`
Expected: Shows line numbers where these sets are defined

- [ ] **Step 2: Replace TERMINAL_NODES hardcoded set with suffix-based regex**

Find the `TERMINAL_NODES = {` block and replace it with:
```python
# Suffix-based terminal node detection — any node whose ID ends with these suffixes
TERMINAL_NODE_SUFFIXES = frozenset({
    '_proj', '_out', 'lm_head', 'final_norm', 'softmax', '_head', 'o_proj',
    'tokens', 'output', 'output_stage', 'input_stage', 'embed', 'head',
    'add1', 'add2', 'ln1', 'ln2', 'final', 'moe_out', 'attn_out',
    'down', 'ex', 'sw', 'o', 'proj',
})
# Case-insensitive variants
TERMINAL_NODE_SUFFIXES_LOWER = frozenset(s.lower() for s in TERMINAL_NODE_SUFFIXES)
```

- [ ] **Step 3: Replace ALWAYS_ALONE_NODES hardcoded set with regex rules**

Find the `ALWAYS_ALONE_NODES = {` block and replace it with:
```python
# Regex-based "always alone" node detection — subgraphs and detail expansion targets
ALWAYS_ALONE_NODES_PATTERNS = re.compile(
    r'.*_[Dd]etail|'       # *_Detail / *_detail subgraph containers
    r'Hybrid_|'             # Hybrid_* subgraph containers
    r'DSA_|'               # DSA_* detail expansion nodes
    r'Linear_|'            # Linear_* detail expansion nodes
    r'Gated_'              # Gated_* detail expansion nodes
    r'^(?:Title|ExpertX|\.\.\.|FFN_Detail|GQA_Detail|MoE_Pool|Attention_Detail|'
    r'MoE_Detail|MLA_Detail|Expert_Pool|Vision_Encoder|MM_Projector|'
    r'Input_Stage|Output_Stage|Transformer_Layer|'
    r'MoE_Module|Attn_Module|Router|Shared|Sliding_Window|Shared_Expert|'
    r'ConcatQ|ConcatK|Q_A|Q_B|Q_LN|KV_A|KV_LN|K_B|RoPE|Softmax|'
    r'Q_Split|K_Split|V_Split)$|'  # Literal names
    r'^(?:title|expertx|\.\.\.|ffn_detail|gqa_detail|moe_pool|attention_detail|'
    r'mla_detail|expert_pool|vision_encoder|mm_projector|'
    r'input_stage|output_stage|transformer_layer|'
    r'moe_module|attn_module|router|shared|sliding_window|shared_expert|'
    r'concatq|concatk|q_a|q_b|q_ln|kv_a|kv_ln|k_b|rope|softmax|'
    r'q_split|k_split|v_split)$'
)
```

- [ ] **Step 4: Update check_connectivity function to use regex/suffix checks**

Find `check_connectivity` function. Replace all `node in TERMINAL_NODES` checks with:
```python
_is_terminal = lambda n: n in TERMINAL_NODES or n.lower() in TERMINAL_NODE_SUFFIXES_LOWER
_is_always_alone = lambda n: n in ALWAYS_ALONE_NODES or ALWAYS_ALONE_NODES_PATTERNS.match(n)
```

Then replace each `if node in TERMINAL_NODES` and `if node in ALWAYS_ALONE_NODES` with calls to these lambdas.

- [ ] **Step 5: Run existing test to verify no regressions**

Run: `python scripts/verify_mermaid.py deepseek_v3_arch.mmd --verbose 2>&1 | head -30`
Expected: OK or same output as before

- [ ] **Step 6: Commit**

```bash
git add scripts/verify_mermaid.py
git commit -m "refactor(verify_mermaid): replace hardcoded whitelist with regex rules for extensibility"
```

---

## Task 2: Update SKILL.md — mermaid generation logic for attention types

**Files:**
- Modify: `SKILL.md` (find "Expected false positives" section and attention expansion section)

- [ ] **Step 1: Locate sections in SKILL.md**

Run: `grep -n 'Expected false positives\|Attention Expansion\|Attention Type' SKILL.md | head -20`
Expected: Shows line numbers of relevant sections

- [ ] **Step 2: Update "Expected false positives" section**

Find the paragraph starting with `**Expected false positives (ignore):**` and replace it with:
```
**Expected false positives (ignore):**
- `ORPHAN (no outgoing)` on nodes inside subgraphs — subgraph-internal nodes may appear orphaned because edges within subgraphs don't propagate through subgraph container IDs in the checker
- `DEAD END` on expansion target subgraphs (any node matching `*_Detail`, `Hybrid_*`, `DSA_*`, `Linear_*`, `Gated_*`) — `==>` expand arrows are visual-only and don't count as outgoing edges
- `ORPHAN` / `ISOLATED` on subgraph container IDs (`Input_Stage`, `Transformer_Layer`, `Output_Stage`) — these are transparent containers; use subgraph-level connections (`Input_Stage --> Transformer_Layer`) for data flow
```

- [ ] **Step 3: Add attention type branching logic before "Attention Expansion" section**

Find the section heading `### Attention Expansion (for MLA/Standard Attention)` and prepend before it a new section:

```
### Attention Type Expansion Rules

In Step 4 mermaid generation, after detecting attention type from template, use this branching logic:

**Single attention types:**
- `attention_type: standard` or `attention_type: mqa` or `attention_type: gqa` → expand GQA chain (Q_proj → K_proj → V_proj → Softmax → O_proj)
- `attention_type: mla` → expand MLA chain (Q_A → Q_LN → Q_B → ConcatQ; KV_A → KV_LN → ConcatK/K_B → RoPE → Softmax → O_Proj)
- `attention_type: dsa` → expand DSA chain (MLA framework + Indexer top-K → Softmax → O_Proj)
- `attention_type: swa` → expand SWA chain (Q_proj → K_proj → V_proj → SWA_Mask → Softmax → O_proj)
- `attention_type: linear` → expand Linear attention chain
- `attention_type: gated_deltanet` → expand Gated DeltaNet chain
- `attention_type: gated_attention` → expand Gated Attention chain

**Hybrid attention types:**
- `attention_hybrid_types` + `attention_hybrid_mode: parallel` → expand Parallel structure (same-layer multiple attention, outputs merged via `attention_merge_type`: add/gate/concat). Read `attention_merge_type` field to determine merge strategy: `add` (element-wise add), `gate` (element-wise multiply with learnable gate), `concat` (concatenate then project).
- `attention_hybrid_types` + `attention_hybrid_mode: alternating` → expand Alternating stack structure using `layer_types_key` + `hybrid_ratio`
- `attention_hybrid_types` + `attention_hybrid_mode: chain` → expand Chain structure (output of first attention feeds into second attention)

**Fallback:** If only `attention_impl` is present (deprecated), use it as `attention_type` directly (backward compatible).
```

- [ ] **Step 4: Commit**

```bash
git add SKILL.md
git commit -m "feat(SKILL.md): add attention_hybrid_types and all single attention type branches"
```

---

## Task 3: Update DeepSeek family templates (7 YAMLs)

**Files:**
- Modify: `templates/deepseek/common.yaml`
- Modify: `templates/deepseek/deepseek-v3.yaml`
- Modify: `templates/deepseek/deepseek-v3.1.yaml`
- Modify: `templates/deepseek/deepseek-v3.2.yaml` — **critical: DSA not MLA**
- Modify: `templates/deepseek/deepseek-v2.yaml`
- Modify: `templates/deepseek/deepseek-v2.5.yaml`
- Modify: `templates/deepseek/deepseek-r1.yaml`

- [ ] **Step 1: Update templates/deepseek/common.yaml — add MLA defaults**

Read: `templates/deepseek/common.yaml`
Add after the existing content:
```yaml
# Attention defaults for DeepSeek family
attention_type: mla
mla:
  kv_lora_rank_key: kv_lora_rank
  q_lora_rank_key: q_lora_rank
  qk_rope_head_dim_key: qk_rope_head_dim
  qk_nope_head_dim_key: qk_nope_head_dim
  v_head_dim_key: v_head_dim
```

- [ ] **Step 2: Update deepseek-v3.2.yaml — DSA attention**

Read: `templates/deepseek/deepseek-v3.2.yaml`
Add after config block (before `template:`):
```yaml
# DSA = MLA framework + indexer for top-K sparse KV selection
attention_type: dsa
dsa:
  sparse_topk_key: sparse_topk
  block_size_key: block_size
```

- [ ] **Step 3: Update remaining deepseek YAMLs (v3, v3.1, v2, v2.5, r1) — add attention_type: mla**

For each file, add `attention_type: mla` before the `template:` line.

- [ ] **Step 4: Commit**

```bash
git add templates/deepseek/common.yaml templates/deepseek/deepseek-v3.yaml templates/deepseek/deepseek-v3.1.yaml templates/deepseek/deepseek-v3.2.yaml templates/deepseek/deepseek-v2.yaml templates/deepseek/deepseek-v2.5.yaml templates/deepseek/deepseek-r1.yaml
git commit -m "feat(templates): add explicit attention_type for DeepSeek family"
```

---

## Task 4: Update GLM family templates (5 YAMLs)

**Files:**
- Modify: `templates/glm/common.yaml`
- Modify: `templates/glm/glm-4-9b.yaml`
- Modify: `templates/glm/glm-4-28b.yaml`
- Modify: `templates/glm/glm-4v.yaml`
- Create: `templates/glm/glm-5.yaml` — new file for DSA

- [ ] **Step 1: Update glm/common.yaml — add GQA defaults**

Read: `templates/glm/common.yaml`
Add after existing content:
```yaml
# Attention defaults for GLM family
attention_type: gqa
kv_heads_key: num_key_value_heads
```

- [ ] **Step 2: Update glm-4-9b.yaml, glm-4-28b.yaml, glm-4v.yaml — add attention_type: gqa**

- [ ] **Step 3: Create glm-5.yaml — DSA attention type**

Create new file with complete config (search HuggingFace for actual GLM-5 config values if not already known):
```yaml
model_name: GLM-5
family: glm
model_type: glm_5
source: THUDM/glm-5

config:
  hidden_size: 4096
  num_hidden_layers: 40
  num_attention_heads: 128
  num_key_value_heads: 2  # GQA base (DSA is MLA variant)
  vocab_size: 151552
  intermediate_size: 13696
  head_dim: 128
  # GLM-5 specific (from HuggingFace or technical report)
  kv_lora_rank: 512
  q_lora_rank: 1536
  qk_rope_head_dim: 64
  qk_nope_head_dim: 128
  v_head_dim: 128
  sparse_topk: 32
  block_size: 64

# DSA = MLA framework + indexer for top-K sparse KV selection
attention_type: dsa
dsa:
  sparse_topk_key: sparse_topk
  block_size_key: block_size

mla:
  kv_lora_rank_key: kv_lora_rank
  q_lora_rank_key: q_lora_rank
  qk_rope_head_dim_key: qk_rope_head_dim
  qk_nope_head_dim_key: qk_nope_head_dim
  v_head_dim_key: v_head_dim

template: glm/common.yaml
```

- [ ] **Step 4: Commit**

```bash
git add templates/glm/common.yaml templates/glm/glm-4-9b.yaml templates/glm/glm-4-28b.yaml templates/glm/glm-4v.yaml templates/glm/glm-5.yaml
git commit -m "feat(templates): add explicit attention_type for GLM family (glm-5 uses DSA)"
```

---

## Task 5: Update Baichuan family templates (4 YAMLs)

**Files:**
- Modify: `templates/baichuan/common.yaml`
- Modify: `templates/baichuan/baichuan-13b.yaml` — standard (MHA)
- Modify: `templates/baichuan/baichuan2-7b.yaml` — standard (MHA)
- Modify: `templates/baichuan/baichuan-m3.yaml` — gqa

- [ ] **Step 1: Update baichuan/common.yaml — add attention type comments**

Add at end:
```yaml
# Attention type defaults for Baichuan family:
# - Baichuan-13B: standard (MHA)
# - Baichuan2-7B: standard (MHA)
# - Baichuan-M3: gqa
# Each model file must declare its own attention_type explicitly.
```

- [ ] **Step 2: Update baichuan-13b.yaml — attention_type: standard**

Add `attention_type: standard` before `template:` line.

- [ ] **Step 3: Update baichuan2-7b.yaml — attention_type: standard**

- [ ] **Step 4: Update baichuan-m3.yaml — attention_type: gqa**

- [ ] **Step 5: Commit**

```bash
git add templates/baichuan/common.yaml templates/baichuan/baichuan-13b.yaml templates/baichuan/baichuan2-7b.yaml templates/baichuan/baichuan-m3.yaml
git commit -m "feat(templates): add explicit attention_type for Baichuan family"
```

---

## Task 6: Update Llama family templates (8 YAMLs)

**Files:**
- Modify: `templates/llama/common.yaml` (update comments)
- Modify: `templates/llama/llama-2-7b.yaml` — standard (MHA)
- Modify: `templates/llama/llama-3-8b.yaml` — gqa
- Modify: `templates/llama/llama-3.1-8b.yaml` — gqa
- Modify: `templates/llama/llama-3.2-1b.yaml` — gqa
- Modify: `templates/llama/llama-3.2-3b.yaml` — gqa
- Modify: `templates/llama/llama-3.2-11b.yaml` — gqa
- Modify: `templates/llama/llama-3.3-70b.yaml` — gqa

- [ ] **Step 1: Update llama-2-7b.yaml — attention_type: standard**

- [ ] **Step 2: Update llama-3.1-8b.yaml, llama-3.2-*, llama-3.3-70b.yaml, llama-3-8b.yaml — attention_type: gqa**

- [ ] **Step 3: Commit**

```bash
git add templates/llama/llama-2-7b.yaml templates/llama/llama-3-8b.yaml templates/llama/llama-3.1-8b.yaml templates/llama/llama-3.2-1b.yaml templates/llama/llama-3.2-3b.yaml templates/llama/llama-3.2-11b.yaml templates/llama/llama-3.3-70b.yaml
git commit -m "feat(templates): add explicit attention_type for Llama family"
```

---

## Task 7: Update Mistral family templates (5 YAMLs)

**Files:**
- Modify: `templates/mistral/common.yaml` — add swa defaults
- Modify: `templates/mistral/mistral-7b-v0.1.yaml` — swa
- Modify: `templates/mistral/mistral-7b-v0.3.yaml` — swa
- Modify: `templates/mistral/mistral-nemo.yaml` — swa
- Modify: `templates/mistral/mistral-large.yaml` — swa

- [ ] **Step 1: Update mistral/common.yaml — add SWA defaults**

Add:
```yaml
attention_type: swa
sliding_window_key: sliding_window
```

- [ ] **Step 2: Add attention_type: swa to all mistral model files**

- [ ] **Step 3: Commit**

```bash
git add templates/mistral/common.yaml templates/mistral/mistral-7b-v0.1.yaml templates/mistral/mistral-7b-v0.3.yaml templates/mistral/mistral-nemo.yaml templates/mistral/mistral-large.yaml
git commit -m "feat(templates): add explicit attention_type for Mistral family (all SWA)"
```

---

## Task 8: Update Kimi, MiniMax, MiMo, Qwen family templates

**Kimi (3 YAMLs):**
- Modify: `templates/kimi/common.yaml` — add mla defaults
- Modify: `templates/kimi/kimi-k2.0.yaml` — mla
- Modify: `templates/kimi/kimi-k2.5.yaml` — mla

**MiniMax (3 YAMLs):**
- Modify: `templates/minimax/common.yaml` — add gqa defaults
- Modify: `templates/minimax/minimax-m2.yaml` — gqa
- Modify: `templates/minimax/minimax-m2.5.yaml` — gqa

**MiMo (3 YAMLs):**
- Modify: `templates/mimo/common.yaml` — add swa defaults
- Modify: `templates/mimo/mimo-v2-flash.yaml` — swa + hybrid_ratio
- Modify: `templates/mimo/mimo-vl-7b.yaml` — swa

**Qwen (12 YAMLs):**
- Modify: `templates/qwen/common.yaml` — add gqa defaults
- Modify: all 11 qwen model files — gqa

- [ ] **Step 1: Update all Kimi YAMLs**

```bash
git add templates/kimi/common.yaml templates/kimi/kimi-k2.0.yaml templates/kimi/kimi-k2.5.yaml
git commit -m "feat(templates): add explicit attention_type for Kimi family"
```

- [ ] **Step 2: Update all MiniMax YAMLs**

```bash
git add templates/minimax/common.yaml templates/minimax/minimax-m2.yaml templates/minimax/minimax-m2.5.yaml
git commit -m "feat(templates): add explicit attention_type for MiniMax family"
```

- [ ] **Step 3: Update MiMo YAMLs (mimo-v2-flash needs hybrid_ratio)**

For `mimo-v2-flash.yaml`, add:
```yaml
attention_type: swa
hybrid_ratio: "5:1"  # SWA:Full layer alternating
```

```bash
git add templates/mimo/common.yaml templates/mimo/mimo-v2-flash.yaml templates/mimo/mimo-vl-7b.yaml
git commit -m "feat(templates): add explicit attention_type for MiMo family"
```

- [ ] **Step 4: Update all Qwen YAMLs**

```bash
git add templates/qwen/common.yaml templates/qwen/qwen3-32b.yaml templates/qwen/qwen3-8b.yaml templates/qwen/qwen3-4b.yaml templates/qwen/qwen3-1.7b.yaml templates/qwen/qwen3-0.6b.yaml templates/qwen/qwen3-vl-8b.yaml templates/qwen/qwen3.5-397b-a17b.yaml templates/qwen/qwen3.5-122b-a10b.yaml templates/qwen/qwen3.5-35b-a3b.yaml templates/qwen/qwen3.5-27b.yaml templates/qwen/qwen2.5-vl-7b.yaml
git commit -m "feat(templates): add explicit attention_type for Qwen family"
```

---

## Task 9: Update GPT-OSS family templates (3 YAMLs) — Hybrid Alternating

**Files:**
- Modify: `templates/gpt-oss/common.yaml` — add swa defaults
- Modify: `templates/gpt-oss/gpt-oss-120b.yaml` — hybrid alternating
- Modify: `templates/gpt-oss/gpt-oss-20b.yaml` — hybrid alternating

- [ ] **Step 1: Update gpt-oss/common.yaml — add swa defaults**

Add:
```yaml
attention_type: swa
```

- [ ] **Step 2: Update gpt-oss-120b.yaml — hybrid alternating**

Read the current file. Before `template:` line, add:
```yaml
# Hybrid Alternating: SWA layers + Full attention layers alternate
attention_hybrid_types:
  - swa
  - full
attention_hybrid_mode: alternating
layer_types_key: layer_types
hybrid_ratio: "1:1"  # 1 SWA layer : 1 Full layer alternating
```

- [ ] **Step 3: Update gpt-oss-20b.yaml — same as 120b**

- [ ] **Step 4: Commit**

```bash
git add templates/gpt-oss/common.yaml templates/gpt-oss/gpt-oss-120b.yaml templates/gpt-oss/gpt-oss-20b.yaml
git commit -m "feat(templates): add hybrid alternating attention for GPT-OSS family"
```

---

## Task 10: Verify mermaid diagrams

**Files to verify:**
- `gpt_oss_120b_arch.mmd` — verify hybrid alternating structure
- `gpt_oss_20b_arch.mmd` — verify hybrid alternating structure

- [ ] **Step 1: Run verify_mermaid.py on existing GPT-OSS diagrams**

Run: `python scripts/verify_mermaid.py gpt_oss_120b_arch.mmd --verbose`
Expected: OK (no UNDEFINED/ORPHAN/DEAD PATH errors)

- [ ] **Step 2: Verify gpt_oss_20b_arch.mmd**

Run: `python scripts/verify_mermaid.py gpt_oss_20b_arch.mmd --verbose`
Expected: OK

- [ ] **Step 3: Commit any fixes needed**

---

## Task 11: Generate DeepSeek V3.2 mermaid diagram

**Files:**
- Create: `deepseek_v3_2_arch.mmd` (new file; `deepseek_v3_arch.mmd` is for V3, not V3.2)

- [ ] **Step 1: Use SKILL.md to generate deepseek-v3.2 architecture diagram**

Following SKILL.md workflow: analyze model → generate mermaid → verify

- [ ] **Step 2: Run verify_mermaid.py on new diagram**

Run: `python scripts/verify_mermaid.py deepseek_v3_2_arch.mmd --verbose`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add deepseek_v3_2_arch.mmd
git commit -m "feat: generate DeepSeek-V3.2 architecture diagram (DSA attention)"
```

---

## Task 12: Final integration test

- [ ] **Step 1: Pick one model per attention type and verify templates generate correctly**

Pick: DeepSeek-V3.2 (dsa), GPT-OSS-120B (hybrid alternating), Mistral-7B (swa), Llama-3.1-8B (gqa)

For each, run the SKILL.md workflow to generate and verify mermaid:
```bash
# Example for DeepSeek-V3.2
python scripts/verify_mermaid.py deepseek_v3_2_arch.mmd --verbose
# Should pass with OK
```

- [ ] **Step 2: Commit final state**
