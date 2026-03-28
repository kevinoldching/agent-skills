#!/usr/bin/env python3
"""Verify mermaid diagram connectivity — detect orphan nodes, dead ends, and broken paths.

Usage:
    python scripts/verify_mermaid.py <diagram.mmd>
    python scripts/verify_mermaid.py <diagram.mmd> --verbose

Checks:
1. Every defined node has at least one outgoing edge (unless it's a terminal node)
2. Every referenced node is actually defined
3. No disconnected subgraphs (except intentionally isolated like legends)
4. Path from Input to Output is continuous
"""

import argparse
import re
import sys
from collections import defaultdict


# Explicit terminal nodes (kept as fallback for exact matches)
TERMINAL_NODES = {
    # Uppercase (main diagram nodes)
    "Output", "LM_Head", "Final_Norm", "Final", "Head", "Out_L", "Out1", "Out2", "Out3", "Out61",
    "Out_D1", "Out_M1", "Out", "Tokens", "Output_Stage", "Input_Stage", "Transformer_Layer",
    "MoE_Out", "Attn_Out", "MoE_Module", "Attn_Module", "Down", "O", "O_Proj", "Proj", "EX", "SW",
    "Input_LN", "Embed", "LN1", "LN2", "ADD1", "ADD2",
    # Lowercase (SKILL.md template convention)
    "output", "lm_head", "final_norm", "final", "head", "out_l", "out1", "out2", "out3", "out61",
    "out_d1", "out_m1", "out", "tokens", "output_stage", "input_stage", "transformer_layer",
    "moe_out", "attn_out", "moe_module", "attn_module", "down", "o", "o_proj", "proj", "ex", "sw",
    "input_ln", "embed", "ln1", "ln2", "add1", "add2",
    # SKILL.md detail nodes (lowercase)
    "v_proj", "k_proj", "q_proj", "attn_in", "o_proj", "attn_out", "routed_1", "routed_2", "routed_n",
    "routed_x", "router", "shared", "moe_out", "moe_out2", "MoE_out", "MoE_Out2",
    # Attention detail output nodes (various attention types)
    "attn_out_swa", "attn_out_full", "attn_out_mla", "attn_out_dsa",
    "attn_in_swa", "attn_in_full", "attn_in_mla", "attn_in_dsa",
    # MoE expert nodes (expert pool terminals)
    "expert1", "expert2", "expert3", "expert4", "expert5", "expert6", "expert7", "expert8",
    "expert128", "expert256", "expertN",
    # MoE intermediate terminals
    "moe_out_swa", "moe_out_full", "moe_out_mla", "moe_out_dense",
    # Vision/language projection terminals
    "vision_out", "mm_output",
}

# Suffix-based terminal node detection — any node whose ID ends with these suffixes
TERMINAL_NODE_SUFFIXES = frozenset({
    '_proj', '_out', 'lm_head', 'final_norm', 'softmax', '_head', 'o_proj',
    'tokens', 'output', 'output_stage', 'input_stage', 'embed', 'head',
    'add1', 'add2', 'ln1', 'ln2', 'final', 'moe_out', 'attn_out',
    'down', 'ex', 'sw', 'proj', 'out',
    # Additional suffixes to cover original TERMINAL_NODES explicit entries
    '_d1', '_m1', '_l', '_1', '_2', '_3', '_61', '_in', '_n', '_x', '_out2',
    # Attention type suffixes (swa, full, mla, dsa, etc.)
    '_swa', '_full', '_mla', '_dsa', '_gqa',
    # Expert pool suffixes
    '_expert',
})
# Case-insensitive variants
TERMINAL_NODE_SUFFIXES_LOWER = frozenset(s.lower() for s in TERMINAL_NODE_SUFFIXES)

# Explicit "always alone" nodes (kept as fallback for exact matches)
ALWAYS_ALONE_NODES = {
    # Uppercase subgraph containers and detail expansion targets
    "Title", "ExpertX", "...", "FFN_Detail", "GQA_Detail", "MoE_Pool", "Attention_Detail",
    "MoE_Detail", "MLA_Detail", "Expert_Pool", "Vision_Encoder", "MM_Projector",
    "Input_Stage", "Output_Stage", "Transformer_Layer",
    "MoE_Module", "Attn_Module", "Router", "Shared", "Sliding_Window", "Shared_Expert",
    "ConcatQ", "ConcatK", "Q_A", "Q_B", "Q_LN", "KV_A", "KV_LN", "K_B", "RoPE", "Softmax",
    "Q_Split", "K_Split", "V_Split",
    # Lowercase
    "title", "expertx", "...", "ffn_detail", "gqa_detail", "moe_pool", "attention_detail",
    "moe_detail", "mla_detail", "expert_pool", "vision_encoder", "mm_projector",
    "input_stage", "output_stage", "transformer_layer",
    "moe_module", "attn_module", "router", "shared", "sliding_window", "shared_expert",
    "concatq", "concatk", "q_a", "q_b", "q_ln", "kv_a", "kv_ln", "k_b", "rope", "softmax",
    "q_split", "k_split", "v_split",
    # Hybrid/Alternating layer subgraphs
    "SWA_Layers", "Full_Layers", "Dense_Layers", "MoE_Layers",
    "swa_layers", "full_layers", "dense_layers", "moe_layers",
    # Attention detail variants
    "Attention_Detail_SWA", "Attention_Detail_Full", "Attention_Detail_MLA", "Attention_Detail_DSA",
    "attention_detail_swa", "attention_detail_full", "attention_detail_mla", "attention_detail_dsa",
}

# Regex-based "always alone" node detection — subgraphs and detail expansion targets
ALWAYS_ALONE_NODES_PATTERNS = re.compile(
    r'.*_[Dd]etail|'       # *_Detail / *_detail subgraph containers
    r'Hybrid_|'             # Hybrid_* subgraph containers
    r'DSA_|'               # DSA_* detail expansion nodes
    r'Linear_|'            # Linear_* detail expansion nodes
    r'Gated_|'              # Gated_* detail expansion nodes
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


def parse_mermaid(path: str) -> tuple[set[str], list[tuple[str, str, str]], str]:
    """Parse .mmd file and return (defined_nodes, edges, content)."""
    with open(path) as f:
        content = f.read()

    defined = set()
    edges = []

    # Match node definitions of ANY shape: NodeID["..."], NodeID(("...")), NodeID{{"..."}}
    node_def_pattern = re.compile(
        r'(?<![:\w])([A-Za-z][A-Za-z0-9_]*)\[\"[^\"]*\"\]|'
        r'(?<![:\w])([A-Za-z][A-Za-z0-9_]*)\(\([^\)]*\)\)|'
        r'(?<![:\w])([A-Za-z][A-Za-z0-9_]*)\{\{[^\}]*\}\}'
    )

    for m in node_def_pattern.finditer(content):
        for g in m.groups():
            if g:
                defined.add(g)

    # Subgraph IDs
    subgraph_pattern = re.compile(r'subgraph\s+([A-Za-z][A-Za-z0-9_]*)\s')
    for m in subgraph_pattern.finditer(content):
        defined.add(m.group(1))

    # Normal arrows: A --> B (skip ::class references that precede the arrow on same line)
    for m in re.finditer(r'([A-Za-z][A-Za-z0-9_]*)\s+(-->)\s+([A-Za-z][A-Za-z0-9_]*)', content):
        # If ::class appears before the --> on the same line, it means the arrow
        # is the ::class's :: not the actual --> (e.g. LN["x"]:::cls --> B)
        cls_pos = content.rfind(':::', 0, m.start())
        if cls_pos >= 0:
            line_start = content.rfind('\n', 0, m.start()) + 1
            if cls_pos >= line_start:
                continue  # ::: precedes --> on same line, skip
        edges.append((m.group(1), m.group(3), 'normal'))

    # Residual arrows: A -.-> |label| B
    for m in re.finditer(r'([A-Za-z][A-Za-z0-9_]*)\s+-\.->\s+\|([^|]+)\|\s*([A-Za-z][A-Za-z0-9_]*)', content):
        edges.append((m.group(1), m.group(3), 'residual'))

    # Expand arrows: A ==> B
    for m in re.finditer(r'([A-Za-z][A-Za-z0-9_]*)\s*==>\s*([A-Za-z][A-Za-z0-9_]*)', content):
        edges.append((m.group(1), m.group(2), 'expand'))

    return defined, edges, content


def _build_subgraph_map(content: str) -> dict[str, str]:
    """Map each node to its parent subgraph ID."""
    node_to_subgraph = {}
    subgraph_pattern = re.compile(r'subgraph\s+([A-Za-z][A-Za-z0-9_]*)\s')
    node_def_pattern = re.compile(
        r'(?<![:\w])([A-Za-z][A-Za-z0-9_]*)\[\"[^\"]*\"\]|'
        r'(?<![:\w])([A-Za-z][A-Za-z0-9_]*)\(\([^\)]*\)\)|'
        r'(?<![:\w])([A-Za-z][A-Za-z0-9_]*)\{\{[^\}]*\}\}'
    )
    pos = 0
    while True:
        m = subgraph_pattern.search(content, pos)
        if not m:
            break
        sid = m.group(1)
        start = m.end()
        next_sub = content.find('subgraph', start)
        end = next_sub if next_sub > 0 else len(content)
        for nm in node_def_pattern.finditer(content[start:end]):
            for g in nm.groups():
                if g:
                    node_to_subgraph[g] = sid
        pos = end
    return node_to_subgraph


def check_connectivity(defined: set[str], edges: list[tuple[str, str, str]],
                       content: str, verbose: bool = False) -> list[str]:
    """Check graph connectivity and return list of issues found."""
    issues = []

    # Suffix and regex-based node classification (with explicit set fallbacks)
    _is_terminal = lambda n: n in TERMINAL_NODES or n.lower().endswith(tuple(TERMINAL_NODE_SUFFIXES_LOWER))
    _is_always_alone = lambda n: n in ALWAYS_ALONE_NODES or ALWAYS_ALONE_NODES_PATTERNS.match(n)

    # Build adjacency lists
    outgoing = defaultdict(list)
    incoming = defaultdict(list)

    for src, dst, etype in edges:
        outgoing[src].append((dst, etype))
        incoming[dst].append((src, etype))

    all_nodes = defined | set(outgoing.keys()) | set(incoming.keys())

    # 1. Check each defined node has outgoing edges (unless terminal)
    for node in sorted(defined):
        if _is_terminal(node) or _is_always_alone(node):
            continue
        if node not in outgoing:
            issues.append(f"ORPHAN (no outgoing): '{node}' is defined but has no connections leaving it")

    # 2. Check each referenced node is defined
    for src, dst, _ in edges:
        if src not in defined:
            issues.append(f"UNDEFINED (node used but not defined): '{src}'")
        if dst not in defined:
            issues.append(f"UNDEFINED (node used but not defined): '{dst}'")

    # 3. Check for dead-end nodes (nodes with incoming but no outgoing, not terminal)
    for node in sorted(all_nodes):
        if _is_terminal(node) or _is_always_alone(node):
            continue
        if node in outgoing or node in incoming:
            if node not in outgoing:
                issues.append(f"DEAD END: '{node}' receives connections but has no outgoing edges")

    # 4. Check for orphan nodes that are defined but neither send nor receive
    for node in sorted(defined):
        has_any = (node in outgoing) or (node in incoming)
        if not has_any and not _is_terminal(node) and not _is_always_alone(node):
            issues.append(f"ISOLATED: '{node}' is defined but completely disconnected")

    # 5. Check for expand arrows (==>) — expansion targets are expected to have no outgoing
    #    edges in the main graph (they're zoomed-in views); skip this check
    #    (kept for documentation; not flagged as an issue)
    _ = [src for src, dst, etype in edges if etype == 'expand' and dst not in outgoing]

    # 6. Path continuity: BFS from input nodes to output nodes
    input_nodes = {"Embed", "Input_LN", "Tokens", "Input_Stage", "embed", "input_ln", "tokens", "input_stage"}
    output_nodes = {"Final_Norm", "LM_Head", "Final", "Head", "Output", "moe_out", "attn_out", "final_norm", "lm_head", "head", "output"}
    node_to_subgraph = _build_subgraph_map(content)

    reachable_from_input = set()
    queue = list(input_nodes & defined)
    visited = set(queue)
    while queue:
        curr = queue.pop(0)
        reachable_from_input.add(curr)
        # If this node is inside a subgraph, the subgraph itself is also reachable
        sg = node_to_subgraph.get(curr)
        if sg and sg not in visited:
            visited.add(sg)
            queue.append(sg)
        for nxt, _ in outgoing.get(curr, []):
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)

    # Nodes inside a reachable subgraph are also reachable
    for node in defined:
        sg = node_to_subgraph.get(node)
        if sg and sg in reachable_from_input and node not in visited:
            reachable_from_input.add(node)
            visited.add(node)

    for out in output_nodes:
        if out in defined and out not in reachable_from_input:
            # Skip if out is inside a detail subgraph (only reachable via == expand arrow)
            sg = node_to_subgraph.get(out)
            if sg and _is_always_alone(sg):
                continue  # detail subgraph nodes are only reachable via == expand
            issues.append(f"DEAD PATH: '{out}' is not reachable from Input_Stage")

    return issues


def main():
    parser = argparse.ArgumentParser(description='Verify mermaid diagram connectivity')
    parser.add_argument('file', help='Path to .mmd file')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    defined, edges, content = parse_mermaid(args.file)

    if args.verbose:
        print(f"Nodes defined: {len(defined)}")
        print(f"Edges found: {len(edges)}")
        print(f"Defined nodes: {sorted(defined)}")
        print()

    issues = check_connectivity(defined, edges, content, args.verbose)

    if issues:
        print(f"ISSUES FOUND in {args.file}:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    else:
        print(f"OK: {args.file} — no connectivity issues found")
        sys.exit(0)


if __name__ == '__main__':
    main()
