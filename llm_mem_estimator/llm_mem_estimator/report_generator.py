#!/usr/bin/env python3
"""
Report generator for LLM Memory Estimator
"""

import re
from typing import Dict, Optional, Any

from .model_config import ModelConfig, MemoryResult, get_dtype_bytes, calculate_weight_memory


def simplify_weight_name(weight_name: str) -> str:
    """Simplify weight name by replacing numeric indices with N

    Replaces patterns like:
    - .layers.0. -> .layers.N.
    - .blocks.0. -> .blocks.N.
    - .h.0. -> .h.N.
    - .transformer.blocks.0. -> .transformer.blocks.N.
    """
    # Replace .word.digit. with .word.N.
    # Pattern: dot followed by word characters, then dot, then digits, then dot
    simplified = re.sub(r'(\.\w+)\.\d+(\.)', r'\1.N\2', weight_name)
    return simplified


class ReportGenerator:
    """Generate markdown reports"""

    @staticmethod
    def generate_report(config: ModelConfig, result: MemoryResult,
                       batch_size: int, parallel_config: Dict[str, int],
                       prompt_len: int = 0, gen_len: int = 0,
                       chip_info: Optional[Dict[str, Any]] = None) -> str:
        """Generate a markdown report"""
        lines = []

        # Title
        lines.append(f"# {config.model_identity.name} - Memory Estimation Report")
        lines.append("")

        # Model Information
        lines.append("## Model Information")
        lines.append("")
        lines.append(f"- **Model Name**: {config.model_identity.name}")
        lines.append(f"- **Total Parameters**: {config.model_identity.total_params}")
        lines.append(f"- **Number of Layers**: {config.model_identity.num_layers}")
        if config.model_identity.quantization:
            lines.append(f"- **Quantization**: {config.model_identity.quantization}")
        lines.append(f"- **Attention Type**: {config.architecture_config.attention_type}")
        lines.append(f"- **FFN Type**: {config.architecture_config.ffn_type}")
        lines.append(f"- **Normalization Type**: {config.architecture_config.norm_type}")
        lines.append("")

        # Configuration
        lines.append("## Configuration")
        lines.append("")
        lines.append(f"- **Batch Size**: {batch_size}")
        if prompt_len > 0:
            lines.append(f"- **Prompt Length**: {prompt_len}")
        if gen_len > 0:
            lines.append(f"- **Generated Length**: {gen_len}")
        lines.append(f"- **Tensor Parallel (TP)**: {parallel_config.get('tp', 1)}")
        lines.append(f"- **Pipeline Parallel (PP)**: {parallel_config.get('pp', 1)}")
        lines.append(f"- **Data Parallel (DP)**: {parallel_config.get('dp', 1)}")
        lines.append(f"- **Context Parallel (CP)**: {parallel_config.get('cp', 1)}")
        lines.append(f"- **Expert Parallel (EP)**: {parallel_config.get('ep', 1)}")
        lines.append("")

        # Hardware Information
        if chip_info:
            lines.append("## Hardware Information")
            lines.append("")
            lines.append(f"- **Chip**: {chip_info.get('name', 'Unknown')}")
            lines.append(f"- **VRAM**: {chip_info.get('vram_gb', 'Unknown')} GB")
            lines.append(f"- **Bandwidth**: {chip_info.get('bandwidth_gb_s', 'Unknown')} GB/s")
            lines.append("")

        # Memory Usage Summary
        lines.append("## Memory Usage Summary")
        lines.append("")
        lines.append("| Component | Memory (GB) | Percentage |")
        lines.append("|-----------|-------------|------------|")

        total = result.total_memory_gb
        lines.append(f"| Weights | {result.weights_memory_gb:.2f} | {result.weights_memory_gb/total*100:.1f}% |")
        lines.append(f"| KV Cache | {result.kv_cache_memory_gb:.2f} | {result.kv_cache_memory_gb/total*100:.1f}% |")
        lines.append(f"| Activation | {result.activation_memory_gb:.2f} | {result.activation_memory_gb/total*100:.1f}% |")
        lines.append(f"| System Reserved | {result.system_reserved_gb:.2f} | {result.system_reserved_gb/total*100:.1f}% |")
        lines.append(f"| **Total** | **{total:.2f}** | **100.0%** |")
        lines.append("")

        # Add computation rules
        computation_rules = config.computation_rules
        if computation_rules:
            lines.append("**Computation Formulas:**")
            lines.append("")
            if 'kv_cache' in computation_rules:
                kv_formula = computation_rules['kv_cache']
                lines.append(f"- KV Cache: `{kv_formula}`")
            if 'activation' in computation_rules:
                act_formula = computation_rules['activation']
                lines.append(f"- Activation: `{act_formula}`")
            # Show recommended_capacity_factor values if present
            if 'recommended_capacity_factor' in computation_rules:
                rcf = computation_rules['recommended_capacity_factor']
                if isinstance(rcf, dict):
                    lines.append(f"- recommended_capacity_factor: has_prefill={rcf.get('has_prefill', 1.25)}, decode={rcf.get('decode', 12.5)}")
                else:
                    lines.append(f"- recommended_capacity_factor: {rcf}")
            lines.append("")

            # Add calculation example with actual values
            lines.append("**Calculation Example:**")
            lines.append("")
            lines.append("```")
            tp = parallel_config.get('tp', 1)
            cp = parallel_config.get('cp', 1)
            total_seq_len = prompt_len + gen_len

            arch = config.architecture_config
            hidden_size = arch.hidden_size
            num_experts_per_tok = arch.num_experts_per_tok or 1

            if 'kv_cache' in computation_rules:
                kv_formula = computation_rules['kv_cache']
                lines.append(f"# KV Cache formula: {kv_formula}")
                lines.append(f"# Substituting: batch_size={batch_size}, seq_len={total_seq_len}, tp_size={tp}, cp_size={cp}")
                # Show simplified calculation for gpt-oss style formulas
                if 'min' in kv_formula:
                    min_part = min(batch_size * total_seq_len, 128)
                    lines.append(f"# = (18 * {batch_size} * {total_seq_len} * 1024 + 18 * {min_part} * 1024) / ({tp} * {cp})")
                    lines.append(f"# = {result.kv_cache_memory_gb:.4f} GB")
                else:
                    lines.append(f"# = {result.kv_cache_memory_gb:.4f} GB")
                lines.append("")

            if 'activation' in computation_rules:
                act_formula = computation_rules['activation']
                # Determine seq_len for activation based on the actual result
                # If activation is very small, it's likely Decode (seq_len=1)
                # Otherwise it's Prefill (seq_len=total_seq_len)
                if result.activation_memory_gb < 0.001:
                    act_seq_len = 1
                    factor = 12.5
                else:
                    act_seq_len = total_seq_len
                    # Calculate factor from result
                    elements = batch_size * act_seq_len * hidden_size * num_experts_per_tok * 2 / cp
                    if elements > 0:
                        factor = result.activation_memory_gb * (1024**3) / elements
                    else:
                        factor = 1.25

                lines.append(f"# Activation formula: {act_formula}")
                lines.append(f"# Substituting: batch_size={batch_size}, seq_len={act_seq_len}, hidden_size={hidden_size}, num_experts_per_tok={num_experts_per_tok}, cp_size={cp}")
                lines.append(f"# = {batch_size} * {act_seq_len} * {hidden_size} * {num_experts_per_tok} * {factor:.2f} * 2 / {cp}")
                lines.append(f"# = {result.activation_memory_gb:.6f} GB")
            lines.append("```")
            lines.append("")

        # Weights Breakdown - combined table with all details
        if result.breakdown and config.modules:
            lines.append("## Weights Breakdown by Module")
            lines.append("")

            # Calculate memory by module type
            module_order = ['embedding', 'attention', 'ffn_moe', 'ffn_shared_expert', 'ffn_dense', 'norm', 'others']
            module_totals = {}

            for module_type in module_order:
                if module_type not in config.modules:
                    continue
                module_weights = config.modules[module_type]
                if not module_weights:
                    continue

                tp = parallel_config.get('tp', 1)
                pp = parallel_config.get('pp', 1)
                dp = parallel_config.get('dp', 1)
                cp = parallel_config.get('cp', 1)
                ep = parallel_config.get('ep', 1)

                module_memory = 0.0
                for weight_name, weight_info in module_weights.items():
                    weight_memory = calculate_weight_memory(weight_info, tp, pp, dp, cp, ep)
                    module_memory += weight_memory

                if module_memory > 0:
                    module_totals[module_type] = module_memory

            # Print module type summary
            if module_totals:
                lines.append("**By Module Type:**")
                lines.append("")
                lines.append("| Module Type | Memory (GB) | Percentage |")
                lines.append("|-------------|-------------|------------|")
                for module_type in module_order:
                    if module_type in module_totals:
                        mem = module_totals[module_type]
                        pct = mem / result.weights_memory_gb * 100 if result.weights_memory_gb > 0 else 0
                        lines.append(f"| {module_type} | {mem:.2f} | {pct:.1f}% |")
                lines.append("")

            # Detailed breakdown table
            lines.append("**Detailed Breakdown:**")
            lines.append("")
            lines.append("| Module Type | Weight Name | Shape | Layers | Memory (GB) | Percentage | Data Type | Parallel Strategy | World Size |")
            lines.append("|-------------|-------------|-------|--------|-------------|------------|-----------|-------------------|------------|")

            # Define module order
            module_order = ['embedding', 'attention', 'ffn_moe', 'ffn_shared_expert', 'ffn_dense', 'norm', 'others']

            # Get parallel config
            tp = parallel_config.get('tp', 1)
            pp = parallel_config.get('pp', 1)
            dp = parallel_config.get('dp', 1)
            cp = parallel_config.get('cp', 1)
            ep = parallel_config.get('ep', 1)

            # Collect and merge weights by simplified name, dtype, parallel_strategy
            # Key: (module_type, simplified_name, dtype, parallel_strategy)
            # Value: {(shape_tuple): {'count': N, 'memory': total_memory}}
            merged_weights = {}

            for module_type in module_order:
                if module_type not in config.modules:
                    continue

                module_weights = config.modules[module_type]
                if not module_weights:
                    continue

                for weight_name, weight_info in module_weights.items():
                    # Simplify weight name for display (e.g., blocks.0. -> blocks.N.)
                    display_name = simplify_weight_name(weight_name)
                    parallel_strategy = weight_info.parallel_strategy or "N/A"
                    dtype = weight_info.dtype

                    # Calculate memory with parallel strategy
                    weight_memory = calculate_weight_memory(weight_info, tp, pp, dp, cp, ep)

                    # Create key for merging (without shape)
                    key = (module_type, display_name, dtype, parallel_strategy)
                    shape_tuple = tuple(weight_info.shape)

                    if key not in merged_weights:
                        merged_weights[key] = {}

                    if shape_tuple not in merged_weights[key]:
                        merged_weights[key][shape_tuple] = {
                            'count': 0,
                            'memory': 0.0,
                        }

                    merged_weights[key][shape_tuple]['count'] += 1
                    merged_weights[key][shape_tuple]['memory'] += weight_memory

            # Print merged rows
            for (module_type, display_name, dtype, parallel_strategy), shape_data in merged_weights.items():
                for shape_tuple, data in shape_data.items():
                    count = data['count']
                    weight_memory = data['memory']

                    # Insert count into shape if count > 1 (like MoE expert count)
                    if count > 1:
                        display_shape = [count] + list(shape_tuple)
                    else:
                        display_shape = list(shape_tuple)
                    shape_str = str(display_shape).replace(" ", "")

                    # Calculate percentage relative to total weights memory
                    pct = weight_memory / result.weights_memory_gb * 100 if result.weights_memory_gb > 0 else 0

                    # Calculate actual world size based on parallel strategy and CLI params
                    if parallel_strategy.upper() == "TP":
                        world_size = tp
                    elif parallel_strategy.upper() == "EP":
                        world_size = ep
                    elif parallel_strategy.upper() == "PP":
                        world_size = pp
                    elif parallel_strategy.upper() == "DP":
                        world_size = dp
                    elif parallel_strategy.upper() == "CP":
                        world_size = cp
                    else:
                        world_size = 1  # replicated

                    lines.append(f"| {module_type} | {display_name} | {shape_str} | {count} | {weight_memory:.5f} | {pct:.2f}% | {dtype} | {parallel_strategy} | {world_size} |")

            # Print Total row
            lines.append(f"| **Total** | - | - | - | **{result.weights_memory_gb:.5f}** | **100.00%** | - | - | - |")

            lines.append("")

        # Maximum Capacity
        if result.max_sequence_length:
            lines.append("## Maximum Capacity")
            lines.append("")
            lines.append(f"- **Maximum Sequence Length**: {result.max_sequence_length:,}")
            lines.append("")

        if result.max_batch_size:
            lines.append(f"- **Maximum Batch Size**: {result.max_batch_size}")
            lines.append("")

        return "\n".join(lines)
