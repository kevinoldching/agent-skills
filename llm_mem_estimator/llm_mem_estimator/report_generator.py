#!/usr/bin/env python3
"""
Report generator for LLM Memory Estimator
"""

from typing import Dict, Optional, Any

from .model_config import ModelConfig, MemoryResult, get_dtype_bytes, calculate_weight_memory


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
            lines.append("**计算公式说明：**")
            lines.append("")
            if 'kv_cache' in computation_rules:
                kv_formula = computation_rules['kv_cache']
                lines.append(f"- KV Cache: `{kv_formula}`")
            if 'activation' in computation_rules:
                act_formula = computation_rules['activation']
                lines.append(f"- Activation: `{act_formula}`")
            lines.append("")

        # Weights Breakdown - combined table with all details
        if result.breakdown and config.modules:
            lines.append("## Weights Breakdown by Module")
            lines.append("")
            lines.append("| Module Type | Weight Name | Shape | Layers | Memory (GB) | Percentage | Data Type | Parallel Strategy | World Size |")
            lines.append("|-------------|-------------|-------|--------|-------------|------------|-----------|-------------------|------------|")

            # Define module order
            module_order = ['embedding', 'attention', 'ffn_moe', 'ffn_shared_expert', 'ffn_dense', 'norm', 'others']

            # Print rows grouped by module type (in defined order)
            for module_type in module_order:
                if module_type not in config.modules:
                    continue

                module_weights = config.modules[module_type]
                if not module_weights:
                    continue

                for weight_name, weight_info in module_weights.items():
                    # Calculate memory with parallel strategy
                    tp = parallel_config.get('tp', 1)
                    pp = parallel_config.get('pp', 1)
                    dp = parallel_config.get('dp', 1)
                    cp = parallel_config.get('cp', 1)
                    ep = parallel_config.get('ep', 1)
                    weight_memory = calculate_weight_memory(weight_info, tp, pp, dp, cp, ep)

                    # Calculate percentage relative to total weights memory
                    pct = weight_memory / result.weights_memory_gb * 100 if result.weights_memory_gb > 0 else 0

                    # Format shape
                    shape_str = str(weight_info.shape).replace(" ", "")

                    # Format parallel strategy
                    parallel_strategy = weight_info.parallel_strategy or "N/A"
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

                    lines.append(f"| {module_type} | {weight_name} | {shape_str} | {weight_info.layers} | {weight_memory:.5f} | {pct:.2f}% | {weight_info.dtype} | {parallel_strategy} | {world_size} |")

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
