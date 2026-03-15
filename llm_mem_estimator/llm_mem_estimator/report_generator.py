#!/usr/bin/env python3
"""
Report generator for LLM Memory Estimator
"""

from typing import Dict, Optional, Any

from .model_config import ModelConfig, MemoryResult


class ReportGenerator:
    """Generate markdown reports"""

    @staticmethod
    def generate_report(config: ModelConfig, result: MemoryResult,
                       batch_size: int, seq_len: int,
                       parallel_config: Dict[str, int],
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
        lines.append(f"- **Sequence Length**: {seq_len}")
        lines.append(f"- **Tensor Parallel (TP)**: {parallel_config.get('tp', 1)}")
        lines.append(f"- **Pipeline Parallel (PP)**: {parallel_config.get('pp', 1)}")
        lines.append(f"- **Data Parallel (DP)**: {parallel_config.get('dp', 1)}")
        lines.append(f"- **Context Parallel (CP)**: {parallel_config.get('cp', 1)}")
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

        # Weights Breakdown
        if result.breakdown:
            lines.append("## Weights Breakdown by Module")
            lines.append("")
            lines.append("| Module Type | Memory (GB) | Percentage |")
            lines.append("|-------------|-------------|------------|")

            for module_type, memory in sorted(result.breakdown.items(), key=lambda x: x[1], reverse=True):
                pct = memory / result.weights_memory_gb * 100 if result.weights_memory_gb > 0 else 0
                lines.append(f"| {module_type} | {memory:.2f} | {pct:.1f}% |")
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
