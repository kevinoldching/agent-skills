#!/usr/bin/env python3
"""
Data structures, configuration loading, and formula evaluation for LLM Memory Estimator

This module combines:
- Data structures (ModelConfig, WeightInfo, etc.)
- Configuration loading (YAML, chips.json, weight_mapping_rules)
- Formula evaluation (FormulaEvaluator)
"""

import json
import re
import yaml
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class WeightInfo:
    """Information about a model weight"""
    shape: List[int]
    dtype: str
    layers: int = 1  # Number of layers this weight appears in
    parallel_strategy: str = "replicated"
    world_size: int = 0  # 0 means no sharding


@dataclass
class ModelIdentity:
    """Model identity information"""
    name: str
    total_params: str
    num_layers: int
    quantization: Optional[str] = None


@dataclass
class ArchitectureConfig:
    """Model architecture configuration"""
    hidden_size: int
    num_layers: int
    attention_type: str
    ffn_type: str
    norm_type: str
    vocab_size: int
    # Optional fields for specific architectures
    head_dim: int = 0
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    intermediate_size: Optional[int] = None
    num_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    # MLA-specific fields
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    # Sliding window attention
    window_size: Optional[int] = None


@dataclass
class ModelConfig:
    """Complete model configuration"""
    model_identity: ModelIdentity
    architecture_config: ArchitectureConfig
    modules: Dict[str, Dict[str, WeightInfo]]  # module_type -> {weight_name: WeightInfo}
    computation_rules: Dict[str, Any]  # rule_name -> formula or value


@dataclass
class MemoryResult:
    """Memory calculation result"""
    total_memory_gb: float
    weights_memory_gb: float
    kv_cache_memory_gb: float
    activation_memory_gb: float
    system_reserved_gb: float
    breakdown: Dict[str, float]  # Detailed breakdown by module type
    max_sequence_length: Optional[int] = None
    max_batch_size: Optional[int] = None


# ============================================================================
# Utility Functions
# ============================================================================

def get_dtype_bytes(dtype: str) -> float:
    """Get the number of bytes for a given data type"""
    dtype_map = {
        "fp32": 4,
        "float32": 4,
        "f32": 4,
        "fp16": 2,
        "float16": 2,
        "f16": 2,
        "bf16": 2,
        "bfloat16": 2,
        "fp8": 1,
        "float8": 1,
        "f8_e4m3": 1,
        "float8_e4m3fn": 1,
        "int8": 1,
        "uint8": 1,
        "u8": 1,
        "int4": 0.5,
        "uint4": 0.5,
        "u4": 0.5,
    }
    dtype_lower = dtype.lower()
    if dtype_lower not in dtype_map:
        raise ValueError(f"Unknown dtype: {dtype}")
    return dtype_map[dtype_lower]


def calculate_weight_memory(weight_info: WeightInfo, tp: int = 1, pp: int = 1, dp: int = 1, cp: int = 1, ep: int = 1) -> float:
    """Calculate memory for a single weight in GB

    Args:
        weight_info: Weight information
        tp: Tensor Parallel degree
        pp: Pipeline Parallel degree
        dp: Data Parallel degree
        cp: Context Parallel degree
        ep: Expert Parallel degree
    """
    # Calculate total elements
    total_elements = 1
    for dim in weight_info.shape:
        total_elements *= dim

    # Multiply by number of layers
    total_elements *= weight_info.layers

    # Get bytes per element
    bytes_per_element = get_dtype_bytes(weight_info.dtype)

    # Calculate total bytes
    total_bytes = total_elements * bytes_per_element

    # Apply parallel strategy sharding based on parallel_strategy
    strategy = weight_info.parallel_strategy.upper() if weight_info.parallel_strategy else ""
    if strategy == "TP":
        total_bytes /= tp
    elif strategy == "PP":
        total_bytes /= pp
    elif strategy == "DP":
        total_bytes /= dp
    elif strategy == "CP":
        total_bytes /= cp
    elif strategy == "EP":
        total_bytes /= ep
    # replicated: no sharding

    # Convert to GB
    return total_bytes / (1024 ** 3)


# ============================================================================
# Configuration Loader
# ============================================================================

class ConfigLoader:
    """Load and validate configuration files"""

    @staticmethod
    def load_yaml_config(config_path: str) -> ModelConfig:
        """Load model configuration from YAML file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Parse model identity
        identity_data = data.get('model_identity', {})
        model_identity = ModelIdentity(
            name=identity_data.get('name', 'unknown'),
            total_params=identity_data.get('total_params', 'unknown'),
            num_layers=identity_data.get('num_layers', 0),
            quantization=identity_data.get('quantization')
        )

        # Parse architecture config
        arch_data = data.get('architecture_config', {})

        # Get head_dim from config or calculate from hidden_size / num_attention_heads
        head_dim = arch_data.get('head_dim', 0)
        if not head_dim:
            hidden_size = arch_data.get('hidden_size', 0)
            num_attention_heads = arch_data.get('num_attention_heads')
            if hidden_size and num_attention_heads:
                head_dim = hidden_size // num_attention_heads

        architecture_config = ArchitectureConfig(
            hidden_size=arch_data.get('hidden_size', 0),
            head_dim=head_dim,
            num_layers=arch_data.get('num_layers', 0),
            attention_type=arch_data.get('attention_type', 'unknown'),
            ffn_type=arch_data.get('ffn_type', 'unknown'),
            norm_type=arch_data.get('norm_type', 'unknown'),
            vocab_size=arch_data.get('vocab_size', 0),
            num_attention_heads=arch_data.get('num_attention_heads'),
            num_key_value_heads=arch_data.get('num_key_value_heads'),
            intermediate_size=arch_data.get('intermediate_size'),
            num_experts=arch_data.get('num_experts'),
            num_experts_per_tok=arch_data.get('num_experts_per_tok'),
            moe_intermediate_size=arch_data.get('moe_intermediate_size'),
            q_lora_rank=arch_data.get('q_lora_rank'),
            kv_lora_rank=arch_data.get('kv_lora_rank'),
            qk_rope_head_dim=arch_data.get('qk_rope_head_dim'),
            v_head_dim=arch_data.get('v_head_dim'),
            qk_nope_head_dim=arch_data.get('qk_nope_head_dim'),
            window_size=arch_data.get('window_size')
        )

        # Parse modules
        modules = {}
        modules_data = data.get('modules', {})
        for module_type, weights_data in modules_data.items():
            weights = {}
            for weight_name, weight_data in weights_data.items():
                weights[weight_name] = WeightInfo(
                    shape=weight_data.get('shape', []),
                    dtype=weight_data.get('dtype', 'fp16'),
                    layers=weight_data.get('layers', 1),
                    parallel_strategy=weight_data.get('parallel_strategy', 'replicated'),
                    world_size=weight_data.get('world_size', 0)
                )
            modules[module_type] = weights

        # Parse computation rules
        computation_rules = data.get('computation_rules', {})

        return ModelConfig(
            model_identity=model_identity,
            architecture_config=architecture_config,
            modules=modules,
            computation_rules=computation_rules
        )

    @staticmethod
    def load_chips_config(chips_path: str) -> Dict[str, Any]:
        """Load hardware chips configuration"""
        with open(chips_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def load_weight_mapping_rules(rules_path: str) -> Dict[str, Any]:
        """Load weight mapping rules"""
        with open(rules_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @staticmethod
    def save_yaml_config(config: ModelConfig, output_path: str) -> None:
        """Save model configuration to YAML file in compact format"""
        yaml_content = ConfigLoader.config_to_yaml(config)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)

    @staticmethod
    def format_params_billions(total_params: str) -> str:
        """Convert total_params to billions (B) format"""
        try:
            # Try to parse as integer
            params = int(total_params)
            billions = params / 1e9
            return f"{billions:.2f}B"
        except (ValueError, TypeError):
            # If already in B format or invalid, return as-is
            return total_params

    @staticmethod
    def config_to_yaml(config: ModelConfig) -> str:
        """Convert ModelConfig to compact YAML format string"""
        lines = []

        # model_identity
        lines.append("model_identity:")
        lines.append(f"  name: {config.model_identity.name}")
        lines.append(f"  total_params: '{ConfigLoader.format_params_billions(config.model_identity.total_params)}'")
        lines.append(f"  num_layers: {config.model_identity.num_layers}")
        if config.model_identity.quantization:
            lines.append(f"  quantization: {config.model_identity.quantization}")

        # architecture_config
        lines.append("architecture_config:")
        arch = config.architecture_config
        lines.append(f"  hidden_size: {arch.hidden_size}")
        lines.append(f"  num_layers: {arch.num_layers}")
        lines.append(f"  attention_type: {arch.attention_type}")
        lines.append(f"  ffn_type: {arch.ffn_type}")
        lines.append(f"  norm_type: {arch.norm_type}")
        lines.append(f"  vocab_size: {arch.vocab_size}")

        # Optional fields
        if arch.head_dim:
            lines.append(f"  head_dim: {arch.head_dim}")
        if arch.num_attention_heads:
            lines.append(f"  num_attention_heads: {arch.num_attention_heads}")
        if arch.num_key_value_heads:
            lines.append(f"  num_key_value_heads: {arch.num_key_value_heads}")
        if arch.intermediate_size:
            lines.append(f"  intermediate_size: {arch.intermediate_size}")
        if arch.num_experts_per_tok:
            lines.append(f"  num_experts_per_tok: {arch.num_experts_per_tok}")
        if arch.window_size:
            lines.append(f"  window_size: {arch.window_size}")
        if arch.q_lora_rank:
            lines.append(f"  q_lora_rank: {arch.q_lora_rank}")
        if arch.kv_lora_rank:
            lines.append(f"  kv_lora_rank: {arch.kv_lora_rank}")
        if arch.qk_rope_head_dim:
            lines.append(f"  qk_rope_head_dim: {arch.qk_rope_head_dim}")
        if arch.v_head_dim:
            lines.append(f"  v_head_dim: {arch.v_head_dim}")
        if arch.qk_nope_head_dim:
            lines.append(f"  qk_nope_head_dim: {arch.qk_nope_head_dim}")

        # modules - using compact format
        lines.append("modules:")
        for module_type, module_weights in config.modules.items():
            lines.append(f"  {module_type}:")
            for weight_name, weight_info in module_weights.items():
                # Compact format: weight_name: {shape: [...], dtype: XXX, layers: N, ...}
                shape_str = str(weight_info.shape)
                parts = [f"shape: {shape_str}", f"dtype: {weight_info.dtype}",
                         f"layers: {weight_info.layers}", f"parallel_strategy: {weight_info.parallel_strategy}"]
                # Note: world_size is determined by CLI parameters (--tp/--ep/etc), not stored in YAML
                lines.append(f"    {weight_name}: {{{', '.join(parts)}}}")

        # computation_rules
        lines.append("computation_rules:")
        for key, value in config.computation_rules.items():
            if isinstance(value, dict):
                # Handle nested dict (like recommended_capacity_factor)
                lines.append(f"  {key}:")
                for sub_key, sub_value in value.items():
                    lines.append(f"    {sub_key}: {sub_value}")
            else:
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)


# ============================================================================
# Formula Evaluator
# ============================================================================

class FormulaEvaluator:
    """Evaluate computation formulas from configuration"""

    def __init__(self, architecture_config: ArchitectureConfig, computation_rules: Dict[str, Any] = None):
        self.arch = architecture_config
        self.computation_rules = computation_rules or {}
        self.context = self._build_context()

    def _build_context(self, use_decode_factor: bool = False) -> Dict[str, Any]:
        """Build evaluation context from architecture config

        Args:
            use_decode_factor: If True, use decode factor (12.5); otherwise use has_prefill factor (1.25)
        """
        context = {
            'hidden_size': self.arch.hidden_size,
            'num_layers': self.arch.num_layers,
            'vocab_size': self.arch.vocab_size,
        }

        # Add recommended_capacity_factor from computation_rules if present
        # Support both old format (single float) and new format (nested dict)
        if 'recommended_capacity_factor' in self.computation_rules:
            rcf = self.computation_rules['recommended_capacity_factor']
            if isinstance(rcf, dict):
                # New nested format: {has_prefill: X, decode: Y}
                if use_decode_factor:
                    context['recommended_capacity_factor'] = rcf.get('decode', 12.5)
                else:
                    context['recommended_capacity_factor'] = rcf.get('has_prefill', 1.25)
            else:
                # Old format: single float value
                context['recommended_capacity_factor'] = rcf

        # Add optional fields if present
        if self.arch.num_attention_heads:
            context['num_attention_heads'] = self.arch.num_attention_heads
        if self.arch.num_key_value_heads:
            context['num_key_value_heads'] = self.arch.num_key_value_heads
        if self.arch.intermediate_size:
            context['intermediate_size'] = self.arch.intermediate_size
        if self.arch.num_experts:
            context['num_experts'] = self.arch.num_experts
        if self.arch.num_experts_per_tok:
            context['num_experts_per_tok'] = self.arch.num_experts_per_tok
        if self.arch.moe_intermediate_size:
            context['moe_intermediate_size'] = self.arch.moe_intermediate_size
        if self.arch.q_lora_rank:
            context['q_lora_rank'] = self.arch.q_lora_rank
        if self.arch.kv_lora_rank:
            context['kv_lora_rank'] = self.arch.kv_lora_rank
        if self.arch.qk_rope_head_dim:
            context['qk_rope_head_dim'] = self.arch.qk_rope_head_dim
        if self.arch.v_head_dim:
            context['v_head_dim'] = self.arch.v_head_dim
        if self.arch.qk_nope_head_dim:
            context['qk_nope_head_dim'] = self.arch.qk_nope_head_dim
        if self.arch.window_size:
            context['window_size'] = self.arch.window_size

        return context

    def evaluate(self, formula: str, use_decode_factor: bool = False, **kwargs) -> float:
        """Evaluate a formula string with given context

        Args:
            formula: The formula string to evaluate
            use_decode_factor: If True, use decode factor (12.5); otherwise use has_prefill factor (1.25)
            **kwargs: Additional variables to include in the evaluation context

        Supports common built-in functions: min, max, abs, round, pow, len
        Example formula: '(18 * seq_len + 18 * min(seq_len, 128)) * kv_dim * num_layers'
        """
        # Build context with the appropriate factor
        base_context = self._build_context(use_decode_factor=use_decode_factor)

        # Merge kwargs into context
        eval_context = {**base_context, **kwargs}

        # Inject default values for parallel sizes if not provided
        if 'tp_size' not in eval_context:
            eval_context['tp_size'] = 1
        if 'cp_size' not in eval_context:
            eval_context['cp_size'] = 1

        # Add safe built-in functions
        safe_builtins = {
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'pow': pow,
        }

        try:
            # Safely evaluate the formula
            result = eval(formula, {"__builtins__": safe_builtins}, eval_context)
            return float(result)
        except Exception as e:
            raise ValueError(f"Failed to evaluate formula '{formula}': {e}")
