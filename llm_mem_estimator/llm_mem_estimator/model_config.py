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
from dataclasses import dataclass
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


def calculate_weight_memory(weight_info: WeightInfo) -> float:
    """Calculate memory for a single weight in GB"""
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

    # Apply parallel strategy sharding
    if weight_info.parallel_strategy != "replicated" and weight_info.world_size > 0:
        total_bytes /= weight_info.world_size

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
        architecture_config = ArchitectureConfig(
            hidden_size=arch_data.get('hidden_size', 0),
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
            qk_nope_head_dim=arch_data.get('qk_nope_head_dim')
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


# ============================================================================
# Formula Evaluator
# ============================================================================

class FormulaEvaluator:
    """Evaluate computation formulas from configuration"""

    def __init__(self, architecture_config: ArchitectureConfig):
        self.arch = architecture_config
        self.context = self._build_context()

    def _build_context(self) -> Dict[str, Any]:
        """Build evaluation context from architecture config"""
        context = {
            'hidden_size': self.arch.hidden_size,
            'num_layers': self.arch.num_layers,
            'vocab_size': self.arch.vocab_size,
        }

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

        return context

    def evaluate(self, formula: str, **kwargs) -> float:
        """Evaluate a formula string with given context"""
        # Merge kwargs into context
        eval_context = {**self.context, **kwargs}

        try:
            # Safely evaluate the formula
            result = eval(formula, {"__builtins__": {}}, eval_context)
            return float(result)
        except Exception as e:
            raise ValueError(f"Failed to evaluate formula '{formula}': {e}")
