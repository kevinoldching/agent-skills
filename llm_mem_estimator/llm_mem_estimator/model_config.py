#!/usr/bin/env python3
"""
Data structures and utility functions for LLM Memory Estimator
"""

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
