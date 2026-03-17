#!/usr/bin/env python3
"""
Memory estimator for LLM Memory Estimator
"""

from typing import Dict, Tuple

from .model_config import ModelConfig, MemoryResult, calculate_weight_memory, get_dtype_bytes, FormulaEvaluator


class MemoryEstimator:
    """Main controller for memory estimation"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.evaluator = FormulaEvaluator(config.architecture_config)

    def calculate_weights_memory(self) -> Tuple[float, Dict[str, float]]:
        """Calculate total weights memory and breakdown by module type"""
        total_memory = 0.0
        breakdown = {}

        for module_type, weights in self.config.modules.items():
            module_memory = 0.0
            for weight_name, weight_info in weights.items():
                weight_memory = calculate_weight_memory(weight_info)
                module_memory += weight_memory

            breakdown[module_type] = module_memory
            total_memory += module_memory

        return total_memory, breakdown

    def calculate_kv_cache_memory(self, batch_size: int, seq_len: int,
                                   dtype: str = "fp16", tp: int = 1, cp: int = 1) -> float:
        """Calculate KV cache memory"""
        if 'kv_cache' not in self.config.computation_rules:
            return 0.0

        formula = self.config.computation_rules['kv_cache']
        dtype_bytes = get_dtype_bytes(dtype)

        # Evaluate formula
        memory_elements = self.evaluator.evaluate(
            formula,
            batch_size=batch_size,
            seq_len=seq_len
        )

        # Convert to GB
        memory_gb = (memory_elements * dtype_bytes) / (1024 ** 3)

        # Apply parallel strategies
        memory_gb = memory_gb / tp / cp

        return memory_gb

    def calculate_activation_memory(self, batch_size: int, seq_len: int,
                                     dtype: str = "fp16", tp: int = 1, cp: int = 1) -> float:
        """Calculate activation memory"""
        if 'activation' not in self.config.computation_rules:
            return 0.0

        formula = self.config.computation_rules['activation']
        dtype_bytes = get_dtype_bytes(dtype)

        # Evaluate formula with dtype_bytes included
        memory_elements = self.evaluator.evaluate(
            formula,
            batch_size=batch_size,
            seq_len=seq_len,
            dtype_bytes=dtype_bytes
        )

        # Convert to GB
        memory_gb = memory_elements / (1024 ** 3)

        # Apply parallel strategies
        memory_gb = memory_gb / tp / cp

        return memory_gb

    def estimate_memory(self, batch_size: int = 1, seq_len: int = 2048,
                        kv_dtype: str = "fp16", activation_dtype: str = "fp16",
                        tp: int = 1, pp: int = 1, dp: int = 1, cp: int = 1,
                        system_reserved_gb: float = 2.0) -> MemoryResult:
        """Estimate total memory usage"""
        # Calculate weights memory
        weights_memory, weights_breakdown = self.calculate_weights_memory()

        # Apply pipeline parallel
        weights_memory = weights_memory / pp

        # Calculate KV cache memory
        kv_cache_memory = self.calculate_kv_cache_memory(
            batch_size, seq_len, kv_dtype, tp, cp
        )

        # Calculate activation memory
        activation_memory = self.calculate_activation_memory(
            batch_size, seq_len, activation_dtype, tp, cp
        )

        # Total memory
        total_memory = (weights_memory + kv_cache_memory +
                       activation_memory + system_reserved_gb)

        return MemoryResult(
            total_memory_gb=total_memory,
            weights_memory_gb=weights_memory,
            kv_cache_memory_gb=kv_cache_memory,
            activation_memory_gb=activation_memory,
            system_reserved_gb=system_reserved_gb,
            breakdown=weights_breakdown
        )

    def find_max_sequence_length(self, available_memory_gb: float, batch_size: int = 1,
                                  kv_dtype: str = "fp16", activation_dtype: str = "fp16",
                                  tp: int = 1, pp: int = 1, cp: int = 1,
                                  system_reserved_gb: float = 2.0) -> int:
        """Binary search to find maximum sequence length"""
        # Calculate fixed memory (weights + system reserved)
        weights_memory, _ = self.calculate_weights_memory()
        weights_memory = weights_memory / pp
        fixed_memory = weights_memory + system_reserved_gb

        if fixed_memory >= available_memory_gb:
            return 0

        # Available memory for KV cache and activation
        dynamic_memory = available_memory_gb - fixed_memory

        # Binary search for max sequence length
        left, right = 1, 1000000
        max_seq_len = 0

        while left <= right:
            mid = (left + right) // 2

            kv_memory = self.calculate_kv_cache_memory(batch_size, mid, kv_dtype, tp, cp)
            act_memory = self.calculate_activation_memory(batch_size, mid, activation_dtype, tp, cp)
            total_dynamic = kv_memory + act_memory

            if total_dynamic <= dynamic_memory:
                max_seq_len = mid
                left = mid + 1
            else:
                right = mid - 1

        return max_seq_len
