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

    def calculate_weights_memory(self, tp: int = 1, pp: int = 1, dp: int = 1, cp: int = 1, ep: int = 1) -> Tuple[float, Dict[str, float]]:
        """Calculate total weights memory and breakdown by module type

        Args:
            tp: Tensor Parallel degree
            pp: Pipeline Parallel degree
            dp: Data Parallel degree
            cp: Context Parallel degree
            ep: Expert Parallel degree
        """
        total_memory = 0.0
        breakdown = {}

        for module_type, weights in self.config.modules.items():
            module_memory = 0.0
            for weight_name, weight_info in weights.items():
                weight_memory = calculate_weight_memory(weight_info, tp, pp, dp, cp, ep)
                module_memory += weight_memory

            breakdown[module_type] = module_memory
            total_memory += module_memory

        return total_memory, breakdown

    def calculate_kv_cache_memory(self, batch_size: int, prompt_len: int, gen_len: int,
                                   dtype: str = "fp16", tp: int = 1, cp: int = 1) -> float:
        """Calculate KV cache memory

        Args:
            batch_size: Batch size
            prompt_len: Input prompt length
            gen_len: Generated output length
            dtype: KV cache data type
            tp: Tensor Parallel degree
            cp: Context Parallel degree
        """
        if 'kv_cache' not in self.config.computation_rules:
            return 0.0

        formula = self.config.computation_rules['kv_cache']
        dtype_bytes = get_dtype_bytes(dtype)

        # Total sequence length = prompt + generated
        total_seq_len = prompt_len + gen_len

        # Evaluate formula
        memory_elements = self.evaluator.evaluate(
            formula,
            batch_size=batch_size,
            prompt_len=prompt_len,
            gen_len=gen_len,
            seq_len=total_seq_len
        )

        # Convert to GB
        memory_gb = (memory_elements * dtype_bytes) / (1024 ** 3)

        # Apply parallel strategies: TP (num_kv_heads sharded), CP (sequence sharded)
        memory_gb = memory_gb / tp / cp

        return memory_gb

    def calculate_activation_memory(self, batch_size: int, gen_len: int,
                                     dtype: str = "fp16", tp: int = 1, cp: int = 1) -> float:
        """Calculate activation memory

        Args:
            batch_size: Batch size
            gen_len: Generated output length (activation only depends on gen_len)
            dtype: Activation data type
            tp: Tensor Parallel degree
            cp: Context Parallel degree
        """
        if 'activation' not in self.config.computation_rules:
            return 0.0

        formula = self.config.computation_rules['activation']
        dtype_bytes = get_dtype_bytes(dtype)

        # Evaluate formula (returns element count)
        memory_elements = self.evaluator.evaluate(
            formula,
            batch_size=batch_size,
            gen_len=gen_len
        )

        # Convert to GB (multiply by dtype_bytes)
        memory_gb = (memory_elements * dtype_bytes) / (1024 ** 3)

        # Apply parallel strategies (only CP, TP/EP don't affect per-device memory)
        memory_gb = memory_gb / cp

        return memory_gb

    def estimate_memory(self, batch_size: int = 1, prompt_len: int = 4096, gen_len: int = 1024,
                        kv_dtype: str = "fp16", activation_dtype: str = "fp16",
                        tp: int = 1, pp: int = 1, dp: int = 1, cp: int = 1, ep: int = 1,
                        system_reserved_gb: float = 2.0) -> MemoryResult:
        """Estimate total memory usage

        Args:
            batch_size: Batch size
            prompt_len: Input prompt length
            gen_len: Generated output length
            kv_dtype: KV cache data type
            activation_dtype: Activation data type
            tp: Tensor Parallel degree
            pp: Pipeline Parallel degree
            dp: Data Parallel degree
            cp: Context Parallel degree
            ep: Expert Parallel degree
            system_reserved_gb: System reserved memory in GB
        """
        # Calculate weights memory (with parallel strategy sharding)
        weights_memory, weights_breakdown = self.calculate_weights_memory(
            tp=tp, pp=pp, dp=dp, cp=cp, ep=ep
        )

        # Calculate KV cache memory (prompt_len + gen_len)
        kv_cache_memory = self.calculate_kv_cache_memory(
            batch_size, prompt_len, gen_len, kv_dtype, tp, cp
        )

        # Calculate activation memory (only gen_len)
        activation_memory = self.calculate_activation_memory(
            batch_size, gen_len, activation_dtype, tp, cp
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
                                  prompt_len: int = 4096,
                                  kv_dtype: str = "fp16", activation_dtype: str = "fp16",
                                  tp: int = 1, pp: int = 1, cp: int = 1, ep: int = 1,
                                  system_reserved_gb: float = 2.0) -> int:
        """Binary search to find maximum generated length (gen_len)

        Args:
            available_memory_gb: Available GPU memory in GB
            batch_size: Batch size
            prompt_len: Input prompt length (fixed)
            kv_dtype: KV cache data type
            activation_dtype: Activation data type
            tp: Tensor Parallel degree
            pp: Pipeline Parallel degree
            cp: Context Parallel degree
            ep: Expert Parallel degree
            system_reserved_gb: System reserved memory in GB
        """
        # Calculate fixed memory (weights + system reserved)
        weights_memory, _ = self.calculate_weights_memory(tp=tp, pp=pp, cp=cp, ep=ep)
        fixed_memory = weights_memory + system_reserved_gb

        # KV cache for prompt (fixed)
        prompt_kv_memory = self.calculate_kv_cache_memory(
            batch_size, prompt_len, 0, kv_dtype, tp, cp
        )

        # Total fixed memory including prompt KV
        total_fixed = fixed_memory + prompt_kv_memory

        if total_fixed >= available_memory_gb:
            return 0

        # Available memory for generated KV + activation
        dynamic_memory = available_memory_gb - total_fixed

        # Estimate per-token memory with seq_len=1 to set upper bound
        per_token_kv = self.calculate_kv_cache_memory(batch_size, prompt_len, 1, kv_dtype, tp, cp)
        per_token_act = self.calculate_activation_memory(batch_size, 1, activation_dtype, tp, cp)
        per_token_total = per_token_kv + per_token_act

        # Set upper bound: available memory / per-token memory * 10
        # Use a safe upper bound if per_token_total is too small
        if per_token_total > 0:
            upper_bound = min(int(dynamic_memory / per_token_total * 10), 10000000)
        else:
            upper_bound = 1000000

        # Binary search for max generated length
        left, right = 1, upper_bound
        max_gen_len = 0

        while left <= right:
            mid = (left + right) // 2

            # KV grows with prompt + gen_len, activation only with gen_len
            kv_memory = self.calculate_kv_cache_memory(
                batch_size, prompt_len, mid, kv_dtype, tp, cp
            )
            act_memory = self.calculate_activation_memory(
                batch_size, mid, activation_dtype, tp, cp
            )
            total_dynamic = kv_memory + act_memory

            if total_dynamic <= dynamic_memory:
                max_gen_len = mid
                left = mid + 1
            else:
                right = mid - 1

        return max_gen_len
