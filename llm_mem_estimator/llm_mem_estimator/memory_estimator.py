#!/usr/bin/env python3
"""
Memory estimator for LLM Memory Estimator
"""

from typing import Dict, Tuple, Literal, Optional

from .model_config import ModelConfig, MemoryResult, calculate_weight_memory, get_dtype_bytes, FormulaEvaluator, WeightInfo


class MemoryEstimator:
    """Main controller for memory estimation"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.evaluator = FormulaEvaluator(config.architecture_config, config.computation_rules)

    def calculate_weights_memory(self, tp: int = 1, pp: int = 1, dp: int = 1, cp: int = 1, ep: int = 1,
                                  stage: Literal["hybrid", "prefill", "decode"] = "hybrid") -> Tuple[float, Dict[str, float]]:
        """Calculate total weights memory and breakdown by module type

        Args:
            tp: Tensor Parallel degree
            pp: Pipeline Parallel degree
            dp: Data Parallel degree
            cp: Context Parallel degree
            ep: Expert Parallel degree
            stage: PD separation stage - "hybrid" (混部/通用), "prefill" (PD分离-prefill), "decode" (PD分离-decode)
        """
        total_memory = 0.0
        breakdown = {}

        # Check if we have a classifier for stage-specific parallel strategy
        classifier = self.config.weight_classifier
        model_type = self.config.model_type

        for module_type, weights in self.config.modules.items():
            module_memory = 0.0
            for weight_name, weight_info in weights.items():
                # If classifier available, recompute parallel strategy with stage
                if classifier is not None:
                    parallel_strategy = classifier.get_parallel_strategy(
                        weight_name, module_type, model_type, stage=stage
                    )
                    # Create a temporary WeightInfo with updated strategy
                    updated_weight_info = WeightInfo(
                        shape=weight_info.shape,
                        dtype=weight_info.dtype,
                        layers=weight_info.layers,
                        parallel_strategy=parallel_strategy,
                        world_size=weight_info.world_size
                    )
                    weight_memory = calculate_weight_memory(updated_weight_info, tp, pp, dp, cp, ep)
                else:
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

        # Evaluate formula (includes tp_size/cp_size in the formula)
        memory_elements = self.evaluator.evaluate(
            formula,
            batch_size=batch_size,
            prompt_len=prompt_len,
            gen_len=gen_len,
            seq_len=total_seq_len,
            tp_size=tp,
            cp_size=cp
        )

        # Convert to GB
        memory_gb = (memory_elements * dtype_bytes) / (1024 ** 3)

        return memory_gb

    def calculate_activation_memory(self, batch_size: int, seq_len: int,
                                     dtype: str = "fp16", tp: int = 1, cp: int = 1,
                                     use_decode_factor: bool = True) -> float:
        """Calculate activation memory

        Args:
            batch_size: Batch size
            seq_len: Sequence length (for activation calculation)
            dtype: Activation data type
            tp: Tensor Parallel degree
            cp: Context Parallel degree
            use_decode_factor: If True, use decode factor (12.5); otherwise use has_prefill factor (1.25)
        """
        if 'activation' not in self.config.computation_rules:
            return 0.0

        formula = self.config.computation_rules['activation']
        dtype_bytes = get_dtype_bytes(dtype)

        # Evaluate formula (includes tp_size/cp_size in the formula)
        memory_elements = self.evaluator.evaluate(
            formula,
            use_decode_factor=use_decode_factor,
            batch_size=batch_size,
            gen_len=seq_len,
            seq_len=seq_len,
            tp_size=tp,
            cp_size=cp
        )

        # Convert to GB
        memory_gb = (memory_elements * dtype_bytes) / (1024 ** 3)

        return memory_gb

    def estimate_memory(self, batch_size: int = 1, prompt_len: int = 4096, gen_len: int = 1024,
                        kv_dtype: str = "fp16", activation_dtype: str = "fp16",
                        tp: int = 1, pp: int = 1, dp: int = 1, cp: int = 1, ep: int = 1,
                        system_reserved_gb: float = 2.0,
                        use_decode_factor: bool = True,
                        activation_peak_gb: float = None,
                        stage: Literal["hybrid", "prefill", "decode"] = "hybrid") -> MemoryResult:
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
            use_decode_factor: If True, use decode factor (12.5) with seq_len=1; otherwise use has_prefill factor (1.25) with seq_len=total_seq_len
            activation_peak_gb: If specified, use this fixed activation peak value (GB) instead of calculating from formula
            stage: PD separation stage - "hybrid" (混部/通用), "prefill" (PD分离-prefill), "decode" (PD分离-decode)
        """
        # Calculate weights memory (with parallel strategy sharding)
        weights_memory, weights_breakdown = self.calculate_weights_memory(
            tp=tp, pp=pp, dp=dp, cp=cp, ep=ep, stage=stage
        )

        # Calculate KV cache memory (prompt_len + gen_len)
        kv_cache_memory = self.calculate_kv_cache_memory(
            batch_size, prompt_len, gen_len, kv_dtype, tp, cp
        )

        # Calculate activation memory
        if activation_peak_gb is not None:
            # Use user-specified fixed activation peak value
            activation_memory = activation_peak_gb
        elif use_decode_factor:
            # Decode scenario: seq_len = 1
            activation_memory = self.calculate_activation_memory(
                batch_size, 1, activation_dtype, tp, cp,
                use_decode_factor=True
            )
        else:
            # Prefill scenario: seq_len = total_seq_len
            total_seq_len = prompt_len + gen_len
            activation_memory = self.calculate_activation_memory(
                batch_size, total_seq_len, activation_dtype, tp, cp,
                use_decode_factor=False
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
            breakdown=weights_breakdown,
            stage=stage
        )

    def find_max_sequence_length(self, available_memory_gb: float, batch_size: int = 1,
                                  prompt_len: int = 4096,
                                  kv_dtype: str = "fp16", activation_dtype: str = "fp16",
                                  tp: int = 1, pp: int = 1, cp: int = 1, ep: int = 1,
                                  system_reserved_gb: float = 2.0,
                                  use_decode_factor: bool = True,
                                  activation_peak_gb: float = None,
                                  stage: Literal["hybrid", "prefill", "decode"] = "hybrid") -> int:
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
            use_decode_factor: If True, use decode factor (12.5); otherwise use has_prefill factor (1.25)
            activation_peak_gb: If specified, use this fixed activation peak value (GB)
        """
        # Calculate fixed memory (weights + system reserved)
        weights_memory, _ = self.calculate_weights_memory(tp=tp, pp=pp, cp=cp, ep=ep, stage=stage)
        fixed_memory = weights_memory + system_reserved_gb

        # KV cache for prompt (fixed)
        prompt_kv_memory = self.calculate_kv_cache_memory(
            batch_size, prompt_len, 0, kv_dtype, tp, cp
        )

        # Activation is always fixed in Decode scenario (seq_len=1, not incremental per token)
        # Use user-specified value if provided, otherwise calculate with decode factor
        if activation_peak_gb is not None:
            activation_fixed = activation_peak_gb
        else:
            # Decode stage: seq_len = 1, use decode factor
            activation_fixed = self.calculate_activation_memory(
                batch_size, 1, activation_dtype, tp, cp,
                use_decode_factor=True
            )

        # Total fixed memory including prompt KV and activation
        total_fixed = fixed_memory + prompt_kv_memory + activation_fixed

        if total_fixed >= available_memory_gb:
            return 0

        # Available memory for generated KV only (activation is fixed)
        dynamic_memory = available_memory_gb - total_fixed

        # Use a fixed large upper bound for binary search
        # The actual max will be found by the binary search using the real formula
        upper_bound = 100_000_000  # 100 million

        # Binary search for max generated length
        left, right = 1, upper_bound
        max_gen_len = 0

        while left <= right:
            mid = (left + right) // 2

            # KV grows with prompt + gen_len
            kv_memory = self.calculate_kv_cache_memory(
                batch_size, prompt_len, mid, kv_dtype, tp, cp
            )

            # Activation is already accounted in total_fixed (either user-specified or calculated with seq_len=1)
            # In Decode stage, activation does NOT grow with gen_len
            act_memory = 0

            total_dynamic = kv_memory + act_memory

            if total_dynamic <= dynamic_memory:
                max_gen_len = mid
                left = mid + 1
            else:
                right = mid - 1

        return max_gen_len

    def find_max_prompt_len(self, available_memory_gb: float, batch_size: int = 1,
                            gen_len: int = 1,
                            kv_dtype: str = "fp16", activation_dtype: str = "fp16",
                            tp: int = 1, pp: int = 1, cp: int = 1, ep: int = 1,
                            system_reserved_gb: float = 2.0,
                            activation_peak_gb: float = None,
                            stage: Literal["hybrid", "prefill", "decode"] = "hybrid") -> int:
        """Binary search to find maximum prompt length (prompt_len) with fixed gen_len

        This is used for PD separation scenarios where we want to find the maximum
        prompt length that can fit in memory when generation length is fixed.

        Args:
            available_memory_gb: Available GPU memory in GB
            batch_size: Batch size
            gen_len: Generated output length (fixed)
            kv_dtype: KV cache data type
            activation_dtype: Activation data type
            tp: Tensor Parallel degree
            pp: Pipeline Parallel degree
            cp: Context Parallel degree
            ep: Expert Parallel degree
            system_reserved_gb: System reserved memory in GB
            activation_peak_gb: If specified, use this fixed activation peak value (GB)
            stage: PD separation stage - "hybrid" (混部/通用), "prefill" (PD分离-prefill), "decode" (PD分离-decode)
        """
        # Calculate fixed memory (weights + system reserved)
        weights_memory, _ = self.calculate_weights_memory(tp=tp, pp=pp, cp=cp, ep=ep, stage=stage)
        fixed_memory = weights_memory + system_reserved_gb

        # Activation peak is fixed when specified (not incremental per token)
        activation_fixed = activation_peak_gb if activation_peak_gb is not None else 0

        # Use a fixed large upper bound for binary search
        upper_bound = 100_000_000  # 100 million

        # Binary search for max prompt length
        left, right = 1, upper_bound
        max_prompt_len = 0

        while left <= right:
            mid = (left + right) // 2

            # Total sequence length = prompt_len + gen_len
            total_seq_len = mid + gen_len

            # KV grows with prompt_len + gen_len
            kv_memory = self.calculate_kv_cache_memory(
                batch_size, mid, gen_len, kv_dtype, tp, cp
            )

            # Activation: 0 if activation_peak_gb is specified (already in fixed_memory)
            # Otherwise calculate from formula (incremental based on total_seq_len)
            if activation_peak_gb is not None:
                act_memory = 0  # Already accounted in fixed_memory
            else:
                # Prefill stage: use has_prefill factor
                act_memory = self.calculate_activation_memory(
                    batch_size, total_seq_len, activation_dtype, tp, cp,
                    use_decode_factor=False
                )

            total_memory = fixed_memory + activation_fixed + kv_memory + act_memory

            if total_memory <= available_memory_gb:
                max_prompt_len = mid
                left = mid + 1
            else:
                right = mid - 1

        return max_prompt_len

    def find_max_batch_size(self, available_memory_gb: float, prompt_len: int, gen_len: int,
                            kv_dtype: str = "fp16", activation_dtype: str = "fp16",
                            tp: int = 1, pp: int = 1, cp: int = 1, ep: int = 1,
                            system_reserved_gb: float = 2.0,
                            activation_peak_gb: float = None,
                            stage: Literal["hybrid", "prefill", "decode"] = "hybrid") -> int:
        """Binary search to find maximum batch_size that fits in available memory

        This is used for Scene 8 scenarios where both prompt_len and gen_len are fixed,
        and we want to find the maximum batch size that can fit in memory.

        Args:
            available_memory_gb: Available GPU memory in GB
            prompt_len: Input prompt length (fixed)
            gen_len: Generated output length (fixed)
            kv_dtype: KV cache data type
            activation_dtype: Activation data type
            tp: Tensor Parallel degree
            pp: Pipeline Parallel degree
            cp: Context Parallel degree
            ep: Expert Parallel degree
            system_reserved_gb: System reserved memory in GB
            activation_peak_gb: If specified, use this fixed activation peak value (GB)
            stage: PD separation stage - "hybrid" (混部/通用), "prefill" (PD分离-prefill), "decode" (PD分离-decode)
        """
        # Calculate fixed memory (weights + system reserved)
        weights_memory, _ = self.calculate_weights_memory(tp=tp, pp=pp, cp=cp, ep=ep, stage=stage)
        fixed_memory = weights_memory + system_reserved_gb

        # Activation peak is fixed when specified (not incremental per token)
        activation_fixed = activation_peak_gb if activation_peak_gb is not None else 0

        # Use a large upper bound for binary search
        upper_bound = 10000

        # Binary search for max batch size
        left, right = 1, upper_bound
        max_batch_size = 0

        while left <= right:
            mid = (left + right) // 2

            # KV memory scales with batch_size
            kv_memory = self.calculate_kv_cache_memory(
                mid, prompt_len, gen_len, kv_dtype, tp, cp
            )

            # Activation: 0 if activation_peak_gb is specified (already in fixed_memory)
            # Otherwise calculate from formula (scales with batch_size)
            if activation_peak_gb is not None:
                act_memory = 0  # Already accounted in fixed_memory
            else:
                # Prefill stage: has_prefill factor
                act_memory = self.calculate_activation_memory(
                    mid, prompt_len + gen_len, activation_dtype, tp, cp,
                    use_decode_factor=False
                )

            total_memory = fixed_memory + activation_fixed + kv_memory + act_memory

            if total_memory <= available_memory_gb:
                max_batch_size = mid
                left = mid + 1
            else:
                right = mid - 1

        return max_batch_size
