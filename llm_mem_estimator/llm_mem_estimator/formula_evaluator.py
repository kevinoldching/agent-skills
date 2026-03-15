#!/usr/bin/env python3
"""
Formula evaluator for LLM Memory Estimator
"""

from typing import Dict, Any

from .model_config import ArchitectureConfig


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
