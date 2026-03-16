#!/usr/bin/env python3
"""
Tests for FormulaEvaluator
"""

import pytest

from llm_mem_estimator import FormulaEvaluator
from llm_mem_estimator.model_config import ArchitectureConfig


class TestFormulaEvaluator:
    """Test cases for FormulaEvaluator"""

    def test_evaluate_simple_formula(self):
        """Test evaluating a simple formula"""
        arch = ArchitectureConfig(
            hidden_size=4096,
            num_layers=32,
            attention_type="gqa",
            ffn_type="standard",
            norm_type="rmsnorm",
            vocab_size=32000
        )
        evaluator = FormulaEvaluator(arch)

        formula = "a + b * c"
        result = evaluator.evaluate(formula, a=10, b=5, c=2)
        assert result == 20.0

    def test_evaluate_with_context_variables(self):
        """Test formula evaluation with context variables"""
        arch = ArchitectureConfig(
            hidden_size=4096,
            num_layers=32,
            attention_type="gqa",
            ffn_type="standard",
            norm_type="rmsnorm",
            vocab_size=32000
        )
        evaluator = FormulaEvaluator(arch)

        formula = "batch_size * seq_len * hidden_size"
        result = evaluator.evaluate(formula, batch_size=2, seq_len=2048)
        assert result == 2 * 2048 * 4096

    def test_evaluate_kv_cache_formula(self):
        """Test KV Cache formula evaluation"""
        arch = ArchitectureConfig(
            hidden_size=4096,
            num_layers=32,
            attention_type="gqa",
            ffn_type="standard",
            norm_type="rmsnorm",
            vocab_size=32000,
            num_attention_heads=32,
            num_key_value_heads=8
        )
        evaluator = FormulaEvaluator(arch)

        # GQA formula: 2 * batch_size * seq_len * kv_dim * num_layers
        # kv_dim = num_key_value_heads * head_dim = 8 * 128 = 1024
        formula = "2 * batch_size * seq_len * num_key_value_heads * head_dim * num_layers"
        result = evaluator.evaluate(formula, batch_size=1, seq_len=2048, head_dim=128)
        expected = 2 * 1 * 2048 * 8 * 128 * 32
        assert result == expected

    def test_evaluate_mla_kv_cache_formula(self):
        """Test MLA KV Cache formula evaluation"""
        arch = ArchitectureConfig(
            hidden_size=4096,
            num_layers=32,
            attention_type="mla",
            ffn_type="standard",
            norm_type="rmsnorm",
            vocab_size=32000,
            kv_lora_rank=512,
            qk_rope_head_dim=64
        )
        evaluator = FormulaEvaluator(arch)

        # MLA formula: 2 * batch_size * seq_len * (kv_lora_rank + qk_rope_head_dim) * num_layers
        formula = "2 * batch_size * seq_len * (kv_lora_rank + qk_rope_head_dim) * num_layers"
        result = evaluator.evaluate(formula, batch_size=1, seq_len=4096)
        expected = 2 * 1 * 4096 * (512 + 64) * 32
        assert result == expected

    def test_evaluate_activation_formula_with_capacity_factor(self):
        """Test activation formula with capacity factor"""
        arch = ArchitectureConfig(
            hidden_size=4096,
            num_layers=32,
            attention_type="gqa",
            ffn_type="standard",
            norm_type="rmsnorm",
            vocab_size=32000
        )
        evaluator = FormulaEvaluator(arch)

        # Activation formula with capacity factor
        formula = "batch_size * seq_len * hidden_size * num_layers * recommended_capacity_factor * dtype_bytes"
        result = evaluator.evaluate(
            formula,
            batch_size=1,
            seq_len=2048,
            recommended_capacity_factor=1.25,
            dtype_bytes=2
        )
        expected = 1 * 2048 * 4096 * 32 * 1.25 * 2
        assert result == expected

    def test_evaluate_moe_activation_formula(self):
        """Test MoE activation formula"""
        arch = ArchitectureConfig(
            hidden_size=4096,
            num_layers=32,
            attention_type="gqa",
            ffn_type="moe",
            norm_type="rmsnorm",
            vocab_size=32000,
            num_experts=8,
            num_experts_per_tok=2
        )
        evaluator = FormulaEvaluator(arch)

        # MoE activation formula
        formula = "batch_size * seq_len * hidden_size * num_layers * num_experts_per_tok * recommended_capacity_factor * dtype_bytes"
        result = evaluator.evaluate(
            formula,
            batch_size=1,
            seq_len=2048,
            recommended_capacity_factor=1.25,
            dtype_bytes=2
        )
        expected = 1 * 2048 * 4096 * 32 * 2 * 1.25 * 2
        assert result == expected

    def test_evaluate_invalid_formula(self):
        """Test evaluating an invalid formula"""
        arch = ArchitectureConfig(
            hidden_size=4096,
            num_layers=32,
            attention_type="gqa",
            ffn_type="standard",
            norm_type="rmsnorm",
            vocab_size=32000
        )
        evaluator = FormulaEvaluator(arch)

        formula = "a / b"
        with pytest.raises(ValueError):
            evaluator.evaluate(formula, a=10, b=0)

    def test_context_includes_all_architecture_params(self):
        """Test that evaluator context includes all architecture parameters"""
        arch = ArchitectureConfig(
            hidden_size=4096,
            num_layers=32,
            attention_type="gqa",
            ffn_type="moe",
            norm_type="rmsnorm",
            vocab_size=32000,
            num_attention_heads=32,
            num_key_value_heads=8,
            intermediate_size=11008,
            num_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=1408,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            v_head_dim=128,
            qk_nope_head_dim=128
        )
        evaluator = FormulaEvaluator(arch)

        # Verify all parameters are in context
        assert evaluator.context['hidden_size'] == 4096
        assert evaluator.context['num_layers'] == 32
        assert evaluator.context['num_attention_heads'] == 32
        assert evaluator.context['num_key_value_heads'] == 8
        assert evaluator.context['intermediate_size'] == 11008
        assert evaluator.context['num_experts'] == 8
        assert evaluator.context['num_experts_per_tok'] == 2
        assert evaluator.context['moe_intermediate_size'] == 1408
        assert evaluator.context['q_lora_rank'] == 1536
        assert evaluator.context['kv_lora_rank'] == 512
        assert evaluator.context['qk_rope_head_dim'] == 64
        assert evaluator.context['v_head_dim'] == 128
        assert evaluator.context['qk_nope_head_dim'] == 128
