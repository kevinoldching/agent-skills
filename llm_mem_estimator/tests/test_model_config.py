#!/usr/bin/env python3
"""
Tests for Model Configuration
Combines test_config_loader.py and test_formula_evaluator.py

Tests are based on outputs/ directory which contains generated model configs.
Run test_model_detection.py first to generate the configs.
"""

import pytest
import tempfile
import os
from pathlib import Path

from llm_mem_estimator import ConfigLoader, FormulaEvaluator
from llm_mem_estimator.model_config import (
    ModelConfig, ModelIdentity, ArchitectureConfig, WeightInfo
)
from tests.conftest import get_outputs_configs, get_outputs_dir


# ============================================================================
# Parameterized Tests based on outputs/
# ============================================================================

class TestConfigFromOutputs:
    """基于 outputs/ 目录的模型配置测试（参数化）"""

    @pytest.fixture(autouse=True)
    def check_outputs(self):
        """检查 outputs 目录"""
        configs = get_outputs_configs()
        if not configs:
            pytest.skip("No configs in outputs/. Run test_model_detection.py first.")

    @pytest.mark.parametrize("config_path", get_outputs_configs(), ids=lambda p: p.stem.replace("_config", ""))
    def test_load_yaml_config(self, config_path):
        """测试加载 YAML 配置文件"""
        config = ConfigLoader.load_yaml_config(str(config_path))

        # 验证基本结构
        assert config.model_identity.name is not None
        assert config.model_identity.num_layers > 0
        assert config.architecture_config.hidden_size > 0
        assert config.architecture_config.num_layers > 0

    @pytest.mark.parametrize("config_path", get_outputs_configs(), ids=lambda p: p.stem.replace("_config", ""))
    def test_config_has_modules(self, config_path):
        """测试配置包含模块"""
        config = ConfigLoader.load_yaml_config(str(config_path))

        # 验证模块存在
        assert len(config.modules) > 0

    @pytest.mark.parametrize("config_path", get_outputs_configs(), ids=lambda p: p.stem.replace("_config", ""))
    def test_config_has_computation_rules(self, config_path):
        """测试配置包含计算规则"""
        config = ConfigLoader.load_yaml_config(str(config_path))

        # 验证计算规则存在
        assert len(config.computation_rules) > 0
        assert "kv_cache" in config.computation_rules or "activation" in config.computation_rules


# ============================================================================
# Data Structure Tests (non-parameterized)
# ============================================================================

class TestConfigLoader:
    """Test cases for ConfigLoader (generic tests)"""

    def test_load_chips_config(self):
        """Test loading chips configuration"""
        chips = ConfigLoader.load_chips_config("configs/chips.json")

        assert isinstance(chips, dict)
        assert len(chips) > 0

        # Chips.json has vendor as top-level key (e.g., "nvidia")
        # Get first vendor's chips
        first_vendor = next(iter(chips.keys()))
        vendor_chips = chips[first_vendor]
        assert isinstance(vendor_chips, dict)
        assert len(vendor_chips) > 0

        # Check first chip has required fields
        first_chip_name = next(iter(vendor_chips.keys()))
        first_chip_info = vendor_chips[first_chip_name]
        assert "vram_gb" in first_chip_info
        assert "bandwidth_gb_s" in first_chip_info

    def test_load_weight_mapping_rules(self):
        """Test loading weight mapping rules"""
        rules = ConfigLoader.load_weight_mapping_rules("configs/weight_mapping_rules.yaml")

        assert "generic" in rules
        assert "embedding" in rules["generic"]
        assert "attention" in rules["generic"]
        assert "ffn_moe" in rules["generic"]


class TestModelIdentity:
    """Test cases for ModelIdentity"""

    def test_model_identity_creation(self):
        """Test creating ModelIdentity"""
        identity = ModelIdentity(
            name="test-model",
            total_params="1000000000",
            num_layers=12,
            quantization=None
        )

        assert identity.name == "test-model"
        assert identity.total_params == "1000000000"
        assert identity.num_layers == 12
        assert identity.quantization is None


class TestArchitectureConfig:
    """Test cases for ArchitectureConfig"""

    def test_architecture_config_creation(self):
        """Test creating ArchitectureConfig"""
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

        assert arch.hidden_size == 4096
        assert arch.num_layers == 32
        assert arch.attention_type == "gqa"
        assert arch.num_key_value_heads == 8


class TestWeightInfo:
    """Test cases for WeightInfo"""

    def test_weight_info_creation(self):
        """Test creating WeightInfo"""
        weight = WeightInfo(
            shape=[4096, 4096],
            dtype="BF16",
            layers=32,
            parallel_strategy="tp_col",
            world_size=8
        )

        assert weight.shape == [4096, 4096]
        assert weight.dtype == "BF16"
        assert weight.layers == 32
        assert weight.parallel_strategy == "tp_col"
        assert weight.world_size == 8

    def test_weight_info_defaults(self):
        """Test WeightInfo default values"""
        weight = WeightInfo(
            shape=[4096, 4096],
            dtype="BF16",
            layers=1
        )

        assert weight.parallel_strategy == "replicated"
        assert weight.world_size == 0


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
