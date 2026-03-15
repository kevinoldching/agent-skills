#!/usr/bin/env python3
"""
Tests for ConfigLoader
"""

import pytest
import tempfile
import os
from pathlib import Path

from llm_mem_estimator.config_loader import ConfigLoader
from llm_mem_estimator.model_config import (
    ModelConfig, ModelIdentity, ArchitectureConfig, WeightInfo
)


class TestConfigLoader:
    """Test cases for ConfigLoader"""

    def test_load_yaml_config(self):
        """Test loading YAML configuration"""
        config = ConfigLoader.load_yaml_config("configs/models/gpt-oss-120b.yaml")

        assert config.model_identity.name == "gpt-oss-120b"
        assert config.model_identity.num_layers == 36
        assert config.architecture_config.hidden_size == 2880
        assert config.architecture_config.attention_type == "gqa"

    def test_load_model_config_with_modules(self):
        """Test loading model config with modules"""
        config = ConfigLoader.load_yaml_config("configs/models/gpt-oss-120b.yaml")

        # Check modules exist
        assert "embedding" in config.modules
        assert "attention" in config.modules
        assert "ffn_moe" in config.modules
        assert "norm" in config.modules

    def test_load_model_config_with_computation_rules(self):
        """Test loading model config with computation rules"""
        config = ConfigLoader.load_yaml_config("configs/models/gpt-oss-120b.yaml")

        assert "kv_cache" in config.computation_rules
        assert "activation" in config.computation_rules
        assert "recommended_capacity_factor" in config.computation_rules
        assert config.computation_rules["recommended_capacity_factor"] == 1.25

    def test_weight_info_parsing(self):
        """Test weight info is correctly parsed"""
        config = ConfigLoader.load_yaml_config("configs/models/gpt-oss-120b.yaml")

        # Check embedding weight
        embed_weights = config.modules.get("embedding", {})
        embed_weight = embed_weights.get("model.embed_tokens.weight")
        assert embed_weight is not None
        assert embed_weight.shape == [201088, 2880]
        assert embed_weight.dtype == "BF16"
        assert embed_weight.layers == 1
        assert embed_weight.parallel_strategy == "replicated"

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

    def test_validate_config_complete(self):
        """Test validation of complete config"""
        config = ConfigLoader.load_yaml_config("configs/models/gpt-oss-120b.yaml")

        # Should not raise any exception for valid config
        assert config.model_identity.name is not None
        assert config.architecture_config.hidden_size > 0
        assert len(config.modules) > 0


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
