#!/usr/bin/env python3
"""
Tests for configuration files validation
"""

import pytest
from pathlib import Path

from llm_mem_estimator import ConfigLoader
from llm_mem_estimator.memory_estimator import MemoryEstimator


class TestModelConfigs:
    """Test cases for validating model configuration files"""

    def test_all_model_configs_exist(self):
        """Test that model config directory exists"""
        config_dir = Path("configs/models")
        assert config_dir.exists(), "configs/models directory should exist"

    def test_gpt_oss_120b_config_valid(self):
        """Test gpt-oss-120b configuration is valid"""
        config = ConfigLoader.load_yaml_config("configs/models/gpt-oss-120b.yaml")

        # Verify required fields
        assert config.model_identity.name is not None
        assert config.model_identity.num_layers > 0
        assert config.architecture_config.hidden_size > 0

        # Verify modules exist
        assert len(config.modules) > 0

        # Verify computation rules exist
        assert "kv_cache" in config.computation_rules
        assert "activation" in config.computation_rules

    def test_config_has_required_sections(self):
        """Test that config has all required sections"""
        config = ConfigLoader.load_yaml_config("configs/models/gpt-oss-120b.yaml")

        # Check model_identity
        assert hasattr(config, 'model_identity')
        assert config.model_identity.name == "gpt-oss-120b"

        # Check architecture_config
        assert hasattr(config, 'architecture_config')
        assert config.architecture_config.hidden_size == 2880
        assert config.architecture_config.num_layers == 36
        assert config.architecture_config.attention_type == "gqa"

        # Check modules
        assert hasattr(config, 'modules')
        assert isinstance(config.modules, dict)

        # Check computation_rules
        assert hasattr(config, 'computation_rules')
        assert isinstance(config.computation_rules, dict)


class TestWeightMappingRules:
    """Test cases for weight mapping rules configuration"""

    def test_weight_mapping_rules_exist(self):
        """Test that weight mapping rules file exists"""
        rules_path = Path("configs/weight_mapping_rules.yaml")
        assert rules_path.exists()

    def test_weight_mapping_rules_valid(self):
        """Test that weight mapping rules are valid"""
        rules = ConfigLoader.load_weight_mapping_rules("configs/weight_mapping_rules.yaml")

        # Should have generic rules
        assert "generic" in rules
        assert "embedding" in rules["generic"]
        assert "attention" in rules["generic"]
        assert "ffn_moe" in rules["generic"]
        assert "norm" in rules["generic"]

    def test_weight_mapping_has_patterns(self):
        """Test that weight mapping rules have patterns"""
        rules = ConfigLoader.load_weight_mapping_rules("configs/weight_mapping_rules.yaml")

        # Generic embedding should have patterns
        assert "patterns" in rules["generic"]["embedding"]
        assert len(rules["generic"]["embedding"]["patterns"]) > 0


class TestChipsConfig:
    """Test cases for chips configuration"""

    def test_chips_config_exist(self):
        """Test that chips config file exists"""
        chips_path = Path("configs/chips.json")
        assert chips_path.exists()

    def test_chips_config_valid(self):
        """Test that chips configuration is valid"""
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


class TestEstimatorWithConfig:
    """Test that estimator works with configuration files"""

    def test_estimator_works_with_gpt_oss_config(self):
        """Test that estimator can work with gpt-oss-120b config"""
        config = ConfigLoader.load_yaml_config("configs/models/gpt-oss-120b.yaml")
        estimator = MemoryEstimator(config)

        # Should be able to calculate weights memory
        total_memory, breakdown = estimator.calculate_weights_memory()
        assert total_memory > 0

    def test_estimator_computation_rules_work(self):
        """Test that computation rules can be evaluated"""
        config = ConfigLoader.load_yaml_config("configs/models/gpt-oss-120b.yaml")
        estimator = MemoryEstimator(config)

        # Should be able to calculate KV cache
        kv_memory = estimator.calculate_kv_cache_memory(
            batch_size=1,
            seq_len=2048,
            dtype="fp16"
        )
        assert kv_memory > 0

        # Should be able to calculate activation
        act_memory = estimator.calculate_activation_memory(
            batch_size=1,
            seq_len=2048,
            dtype="fp16"
        )
        assert act_memory > 0


class TestConfigFilesListing:
    """List all available config files"""

    def test_list_all_model_configs(self):
        """List all model configuration files"""
        config_dir = Path("configs/models")
        if config_dir.exists():
            yaml_files = list(config_dir.glob("*.yaml"))
            # Should have at least one config
            assert len(yaml_files) >= 1

            # Print available configs for information
            for f in yaml_files:
                print(f"Available config: {f.name}")
