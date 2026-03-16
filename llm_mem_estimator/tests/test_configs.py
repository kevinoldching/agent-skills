#!/usr/bin/env python3
"""
Tests for configuration files validation
"""

import pytest
from pathlib import Path

from llm_mem_estimator import ConfigLoader


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
