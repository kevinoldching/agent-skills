#!/usr/bin/env python3
"""
Configuration loader for LLM Memory Estimator
"""

import json
import yaml
from typing import Dict, Any

from .model_config import (
    ModelConfig, ModelIdentity, ArchitectureConfig, WeightInfo
)


class ConfigLoader:
    """Load and validate configuration files"""

    @staticmethod
    def load_yaml_config(config_path: str) -> ModelConfig:
        """Load model configuration from YAML file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Parse model identity
        identity_data = data.get('model_identity', {})
        model_identity = ModelIdentity(
            name=identity_data.get('name', 'unknown'),
            total_params=identity_data.get('total_params', 'unknown'),
            num_layers=identity_data.get('num_layers', 0),
            quantization=identity_data.get('quantization')
        )

        # Parse architecture config
        arch_data = data.get('architecture_config', {})
        architecture_config = ArchitectureConfig(
            hidden_size=arch_data.get('hidden_size', 0),
            num_layers=arch_data.get('num_layers', 0),
            attention_type=arch_data.get('attention_type', 'unknown'),
            ffn_type=arch_data.get('ffn_type', 'unknown'),
            norm_type=arch_data.get('norm_type', 'unknown'),
            vocab_size=arch_data.get('vocab_size', 0),
            num_attention_heads=arch_data.get('num_attention_heads'),
            num_key_value_heads=arch_data.get('num_key_value_heads'),
            intermediate_size=arch_data.get('intermediate_size'),
            num_experts=arch_data.get('num_experts'),
            num_experts_per_tok=arch_data.get('num_experts_per_tok'),
            moe_intermediate_size=arch_data.get('moe_intermediate_size'),
            q_lora_rank=arch_data.get('q_lora_rank'),
            kv_lora_rank=arch_data.get('kv_lora_rank'),
            qk_rope_head_dim=arch_data.get('qk_rope_head_dim'),
            v_head_dim=arch_data.get('v_head_dim'),
            qk_nope_head_dim=arch_data.get('qk_nope_head_dim')
        )

        # Parse modules
        modules = {}
        modules_data = data.get('modules', {})
        for module_type, weights_data in modules_data.items():
            weights = {}
            for weight_name, weight_data in weights_data.items():
                weights[weight_name] = WeightInfo(
                    shape=weight_data.get('shape', []),
                    dtype=weight_data.get('dtype', 'fp16'),
                    layers=weight_data.get('layers', 1),
                    parallel_strategy=weight_data.get('parallel_strategy', 'replicated'),
                    world_size=weight_data.get('world_size', 0)
                )
            modules[module_type] = weights

        # Parse computation rules
        computation_rules = data.get('computation_rules', {})

        return ModelConfig(
            model_identity=model_identity,
            architecture_config=architecture_config,
            modules=modules,
            computation_rules=computation_rules
        )

    @staticmethod
    def load_chips_config(chips_path: str) -> Dict[str, Any]:
        """Load hardware chips configuration"""
        with open(chips_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def load_weight_mapping_rules(rules_path: str) -> Dict[str, Any]:
        """Load weight mapping rules"""
        with open(rules_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
