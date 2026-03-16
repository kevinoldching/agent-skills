#!/usr/bin/env python3
"""
Tests for Model Detector and Weight Classifier
Combines test_weight_classifier.py
"""

import pytest
from collections import defaultdict

from llm_mem_estimator import ConfigLoader, WeightClassifier, ModelDetector


class TestWeightClassifier:
    """Test cases for WeightClassifier"""

    @pytest.fixture
    def rules(self):
        """Load weight mapping rules"""
        return ConfigLoader.load_weight_mapping_rules("configs/weight_mapping_rules.yaml")

    @pytest.fixture
    def classifier(self, rules):
        """Create WeightClassifier instance"""
        return WeightClassifier(rules)

    def test_classify_embedding_weights(self, classifier):
        """Test classifying embedding weights"""
        # Test embed_tokens.weight
        result = classifier.classify_weight("embed_tokens.weight")
        assert result == "embedding"

        # Test model.embed_tokens.weight
        result = classifier.classify_weight("model.embed_tokens.weight")
        assert result == "embedding"

        # Test lm_head.weight
        result = classifier.classify_weight("lm_head.weight")
        assert result == "embedding"

    def test_classify_attention_weights(self, classifier):
        """Test classifying attention weights"""
        # Test self_attn.q_proj.weight
        result = classifier.classify_weight("self_attn.q_proj.weight")
        assert result == "attention"

        # Test model.layers.0.self_attn.k_proj.weight
        result = classifier.classify_weight("model.layers.0.self_attn.k_proj.weight")
        assert result == "attention"

        # Test model.layers.5.self_attn.o_proj.weight
        result = classifier.classify_weight("model.layers.5.self_attn.o_proj.weight")
        assert result == "attention"

    def test_classify_ffn_moe_weights(self, classifier):
        """Test classifying FFN MoE weights"""
        # Test mlp.router.weight
        result = classifier.classify_weight("mlp.router.weight")
        assert result == "ffn_moe"

        # Test mlp.gate.weight
        result = classifier.classify_weight("mlp.gate.weight")
        assert result == "ffn_moe"

        # Test mlp.experts.0.gate_proj.weight
        result = classifier.classify_weight("mlp.experts.0.gate_proj.weight")
        assert result == "ffn_moe"

        # Test mlp.experts.5.down_proj.weight
        result = classifier.classify_weight("mlp.experts.5.down_proj.weight")
        assert result == "ffn_moe"

        # Test mlp.experts.gate_up_proj_blocks
        result = classifier.classify_weight("mlp.experts.gate_up_proj_blocks")
        assert result == "ffn_moe"

    def test_classify_norm_weights(self, classifier):
        """Test classifying normalization weights"""
        # Test input_layernorm.weight
        result = classifier.classify_weight("input_layernorm.weight")
        assert result == "norm"

        # Test post_attention_layernorm.weight
        result = classifier.classify_weight("post_attention_layernorm.weight")
        assert result == "norm"

        # Test model.norm.weight
        result = classifier.classify_weight("model.norm.weight")
        assert result == "norm"

    def test_classify_ffn_dense_weights(self, classifier):
        """Test classifying FFN dense weights"""
        # Test mlp.gate_proj.weight
        result = classifier.classify_weight("mlp.gate_proj.weight")
        assert result == "ffn_dense"

        # Test mlp.up_proj.weight
        result = classifier.classify_weight("mlp.up_proj.weight")
        assert result == "ffn_dense"

        # Test mlp.down_proj.weight
        result = classifier.classify_weight("mlp.down_proj.weight")
        assert result == "ffn_dense"

    def test_classify_with_model_type(self, classifier):
        """Test classification with specific model type"""
        # Test with gpt_oss model type
        result = classifier.classify_weight("mlp.router.weight", model_type="gpt_oss")
        assert result == "ffn_moe"

        result = classifier.classify_weight("mlp.experts.gate_up_proj_blocks", model_type="gpt_oss")
        assert result == "ffn_moe"

    def test_classify_unknown_weight(self, classifier):
        """Test classifying unknown weights"""
        # Test unknown weight should return 'others'
        result = classifier.classify_weight("some.unknown.weight")
        assert result == "others"

    def test_classify_deepseek_weights(self, rules):
        """Test DeepSeek-specific weight classification"""
        classifier = WeightClassifier(rules)

        # Test DeepSeek attention weights
        result = classifier.classify_weight("model.layers.0.self_attn.q_a_proj.weight")
        assert result == "attention"

        # Test DeepSeek MoE expert weights
        result = classifier.classify_weight("model.layers.10.mlp.experts.5.gate_proj.weight")
        assert result == "ffn_moe"

    def test_classify_llama_weights(self, rules):
        """Test Llama-specific weight classification"""
        classifier = WeightClassifier(rules)

        # Test Llama attention weights
        result = classifier.classify_weight("model.layers.0.self_attn.q_proj.weight")
        assert result == "attention"

        # Test Llama FFN weights
        result = classifier.classify_weight("model.layers.5.mlp.gate_proj.weight")
        assert result == "ffn_dense"


class TestWeightClassifierInheritance:
    """Test cases for weight classifier rule inheritance"""

    @pytest.fixture
    def rules(self):
        """Load weight mapping rules"""
        return ConfigLoader.load_weight_mapping_rules("configs/weight_mapping_rules.yaml")

    def test_gpt_oss_inherits_from_generic(self, rules):
        """Test that gpt_oss inherits from generic"""
        classifier = WeightClassifier(rules)

        # gpt_oss should have access to generic rules through inheritance
        # It should still classify embedding weights correctly
        result = classifier.classify_weight("embed_tokens.weight", model_type="gpt_oss")
        assert result == "embedding"

    def test_llama_inherits_from_generic(self, rules):
        """Test that llama inherits from generic"""
        classifier = WeightClassifier(rules)

        # llama inherits from generic, so it should work
        result = classifier.classify_weight("model.embed_tokens.weight", model_type="llama")
        assert result == "embedding"

    def test_override_inherited_rules(self, rules):
        """Test that model-specific rules override inherited rules"""
        classifier = WeightClassifier(rules)

        # gpt_oss has specific ffn_moe rules that should override generic
        result = classifier.classify_weight("mlp.router.weight", model_type="gpt_oss")
        assert result == "ffn_moe"


class TestModelDetector:
    """Test cases for ModelDetector"""

    def test_detect_from_huggingface_with_local_config(self):
        """Test detecting model config from local config.json"""
        # Use local model for testing
        config = ModelDetector.detect_from_huggingface("Qwen/Qwen2.5-0.5B")

        # Verify key fields exist
        assert config.get("hidden_size") is not None
        assert config.get("num_hidden_layers") is not None
        assert config.get("model_type") is not None
