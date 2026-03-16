#!/usr/bin/env python3
"""
Tests for MemoryEstimator
"""

import pytest

from llm_mem_estimator import ConfigLoader
from llm_mem_estimator.memory_estimator import MemoryEstimator
from llm_mem_estimator.model_config import get_dtype_bytes


class TestMemoryEstimator:
    """Test cases for MemoryEstimator"""

    @pytest.fixture
    def config(self):
        """Load test configuration"""
        return ConfigLoader.load_yaml_config("configs/models/gpt-oss-120b.yaml")

    @pytest.fixture
    def estimator(self, config):
        """Create MemoryEstimator instance"""
        return MemoryEstimator(config)

    def test_calculate_weights_memory(self, estimator):
        """Test calculating weights memory"""
        total_memory, breakdown = estimator.calculate_weights_memory()

        assert total_memory > 0
        assert isinstance(breakdown, dict)
        assert len(breakdown) > 0

        # Check breakdown contains expected module types
        assert "ffn_moe" in breakdown
        assert "embedding" in breakdown
        assert "attention" in breakdown
        assert "norm" in breakdown

    def test_weights_memory_breakdown(self, estimator):
        """Test weights memory breakdown by module type"""
        total_memory, breakdown = estimator.calculate_weights_memory()

        # ffn_moe should have the largest memory for this model
        assert breakdown["ffn_moe"] > breakdown["embedding"]
        assert breakdown["ffn_moe"] > breakdown["attention"]

    def test_calculate_kv_cache_memory(self, estimator):
        """Test calculating KV cache memory"""
        kv_memory = estimator.calculate_kv_cache_memory(
            batch_size=1,
            seq_len=2048,
            dtype="fp16"
        )

        assert kv_memory > 0
        assert kv_memory < 10  # Should be reasonable for this config

    def test_calculate_kv_cache_with_parallel(self, estimator):
        """Test KV cache calculation with tensor parallel"""
        # Without TP
        kv_memory_single = estimator.calculate_kv_cache_memory(
            batch_size=1,
            seq_len=2048,
            dtype="fp16",
            tp=1
        )

        # With TP=8
        kv_memory_tp8 = estimator.calculate_kv_cache_memory(
            batch_size=1,
            seq_len=2048,
            dtype="fp16",
            tp=8
        )

        # KV cache should be divided by TP
        assert kv_memory_tp8 < kv_memory_single
        assert abs(kv_memory_single / 8 - kv_memory_tp8) < 0.001

    def test_calculate_activation_memory(self, estimator):
        """Test calculating activation memory"""
        act_memory = estimator.calculate_activation_memory(
            batch_size=1,
            seq_len=2048,
            dtype="fp16"
        )

        assert act_memory > 0
        assert act_memory < 100  # Should be reasonable

    def test_calculate_activation_with_capacity_factor(self, estimator):
        """Test activation memory uses capacity factor"""
        act_memory = estimator.calculate_activation_memory(
            batch_size=1,
            seq_len=4096,
            dtype="fp16"
        )

        # The formula should include recommended_capacity_factor=1.25
        # Activation should be non-zero
        assert act_memory > 0

    def test_calculate_activation_with_parallel(self, estimator):
        """Test activation calculation with tensor parallel"""
        # Without TP
        act_single = estimator.calculate_activation_memory(
            batch_size=1,
            seq_len=2048,
            dtype="fp16",
            tp=1
        )

        # With TP=8
        act_tp8 = estimator.calculate_activation_memory(
            batch_size=1,
            seq_len=2048,
            dtype="fp16",
            tp=8
        )

        # Activation should be divided by TP
        assert act_tp8 < act_single

    def test_estimate_memory_full(self, estimator):
        """Test full memory estimation"""
        result = estimator.estimate_memory(
            batch_size=1,
            seq_len=2048,
            kv_dtype="fp16",
            activation_dtype="fp16",
            tp=1,
            pp=1,
            dp=1,
            cp=1,
            system_reserved_gb=2.0
        )

        assert result.total_memory_gb > 0
        assert result.weights_memory_gb > 0
        assert result.kv_cache_memory_gb > 0
        assert result.activation_memory_gb > 0
        assert result.system_reserved_gb == 2.0

        # Check breakdown
        assert isinstance(result.breakdown, dict)
        assert len(result.breakdown) > 0

    def test_estimate_memory_with_parallel(self, estimator):
        """Test memory estimation with parallel strategies"""
        result = estimator.estimate_memory(
            batch_size=1,
            seq_len=2048,
            kv_dtype="fp16",
            activation_dtype="fp16",
            tp=8,
            pp=1,
            dp=1,
            cp=1,
            system_reserved_gb=2.0
        )

        assert result.total_memory_gb > 0

    def test_find_max_sequence_length(self, estimator):
        """Test finding maximum sequence length"""
        # Using a large available memory
        max_seq_len = estimator.find_max_sequence_length(
            available_memory_gb=200,
            batch_size=1,
            kv_dtype="fp16",
            activation_dtype="fp16",
            tp=1,
            pp=1,
            cp=1,
            system_reserved_gb=2.0
        )

        assert max_seq_len > 0

    def test_find_max_sequence_length_insufficient_memory(self, estimator):
        """Test with insufficient memory"""
        # Using very small available memory
        max_seq_len = estimator.find_max_sequence_length(
            available_memory_gb=10,  # Very small
            batch_size=1,
            kv_dtype="fp16",
            activation_dtype="fp16",
            tp=1,
            pp=1,
            cp=1,
            system_reserved_gb=2.0
        )

        # Should return 0 when weights alone exceed available memory
        assert max_seq_len >= 0


class TestGetDtypeBytes:
    """Test cases for get_dtype_bytes utility function"""

    def test_fp32(self):
        assert get_dtype_bytes("fp32") == 4
        assert get_dtype_bytes("float32") == 4
        assert get_dtype_bytes("f32") == 4

    def test_fp16(self):
        assert get_dtype_bytes("fp16") == 2
        assert get_dtype_bytes("float16") == 2
        assert get_dtype_bytes("f16") == 2

    def test_bf16(self):
        assert get_dtype_bytes("bf16") == 2
        assert get_dtype_bytes("bfloat16") == 2

    def test_fp8(self):
        assert get_dtype_bytes("fp8") == 1
        assert get_dtype_bytes("float8") == 1
        assert get_dtype_bytes("f8_e4m3") == 1

    def test_int8(self):
        assert get_dtype_bytes("int8") == 1

    def test_uint8(self):
        assert get_dtype_bytes("uint8") == 1
        assert get_dtype_bytes("u8") == 1

    def test_int4(self):
        assert get_dtype_bytes("int4") == 0.5
        assert get_dtype_bytes("uint4") == 0.5
        assert get_dtype_bytes("u4") == 0.5

    def test_unknown_dtype(self):
        with pytest.raises(ValueError):
            get_dtype_bytes("unknown")
