#!/usr/bin/env python3
"""
Integration tests for LLM Memory Estimator
"""

import pytest

from llm_mem_estimator import ConfigLoader
from llm_mem_estimator.memory_estimator import MemoryEstimator
from llm_mem_estimator.report_generator import ReportGenerator


class TestEndToEnd:
    """End-to-end integration tests"""

    def test_end_to_end_gpt_oss_120b(self):
        """Test end-to-end estimation for gpt-oss-120b"""
        # 1. Load configuration
        config = ConfigLoader.load_yaml_config("configs/models/gpt-oss-120b.yaml")

        # 2. Create estimator
        estimator = MemoryEstimator(config)

        # 3. Estimate memory
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

        # 4. Verify results
        assert result.total_memory_gb > 0
        assert result.weights_memory_gb > 0
        assert result.kv_cache_memory_gb > 0
        assert result.activation_memory_gb > 0

        # 5. Generate report
        report = ReportGenerator.generate_report(
            config=config,
            result=result,
            batch_size=1,
            seq_len=2048,
            parallel_config={'tp': 1, 'pp': 1, 'dp': 1, 'cp': 1}
        )

        # 6. Verify report content
        assert "gpt-oss-120b" in report
        assert "Weights" in report
        assert "KV Cache" in report
        assert "Activation" in report
        assert "Total" in report

    def test_end_to_end_with_parallel(self):
        """Test end-to-end estimation with parallel strategies"""
        # Load configuration
        config = ConfigLoader.load_yaml_config("configs/models/gpt-oss-120b.yaml")

        # Create estimator
        estimator = MemoryEstimator(config)

        # Estimate with TP=8
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

        # Verify total memory is reasonable
        assert result.total_memory_gb > 0

    def test_find_max_seq_len_integration(self):
        """Test finding maximum sequence length"""
        # Load configuration
        config = ConfigLoader.load_yaml_config("configs/models/gpt-oss-120b.yaml")

        # Create estimator
        estimator = MemoryEstimator(config)

        # Find max sequence length with 80GB memory
        max_seq_len = estimator.find_max_sequence_length(
            available_memory_gb=80,
            batch_size=1,
            kv_dtype="fp16",
            activation_dtype="fp16",
            tp=1,
            pp=1,
            cp=1,
            system_reserved_gb=2.0
        )

        assert max_seq_len > 0

    def test_memory_breakdown_accuracy(self):
        """Test that memory breakdown sums to total"""
        # Load configuration
        config = ConfigLoader.load_yaml_config("configs/models/gpt-oss-120b.yaml")

        # Create estimator
        estimator = MemoryEstimator(config)

        # Calculate weights memory
        weights_memory, breakdown = estimator.calculate_weights_memory()

        # Sum of breakdown should equal total
        breakdown_sum = sum(breakdown.values())
        assert abs(breakdown_sum - weights_memory) < 0.001


class TestConfigToReport:
    """Test configuration loading to report generation flow"""

    def test_complete_flow(self):
        """Test complete flow from config to report"""
        # Load config
        config = ConfigLoader.load_yaml_config("configs/models/gpt-oss-120b.yaml")

        # Verify config
        assert config.model_identity.name == "gpt-oss-120b"
        assert config.architecture_config.hidden_size == 2880
        assert "ffn_moe" in config.modules

        # Create estimator and calculate
        estimator = MemoryEstimator(config)
        result = estimator.estimate_memory(
            batch_size=1,
            seq_len=4096,
            kv_dtype="fp16",
            activation_dtype="fp16"
        )

        # Generate report
        report = ReportGenerator.generate_report(
            config=config,
            result=result,
            batch_size=1,
            seq_len=4096,
            parallel_config={'tp': 1, 'pp': 1, 'dp': 1, 'cp': 1}
        )

        # Verify report contains key information
        assert "gpt-oss-120b" in report
        assert "2048" not in report  # Should be 4096
        assert "4096" in report

    def test_with_custom_system_reserved(self):
        """Test with custom system reserved memory"""
        config = ConfigLoader.load_yaml_config("configs/models/gpt-oss-120b.yaml")
        estimator = MemoryEstimator(config)

        result = estimator.estimate_memory(
            batch_size=1,
            seq_len=2048,
            system_reserved_gb=5.0
        )

        assert result.system_reserved_gb == 5.0
        assert result.total_memory_gb > result.weights_memory_gb + result.kv_cache_memory_gb + result.activation_memory_gb


class TestMultipleBatchSizes:
    """Test with multiple batch sizes"""

    @pytest.fixture
    def config(self):
        return ConfigLoader.load_yaml_config("configs/models/gpt-oss-120b.yaml")

    @pytest.fixture
    def estimator(self, config):
        return MemoryEstimator(config)

    def test_different_batch_sizes(self, estimator):
        """Test memory estimation with different batch sizes"""
        seq_len = 2048

        result_bs1 = estimator.estimate_memory(batch_size=1, seq_len=seq_len)
        result_bs2 = estimator.estimate_memory(batch_size=2, seq_len=seq_len)
        result_bs4 = estimator.estimate_memory(batch_size=4, seq_len=seq_len)

        # Memory should increase with batch size
        assert result_bs2.total_memory_gb > result_bs1.total_memory_gb
        assert result_bs4.total_memory_gb > result_bs2.total_memory_gb

        # KV cache should scale linearly with batch size
        assert result_bs2.kv_cache_memory_gb > result_bs1.kv_cache_memory_gb

    def test_different_sequence_lengths(self, estimator):
        """Test memory estimation with different sequence lengths"""
        batch_size = 1

        result_1k = estimator.estimate_memory(batch_size=batch_size, seq_len=1024)
        result_2k = estimator.estimate_memory(batch_size=batch_size, seq_len=2048)
        result_4k = estimator.estimate_memory(batch_size=batch_size, seq_len=4096)

        # Memory should increase with sequence length
        assert result_2k.total_memory_gb > result_1k.total_memory_gb
        assert result_4k.total_memory_gb > result_2k.total_memory_gb

        # KV cache should scale linearly with sequence length
        assert result_2k.kv_cache_memory_gb > result_1k.kv_cache_memory_gb
