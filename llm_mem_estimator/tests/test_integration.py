#!/usr/bin/env python3
"""
Integration tests for LLM Memory Estimator

Tests are based on outputs/ directory which contains generated model configs.
Run test_model_detection.py first to generate the configs.
"""

import pytest
from pathlib import Path

from llm_mem_estimator import ConfigLoader
from llm_mem_estimator.memory_estimator import MemoryEstimator
from llm_mem_estimator.report_generator import ReportGenerator
from tests.conftest import get_outputs_configs


# ============================================================================
# Parameterized Tests based on outputs/
# ============================================================================

class TestEndToEndFromOutputs:
    """基于 outputs/ 目录的端到端测试（参数化）"""

    @pytest.fixture(autouse=True)
    def check_outputs(self):
        """检查 outputs 目录"""
        configs = get_outputs_configs()
        if not configs:
            pytest.skip("No configs in outputs/. Run test_model_detection.py first.")

    @pytest.mark.parametrize("config_path", get_outputs_configs(), ids=lambda p: p.stem.replace("_config", ""))
    def test_end_to_end(self, config_path):
        """端到端测试 - 从配置加载到报告生成"""
        # 1. 加载配置
        config = ConfigLoader.load_yaml_config(str(config_path))

        # 2. 创建估算器
        estimator = MemoryEstimator(config)

        # 3. 估算显存
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

        # 4. 验证结果
        assert result.total_memory_gb > 0
        assert result.weights_memory_gb > 0
        assert result.kv_cache_memory_gb > 0
        assert result.activation_memory_gb > 0

        # 5. 生成报告
        report = ReportGenerator.generate_report(
            config=config,
            result=result,
            batch_size=1,
            seq_len=2048,
            parallel_config={'tp': 1, 'pp': 1, 'dp': 1, 'cp': 1}
        )

        # 6. 验证报告内容
        assert config.model_identity.name in report
        assert "Weights" in report or "权重" in report
        assert "Total" in report or "总计" in report

    @pytest.mark.parametrize("config_path", get_outputs_configs(), ids=lambda p: p.stem.replace("_config", ""))
    def test_end_to_end_with_parallel(self, config_path):
        """测试并行策略"""
        config = ConfigLoader.load_yaml_config(str(config_path))
        estimator = MemoryEstimator(config)

        # With TP=8
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

    @pytest.mark.parametrize("config_path", get_outputs_configs(), ids=lambda p: p.stem.replace("_config", ""))
    def test_different_batch_sizes(self, config_path):
        """测试不同 batch size"""
        config = ConfigLoader.load_yaml_config(str(config_path))
        estimator = MemoryEstimator(config)
        seq_len = 2048

        result_bs1 = estimator.estimate_memory(batch_size=1, seq_len=seq_len)
        result_bs2 = estimator.estimate_memory(batch_size=2, seq_len=seq_len)

        # Memory should increase with batch size
        assert result_bs2.total_memory_gb > result_bs1.total_memory_gb

    @pytest.mark.parametrize("config_path", get_outputs_configs(), ids=lambda p: p.stem.replace("_config", ""))
    def test_different_sequence_lengths(self, config_path):
        """测试不同序列长度"""
        config = ConfigLoader.load_yaml_config(str(config_path))
        estimator = MemoryEstimator(config)
        batch_size = 1

        result_1k = estimator.estimate_memory(batch_size=batch_size, seq_len=1024)
        result_2k = estimator.estimate_memory(batch_size=batch_size, seq_len=2048)

        # Memory should increase with sequence length
        assert result_2k.total_memory_gb > result_1k.total_memory_gb
