#!/usr/bin/env python3
"""
Tests for MemoryEstimator

Tests are based on outputs/ directory which contains generated model configs.
Run test_model_detection.py first to generate the configs.
"""

import pytest
from pathlib import Path

from llm_mem_estimator import ConfigLoader
from llm_mem_estimator.memory_estimator import MemoryEstimator
from llm_mem_estimator.model_config import get_dtype_bytes
from tests.conftest import get_outputs_configs


# ============================================================================
# Parameterized Tests based on outputs/
# ============================================================================

class TestMemoryEstimatorFromOutputs:
    """基于 outputs/ 目录的显存估算测试（参数化）"""

    @pytest.fixture(autouse=True)
    def check_outputs(self):
        """检查 outputs 目录"""
        configs = get_outputs_configs()
        if not configs:
            pytest.skip("No configs in outputs/. Run test_model_detection.py first.")

    @pytest.mark.parametrize("config_path", get_outputs_configs(), ids=lambda p: p.stem.replace("_config", ""))
    def test_calculate_weights_memory(self, config_path):
        """测试计算权重显存"""
        config = ConfigLoader.load_yaml_config(str(config_path))
        estimator = MemoryEstimator(config)

        total_memory, breakdown = estimator.calculate_weights_memory()

        assert total_memory > 0
        assert isinstance(breakdown, dict)
        assert len(breakdown) > 0

    @pytest.mark.parametrize("config_path", get_outputs_configs(), ids=lambda p: p.stem.replace("_config", ""))
    def test_calculate_kv_cache_memory(self, config_path):
        """测试计算 KV Cache 显存"""
        config = ConfigLoader.load_yaml_config(str(config_path))
        estimator = MemoryEstimator(config)

        kv_memory = estimator.calculate_kv_cache_memory(
            batch_size=1,
            seq_len=2048,
            dtype="fp16"
        )

        assert kv_memory > 0

    @pytest.mark.parametrize("config_path", get_outputs_configs(), ids=lambda p: p.stem.replace("_config", ""))
    def test_calculate_activation_memory(self, config_path):
        """测试计算激活值显存"""
        config = ConfigLoader.load_yaml_config(str(config_path))
        estimator = MemoryEstimator(config)

        act_memory = estimator.calculate_activation_memory(
            batch_size=1,
            seq_len=2048,
            dtype="fp16"
        )

        assert act_memory > 0

    @pytest.mark.parametrize("config_path", get_outputs_configs(), ids=lambda p: p.stem.replace("_config", ""))
    def test_estimate_memory_full(self, config_path):
        """测试完整显存估算"""
        config = ConfigLoader.load_yaml_config(str(config_path))
        estimator = MemoryEstimator(config)

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

    @pytest.mark.parametrize("config_path", get_outputs_configs(), ids=lambda p: p.stem.replace("_config", ""))
    def test_find_max_sequence_length(self, config_path):
        """测试估算最大序列长度"""
        config = ConfigLoader.load_yaml_config(str(config_path))
        estimator = MemoryEstimator(config)

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

        # 大模型（如 DeepSeek-V3 600B+）可能权重本身超过 200GB
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
