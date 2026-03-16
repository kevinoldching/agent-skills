#!/usr/bin/env python3
"""
Pytest configuration and utilities for tests
"""

import pytest
import yaml
from pathlib import Path
from typing import List, Dict


# ============================================================================
# Configuration Loading
# ============================================================================

def get_test_models_config() -> Dict:
    """加载 test_models.yaml 配置"""
    config_path = Path(__file__).parent / "test_models.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_test_model_names() -> List[str]:
    """获取测试模型名称列表"""
    config = get_test_models_config()
    return [m["name"] for m in config.get("models", [])]


def get_test_models_by_type(model_type: str) -> List[Dict]:
    """按类型获取测试模型列表

    Args:
        model_type: "local" | "huggingface"

    Returns:
        模型配置列表
    """
    config = get_test_models_config()
    return [m for m in config.get("models", []) if m.get("type") == model_type]


# ============================================================================
# Outputs Directory Utilities
# ============================================================================

def get_outputs_dir() -> Path:
    """获取 outputs 目录路径"""
    return Path(__file__).parent.parent / "outputs"


def get_outputs_configs() -> List[Path]:
    """获取 outputs/ 目录下的所有模型配置文件"""
    outputs_dir = get_outputs_dir()
    if not outputs_dir.exists():
        return []
    return sorted(outputs_dir.glob("*_config.yaml"))


def get_outputs_weights_metadata() -> List[Path]:
    """获取 outputs/ 目录下的所有权重元数据文件"""
    outputs_dir = get_outputs_dir()
    if not outputs_dir.exists():
        return []
    return sorted(outputs_dir.glob("*_weights_metadata.md"))


# ============================================================================
# Pytest Hooks
# ============================================================================

def pytest_configure(config):
    """Pytest 配置钩子"""
    # 检查 outputs 目录
    outputs_dir = get_outputs_dir()
    if not outputs_dir.exists():
        pytest.skip("outputs/ directory does not exist. Run test_model_detection.py first.")

    configs = get_outputs_configs()
    if not configs:
        pytest.skip("No model configs found in outputs/. Run test_model_detection.py first.")


def pytest_collection_modifyitems(config, items):
    """修改测试收集，添加自定义 marker"""
    for item in items:
        # 为需要 outputs 的测试添加 marker
        if "test_model_config" in item.nodeid or \
           "test_memory_estimator" in item.nodeid or \
           "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.requires_outputs)
