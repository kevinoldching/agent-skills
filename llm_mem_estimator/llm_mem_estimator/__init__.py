#!/usr/bin/env python3
"""
LLM Memory Estimator - Calculate GPU memory usage for large language models
"""

from .model_config import (
    WeightInfo,
    ModelIdentity,
    ArchitectureConfig,
    ModelConfig,
    MemoryResult,
    get_dtype_bytes,
    calculate_weight_memory,
)
from .config_loader import ConfigLoader
from .formula_evaluator import FormulaEvaluator
from .memory_estimator import MemoryEstimator
from .report_generator import ReportGenerator
from .weight_classifier import WeightClassifier
from .model_detector import ModelDetector, ConfigGenerator
from .exceptions import (
    ConfigError,
    FormulaError,
    ParallelConfigError,
    ChipConfigError,
    ModelDetectionError,
)

__version__ = "0.1.0"

__all__ = [
    # Data structures
    "WeightInfo",
    "ModelIdentity",
    "ArchitectureConfig",
    "ModelConfig",
    "MemoryResult",
    # Utility functions
    "get_dtype_bytes",
    "calculate_weight_memory",
    # Core classes
    "ConfigLoader",
    "FormulaEvaluator",
    "MemoryEstimator",
    "ReportGenerator",
    "WeightClassifier",
    "ModelDetector",
    "ConfigGenerator",
    # Exceptions
    "ConfigError",
    "FormulaError",
    "ParallelConfigError",
    "ChipConfigError",
    "ModelDetectionError",
]
