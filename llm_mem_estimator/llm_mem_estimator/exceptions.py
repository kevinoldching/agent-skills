#!/usr/bin/env python3
"""
Custom exceptions for LLM Memory Estimator
"""


class ConfigError(Exception):
    """Configuration error"""
    pass


class FormulaError(Exception):
    """Formula evaluation error"""
    pass


class ParallelConfigError(Exception):
    """Parallel configuration error"""
    pass


class ChipConfigError(Exception):
    """Chip configuration error"""
    pass


class ModelDetectionError(Exception):
    """Model detection error"""
    pass
