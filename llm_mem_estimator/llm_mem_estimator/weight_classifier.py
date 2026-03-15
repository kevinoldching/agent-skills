#!/usr/bin/env python3
"""
Weight classifier for LLM Memory Estimator
"""

import re
from typing import Dict, Optional, Any


class WeightClassifier:
    """Classify HuggingFace weights to standard module types"""

    def __init__(self, rules: Dict[str, Any]):
        self.rules = rules

    def classify_weight(self, weight_name: str, model_type: Optional[str] = None) -> str:
        """Classify a weight name to a module type"""
        # Try model-specific rules first
        if model_type and model_type in self.rules.get('model_specific', {}):
            model_rules = self.rules['model_specific'][model_type]
            result = self._match_rules(weight_name, model_rules)
            if result:
                return result

        # Try generic rules
        generic_rules = self.rules.get('generic', {})
        result = self._match_rules(weight_name, generic_rules)
        if result:
            return result

        # Default to others
        return "others"

    def _match_rules(self, weight_name: str, rules: Dict[str, Any]) -> Optional[str]:
        """Match weight name against rules"""
        for module_type, rule_config in rules.items():
            patterns = rule_config.get('patterns', [])
            excludes = rule_config.get('exclude', [])

            # Check if matches any pattern
            matched = False
            for pattern in patterns:
                if self._match_pattern(weight_name, pattern):
                    matched = True
                    break

            if not matched:
                continue

            # Check if matches any exclude pattern
            excluded = False
            for exclude_pattern in excludes:
                if self._match_pattern(weight_name, exclude_pattern):
                    excluded = True
                    break

            if not excluded:
                return module_type

        return None

    def _match_pattern(self, weight_name: str, pattern: str) -> bool:
        """Match a weight name against a pattern (supports wildcards)"""
        # Convert pattern to regex
        # Replace * with .* for wildcard matching
        regex_pattern = pattern.replace('.', r'\.').replace('*', '.*')
        regex_pattern = f"^{regex_pattern}$"

        return bool(re.match(regex_pattern, weight_name))
