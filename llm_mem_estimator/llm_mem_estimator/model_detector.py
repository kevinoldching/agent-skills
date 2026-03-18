#!/usr/bin/env python3
"""
Model detector, config generator, and weight classifier for LLM Memory Estimator

This module combines:
- ModelDetector: Detect model architecture from weights
- ConfigGenerator: Generate model configuration from weights
- WeightClassifier: Classify HuggingFace weights to standard module types
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional, Any, List
from collections import defaultdict

from .model_config import (
    ModelConfig, ModelIdentity, ArchitectureConfig, WeightInfo
)


# ============================================================================
# Weight Classifier
# ============================================================================

class WeightClassifier:
    """Classify HuggingFace weights to standard module types"""

    def __init__(self, rules: Dict[str, Any]):
        self.rules = rules
        self._resolve_inheritance()

    def _resolve_inheritance(self):
        """Resolve 'inherit' directives in rules"""
        for model_type, model_rules in list(self.rules.items()):
            if isinstance(model_rules, dict) and 'inherit' in model_rules:
                parent = model_rules['inherit']
                if parent in self.rules:
                    # Merge parent rules with current rules
                    merged_rules = dict(self.rules[parent])
                    # Override with current rules (excluding 'inherit' key)
                    for key, value in model_rules.items():
                        if key != 'inherit':
                            merged_rules[key] = value
                    self.rules[model_type] = merged_rules

    def classify_weight(self, weight_name: str, model_type: Optional[str] = None) -> str:
        """Classify a weight name to a module type"""
        # Try model-specific rules first
        if model_type and model_type in self.rules:
            model_rules = self.rules[model_type]
            if isinstance(model_rules, dict) and 'inherit' not in model_rules:
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

    def get_parallel_strategy(self, weight_name: str, module_type: str, model_type: Optional[str] = None) -> str:
        """Determine parallel strategy based on weight name and module type

        Args:
            weight_name: The weight name (e.g., 'mlp.experts.0.gate_proj.weight')
            module_type: The module type (e.g., 'ffn_moe', 'attention', 'embedding')
            model_type: Optional model type for model-specific rules

        Returns:
            Parallel strategy (e.g., 'TP', 'EP', 'replicated')
        """
        # Get parallel_defaults from model-specific or generic rules
        parallel_defaults = None

        # Try model-specific rules first
        if model_type and model_type in self.rules:
            model_rules = self.rules[model_type]
            parallel_defaults = model_rules.get('parallel_defaults')

        # Fall back to generic rules
        if not parallel_defaults:
            generic_rules = self.rules.get('generic', {})
            parallel_defaults = generic_rules.get('parallel_defaults', {})

        if not parallel_defaults:
            return 'replicated'

        # Get module-specific defaults
        module_defaults = parallel_defaults.get(module_type)

        # If module not in parallel_defaults, default to replicated
        if module_defaults is None:
            return 'replicated'

        # If it's a simple string (e.g., 'TP', 'EP'), return directly
        if isinstance(module_defaults, str):
            return module_defaults

        # If it's a dict, match by keyword
        if isinstance(module_defaults, dict):
            # Sort keywords by length (longest first) to match more specific patterns first
            # e.g., 'experts' should match before 'gate' in 'mlp.experts.0.gate_proj.weight'
            sorted_keywords = sorted(
                [k for k in module_defaults.keys()],
                key=len,
                reverse=True
            )
            # First check for specific keyword matches (longer keywords first)
            import re
            for keyword in sorted_keywords:
                # Match keyword as complete segment:
                # - keyword ends with .weight: match exactly 'keyword' in weight_name
                # - keyword is a simple name: match as segment in weight_name.split('.')
                if keyword.endswith('.weight'):
                    # For patterns like 'q_proj.weight', check if weight_name ends with it
                    if weight_name.endswith(keyword):
                        return module_defaults[keyword]
                else:
                    # For simple keywords like 'experts', check if it's a complete segment
                    # Split and check exact match
                    weight_parts = weight_name.split('.')
                    if keyword in weight_parts:
                        return module_defaults[keyword]
            # Return replicated if no match
            return 'replicated'

        return 'replicated'

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
        # Pattern is already in regex format from YAML (with \. for literal dots)
        # Just wrap it with ^ and $
        regex_pattern = f"^{pattern}$"

        try:
            return bool(re.match(regex_pattern, weight_name))
        except re.error:
            # If regex is invalid, return False
            return False


# ============================================================================
# Model Detector
# ============================================================================

class ModelDetector:
    """Detect model architecture from weights"""

    @staticmethod
    def detect_from_huggingface(model_name: str) -> Dict[str, Any]:
        """Detect model architecture from HuggingFace model name"""
        try:
            from huggingface_hub import hf_hub_download

            # Download config.json
            config_path = hf_hub_download(repo_id=model_name, filename="config.json")

            with open(config_path, 'r') as f:
                config = json.load(f)

            return config
        except Exception as e:
            raise RuntimeError(f"Failed to download config from HuggingFace: {e}")

    @staticmethod
    def detect_from_local(weights_path: str) -> Dict[str, Any]:
        """Detect model architecture from local weights"""
        weights_path = Path(weights_path)

        # Try to find config.json
        config_path = weights_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)

        raise FileNotFoundError(f"config.json not found in {weights_path}")

    @staticmethod
    def get_weights_metadata(model_name_or_path: str, is_local: bool = False) -> Dict[str, Dict[str, Any]]:
        """Get weights metadata from HuggingFace or local path"""
        try:
            if is_local:
                return ModelDetector._get_local_weights_metadata(model_name_or_path)
            else:
                return ModelDetector._get_huggingface_weights_metadata(model_name_or_path)
        except Exception as e:
            raise RuntimeError(f"Failed to get weights metadata: {e}")

    @staticmethod
    def _get_huggingface_weights_metadata(model_name: str) -> Dict[str, Dict[str, Any]]:
        """Get weights metadata from HuggingFace using HTTP Range Requests (no download)"""
        try:
            from huggingface_hub import get_safetensors_metadata
            
            # 核心：只抓取元数据
            metadata = get_safetensors_metadata(model_name)
            all_tensors = {}

            # 遍历所有文件和其中的张量
            for file_name, file_info in metadata.files_metadata.items():
                for tensor_name, tensor_info in file_info.tensors.items():
                    all_tensors[tensor_name] = {
                        'shape': list(tensor_info.shape),
                        'dtype': str(tensor_info.dtype)
                    }
            
            return all_tensors

        except Exception as e:
            # 常见错误：Repo 不含 safetensors，或者网络连不通 HF
            raise RuntimeError(f"无法获取元数据 (请检查是否包含 safetensors): {e}")

    @staticmethod
    def _get_local_weights_metadata(weights_path: str) -> Dict[str, Dict[str, Any]]:
        """Get weights metadata from local path"""
        import safetensors
        weights_path = Path(weights_path)

        # Try to find safetensors files
        safetensors_files = list(weights_path.glob("*.safetensors"))

        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {weights_path}")

        metadata = {}
        for shard_file in safetensors_files:
            with safetensors.safe_open(str(shard_file), framework="pt") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    metadata[key] = {
                        'shape': list(tensor.shape),
                        'dtype': str(tensor.dtype).replace('torch.', '')
                    }

        return metadata


# ============================================================================
# Config Generator
# ============================================================================

class ConfigGenerator:
    """Generate model configuration from weights"""

    def __init__(self, weight_classifier: WeightClassifier):
        self.classifier = weight_classifier

    def generate_config(self, model_name_or_path: str, is_local: bool = False,
                       model_type: Optional[str] = None) -> ModelConfig:
        """Generate model configuration from HuggingFace or local weights"""
        # Get model config
        if is_local:
            hf_config = ModelDetector.detect_from_local(model_name_or_path)
        else:
            hf_config = ModelDetector.detect_from_huggingface(model_name_or_path)

        # Get weights metadata and calculate total parameters
        weights_metadata = ModelDetector.get_weights_metadata(model_name_or_path, is_local)

        # Calculate total parameters from safetensors metadata
        total_params = "unknown"
        if not is_local:
            try:
                from huggingface_hub import get_safetensors_metadata
                metadata = get_safetensors_metadata(model_name_or_path)
                if hasattr(metadata, 'parameter_count') and metadata.parameter_count:
                    total_params = str(sum(metadata.parameter_count.values()))
            except Exception:
                # Fallback to config if metadata not available
                total_params = f"{hf_config.get('num_parameters', 'unknown')}"
        else:
            # For local weights, calculate from weights metadata
            try:
                from llm_mem_estimator.model_config import get_dtype_bytes
                total_elements = 0
                for weight_name, weight_info in weights_metadata.items():
                    shape = weight_info.get('shape', [])
                    dtype = weight_info.get('dtype', 'fp16')
                    elements = 1
                    for dim in shape:
                        elements *= dim
                    total_elements += elements
                if total_elements > 0:
                    total_params = str(total_elements)
            except Exception:
                pass

        # Detect model type if not provided
        if not model_type:
            model_type = hf_config.get('model_type', 'unknown')

        # Build model identity (extract last part from path or HF name)
        model_name = model_name_or_path.split('/')[-1] if '/' in model_name_or_path else model_name_or_path
        model_identity = ModelIdentity(
            name=model_name,
            total_params=total_params,
            num_layers=hf_config.get('num_hidden_layers', 0),
            quantization=hf_config.get('quantization_config', {}).get('quant_method') if 'quantization_config' in hf_config else None
        )

        # Get weight mapping rules for architecture config mapping and expert detection
        weight_rules = self.classifier.rules

        # Get model-specific rules with priority:
        # 1. Try model name (e.g., gpt-oss-120b) - highest priority
        # 2. Try model_type (e.g., gpt_oss)
        # 3. Fall back to generic
        model_rules = (
            weight_rules.get(model_name) or
            weight_rules.get(model_type) or
            weight_rules.get('generic', {})
        )

        # Get architecture config field mappings
        arch_config_mapping = weight_rules.get('architecture_config', {}).get('field_mappings', {})

        # Helper function to get config value using field mappings
        def get_config_value(field_name: str, default: Any = None) -> Any:
            """Get config value using field mappings"""
            mapping_keys = arch_config_mapping.get(field_name, [field_name])
            for key in mapping_keys:
                if key in hf_config and hf_config[key] is not None:
                    return hf_config[key]
            return default

        # Get head_dim from config or calculate from hidden_size / num_attention_heads
        head_dim = get_config_value('head_dim')
        if not head_dim:
            hidden_size = get_config_value('hidden_size', 0)
            num_attention_heads = get_config_value('num_attention_heads')
            if hidden_size and num_attention_heads:
                head_dim = hidden_size // num_attention_heads

        # Build architecture config using field mappings
        architecture_config = ArchitectureConfig(
            hidden_size=get_config_value('hidden_size', 0),
            head_dim=head_dim or 0,
            num_layers=get_config_value('num_layers', 0),
            attention_type=self._detect_attention_type(hf_config),
            ffn_type=self._detect_ffn_type(hf_config, get_config_value('num_experts')),
            norm_type=self._detect_norm_type(hf_config),
            vocab_size=get_config_value('vocab_size', 0),
            num_attention_heads=get_config_value('num_attention_heads'),
            num_key_value_heads=get_config_value('num_key_value_heads'),
            intermediate_size=get_config_value('intermediate_size'),
            num_experts=get_config_value('num_experts'),
            num_experts_per_tok=get_config_value('num_experts_per_tok'),
            moe_intermediate_size=get_config_value('moe_intermediate_size'),
            q_lora_rank=get_config_value('q_lora_rank'),
            kv_lora_rank=get_config_value('kv_lora_rank'),
            qk_rope_head_dim=get_config_value('qk_rope_head_dim'),
            v_head_dim=get_config_value('v_head_dim'),
            qk_nope_head_dim=get_config_value('qk_nope_head_dim'),
            window_size=get_config_value('window_size')
        )

        # Get ffn_moe patterns for expert detection
        ffn_moe_patterns = model_rules.get('ffn_moe', {}).get('patterns', [])

        # Classify weights (pass ffn_moe patterns for expert detection)
        modules = self._classify_weights(weights_metadata, model_type, architecture_config, ffn_moe_patterns)

        # Get computation rules from weight_mapping_rules.yaml
        # Priority: model_name > model_type > generic
        computation_rules = model_rules.get('computation_rules', {})

        # If not found in model_rules, generate from architecture
        if not computation_rules:
            computation_rules = self._generate_computation_rules(architecture_config)

        return ModelConfig(
            model_identity=model_identity,
            architecture_config=architecture_config,
            modules=modules,
            computation_rules=computation_rules
        )

    def _detect_attention_type(self, config: Dict[str, Any]) -> str:
        """Detect attention type from config"""
        if 'q_lora_rank' in config and 'kv_lora_rank' in config:
            return "mla"
        elif config.get('num_key_value_heads') == 1:
            return "mqa"
        elif config.get('num_key_value_heads') and config.get('num_key_value_heads') < config.get('num_attention_heads', 0):
            return "gqa"
        else:
            return "mha"

    def _detect_ffn_type(self, config: Dict[str, Any], num_experts: Any = None) -> str:
        """Detect FFN type from config

        Args:
            config: HuggingFace model config
            num_experts: Pre-resolved num_experts value (optional)
        """
        # Use provided num_experts or try to get from config
        experts = num_experts if num_experts is not None else config.get('num_experts')
        if experts:
            return "moe"
        elif 'hidden_act' in config and 'swiglu' in config['hidden_act'].lower():
            return "swiglu"
        else:
            return "standard"

    def _detect_norm_type(self, config: Dict[str, Any]) -> str:
        """Detect normalization type from config"""
        norm_type = config.get('norm_type', config.get('rms_norm_eps'))
        if norm_type or 'rms_norm_eps' in config:
            return "rmsnorm"
        else:
            return "layernorm"

    def _classify_weights(self, weights_metadata: Dict[str, Dict[str, Any]],
                         model_type: str, arch_config: ArchitectureConfig,
                         ffn_moe_patterns: List[str] = None) -> Dict[str, Dict[str, WeightInfo]]:
        """Classify weights into module types

        Args:
            weights_metadata: Dictionary of weight names to metadata
            model_type: Model type identifier
            arch_config: Architecture configuration
            ffn_moe_patterns: List of patterns for matching MoE expert weights
        """

        # Step 0: Detect MoE expert count using ffn_moe patterns from weight_mapping_rules
        all_weight_names = list(weights_metadata.keys())
        expert_indices = set()
        expert_pattern_regex = None
        is_moe = False
        num_experts = 0

        if ffn_moe_patterns:
            for weight_name in all_weight_names:
                for pattern in ffn_moe_patterns:
                    # Try to match the pattern
                    if re.match(pattern, weight_name):
                        # Match found, extract numeric indices from weight name
                        # Use all numeric indices, then take the max as expert count
                        indices = re.findall(r'\.(\d+)\.', weight_name)
                        for idx in indices:
                            expert_indices.add(int(idx))
                        # Build regex pattern for replacement if first match
                        if expert_pattern_regex is None:
                            # Try to find the expert index pattern in the weight name
                            match = re.search(r'(\.experts?\d*\.)\d+(\.)', weight_name)
                            if match:
                                expert_pattern_regex = match.group(1) + '{}' + match.group(2)
                        break

            # Calculate total expert count as max index + 1
            if expert_indices:
                num_experts = max(expert_indices) + 1
                is_moe = num_experts > 0

        # Step 1: Group weights by their base pattern
        # If MoE detected, also remove expert numbers during grouping
        pattern_groups = defaultdict(list)

        for weight_name, metadata in weights_metadata.items():
            # Only remove layer numbers first
            base_pattern = re.sub(r'\.layers?\.\d+\.', '.layers.N.', weight_name)
            base_pattern = re.sub(r'model\.layers\.\d+', 'model.layers.N', base_pattern)

            # For MoE models, also remove expert numbers to consolidate
            # Match patterns like: .experts.0. or .experts0. or .block_sparse_moe.experts.0.
            if is_moe and expert_pattern_regex:
                # Replace expert index with N
                base_pattern = re.sub(r'(\.experts?\d*)\.\d+(\.)', r'\1.N\2', base_pattern)

            pattern_groups[base_pattern].append((weight_name, metadata))

        # Update architecture config if MoE detected
        if is_moe and not arch_config.num_experts:
            arch_config.num_experts = num_experts
            arch_config.ffn_type = "moe"

        # Step 2: Process each group
        modules = {}

        for base_pattern, weight_list in pattern_groups.items():
            # Use the first weight to get metadata
            first_weight_name, first_metadata = weight_list[0]

            # Classify the base pattern
            module_type = self.classifier.classify_weight(first_weight_name, model_type)

            if module_type not in modules:
                modules[module_type] = {}

            # Determine if this is a per-layer weight
            is_per_layer = '.layers.N.' in base_pattern or 'model.layers.N' in base_pattern

            # Check if this group has expert consolidation
            has_experts_in_pattern = is_moe and '.experts.N.' in base_pattern

            if is_per_layer and len(weight_list) > 1:
                # For MoE: layers_count = num_layers (not total weight count)
                if has_experts_in_pattern and num_experts > 0:
                    layers_count = arch_config.num_layers or len(weight_list)
                else:
                    layers_count = len(weight_list)

                # Use a simplified name (remove model.layers.N prefix if present)
                simplified_name = base_pattern.replace('model.layers.N.', '')

                # For MoE expert weights, incorporate expert dimension into shape
                shape = first_metadata['shape'].copy()
                if has_experts_in_pattern and num_experts > 0:
                    # Insert expert count as the first dimension
                    # Shape format: [hidden, intermediate] -> [num_experts, hidden, intermediate]
                    if len(shape) >= 2:
                        shape.insert(0, num_experts)

                # Determine parallel strategy using classifier
                parallel_strategy = self.classifier.get_parallel_strategy(
                    simplified_name, module_type, model_type
                )

                modules[module_type][simplified_name] = WeightInfo(
                    shape=shape,
                    dtype=first_metadata['dtype'],
                    layers=layers_count,
                    parallel_strategy=parallel_strategy,
                    world_size=0
                )
            else:
                # This is a shared weight (embedding, final norm, etc.) or single occurrence
                # Store it with its original name
                # Determine parallel strategy using classifier
                parallel_strategy = self.classifier.get_parallel_strategy(
                    first_weight_name, module_type, model_type
                )

                modules[module_type][first_weight_name] = WeightInfo(
                    shape=first_metadata['shape'],
                    dtype=first_metadata['dtype'],
                    layers=1,
                    parallel_strategy=parallel_strategy,
                    world_size=0
                )

        return modules

    def _generate_computation_rules(self, arch_config: ArchitectureConfig) -> Dict[str, Any]:
        """Generate computation rules based on architecture"""
        rules = {}

        # Default capacity factor (used for activation calculation)
        # recommended_capacity_factor: 1.25 (industrial standard)
        # ideal: 1.0, worst_case: 8.0
        recommended_capacity_factor = 1.25
        rules['recommended_capacity_factor'] = recommended_capacity_factor

        # KV Cache formula (with tp_size/cp_size, without dtype_bytes)
        if arch_config.attention_type == "mla":
            # MLA uses compressed KV cache
            if arch_config.kv_lora_rank:
                rules['kv_cache'] = f"batch_size * seq_len * (kv_lora_rank + qk_rope_head_dim) * num_layers / cp_size"
        elif arch_config.attention_type == "swa" or arch_config.attention_type == "sliding_window":
            # Sliding window attention
            window_size = arch_config.window_size or 512
            rules['kv_cache'] = f"2 * batch_size * min(seq_len, {window_size}) * num_key_value_heads * head_dim * num_layers / (tp_size * cp_size)"
        elif arch_config.attention_type in ["mha", "gqa", "mqa"]:
            # Standard KV cache
            rules['kv_cache'] = f"2 * batch_size * seq_len * num_key_value_heads * head_dim * num_layers / (tp_size * cp_size)"

        # Activation formula with capacity factor
        # Formula includes tp_size/cp_size, but NOT dtype_bytes (handled externally)
        if arch_config.ffn_type == "moe" and arch_config.num_experts_per_tok:
            # MoE model: include num_experts_per_tok in formula
            rules['activation'] = (
                f"batch_size * seq_len * hidden_size * "
                f"num_experts_per_tok * {recommended_capacity_factor} / cp_size"
            )
        else:
            # Standard/Dense model
            rules['activation'] = (
                f"batch_size * seq_len * hidden_size * "
                f"{recommended_capacity_factor} / cp_size"
            )

        return rules
