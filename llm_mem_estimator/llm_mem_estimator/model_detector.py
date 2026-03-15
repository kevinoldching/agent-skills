#!/usr/bin/env python3
"""
Model detector and config generator for LLM Memory Estimator
"""

import json
from pathlib import Path
from typing import Dict, Optional, Any

from .model_config import (
    ModelConfig, ModelIdentity, ArchitectureConfig, WeightInfo
)
from .weight_classifier import WeightClassifier


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

            # Use get_safetensors_metadata to read metadata without downloading
            metadata = get_safetensors_metadata(model_name)

            # Extract all tensor information
            all_tensors = {}
            f_meta = metadata.files_metadata

            for tensor_name, file_name in metadata.weight_map.items():
                # Get file metadata
                if isinstance(f_meta, dict):
                    file_obj = f_meta[file_name]
                else:
                    file_obj = next(f for f in f_meta if getattr(f, 'file_name', '') == file_name)

                # Get tensor info from file metadata
                if tensor_name in file_obj.tensors:
                    tensor_info = file_obj.tensors[tensor_name]
                    all_tensors[tensor_name] = {
                        'shape': list(tensor_info.shape),
                        'dtype': str(tensor_info.dtype)
                    }

            return all_tensors

        except Exception as e:
            raise RuntimeError(f"Failed to get HuggingFace weights metadata: {e}")

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

        # Get weights metadata
        weights_metadata = ModelDetector.get_weights_metadata(model_name_or_path, is_local)

        # Detect model type if not provided
        if not model_type:
            model_type = hf_config.get('model_type', 'unknown')

        # Build model identity
        model_identity = ModelIdentity(
            name=model_name_or_path.split('/')[-1] if '/' in model_name_or_path else model_name_or_path,
            total_params=f"{hf_config.get('num_parameters', 'unknown')}",
            num_layers=hf_config.get('num_hidden_layers', 0),
            quantization=hf_config.get('quantization_config', {}).get('quant_method') if 'quantization_config' in hf_config else None
        )

        # Build architecture config
        architecture_config = ArchitectureConfig(
            hidden_size=hf_config.get('hidden_size', 0),
            num_layers=hf_config.get('num_hidden_layers', 0),
            attention_type=self._detect_attention_type(hf_config),
            ffn_type=self._detect_ffn_type(hf_config),
            norm_type=self._detect_norm_type(hf_config),
            vocab_size=hf_config.get('vocab_size', 0),
            num_attention_heads=hf_config.get('num_attention_heads'),
            num_key_value_heads=hf_config.get('num_key_value_heads'),
            intermediate_size=hf_config.get('intermediate_size'),
            num_experts=hf_config.get('num_experts'),
            num_experts_per_tok=hf_config.get('num_experts_per_tok'),
            moe_intermediate_size=hf_config.get('moe_intermediate_size'),
            q_lora_rank=hf_config.get('q_lora_rank'),
            kv_lora_rank=hf_config.get('kv_lora_rank'),
            qk_rope_head_dim=hf_config.get('qk_rope_head_dim'),
            v_head_dim=hf_config.get('v_head_dim'),
            qk_nope_head_dim=hf_config.get('qk_nope_head_dim')
        )

        # Classify weights
        modules = self._classify_weights(weights_metadata, model_type, architecture_config)

        # Generate computation rules (simplified)
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

    def _detect_ffn_type(self, config: Dict[str, Any]) -> str:
        """Detect FFN type from config"""
        if config.get('num_experts'):
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
                         model_type: str, arch_config: ArchitectureConfig) -> Dict[str, Dict[str, WeightInfo]]:
        """Classify weights into module types"""
        import re
        from collections import defaultdict

        # Step 1: Group weights by their base pattern (without layer numbers)
        pattern_groups = defaultdict(list)

        for weight_name, metadata in weights_metadata.items():
            # Extract base pattern by removing layer numbers
            base_pattern = re.sub(r'\.layers?\.\d+\.', '.layers.N.', weight_name)
            base_pattern = re.sub(r'model\.layers\.\d+', 'model.layers.N', base_pattern)

            pattern_groups[base_pattern].append((weight_name, metadata))

        # Step 2: Classify and merge weights
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

            if is_per_layer and len(weight_list) > 1:
                # This is a per-layer weight that appears in multiple layers
                # Store it once with the base pattern name and set layers count
                layers_count = len(weight_list)

                # Use a simplified name (remove model.layers.N prefix if present)
                simplified_name = base_pattern.replace('model.layers.N.', '')

                modules[module_type][simplified_name] = WeightInfo(
                    shape=first_metadata['shape'],
                    dtype=first_metadata['dtype'],
                    layers=layers_count,
                    parallel_strategy="replicated",
                    world_size=0
                )
            else:
                # This is a shared weight (embedding, final norm, etc.) or single occurrence
                # Store it with its original name
                modules[module_type][first_weight_name] = WeightInfo(
                    shape=first_metadata['shape'],
                    dtype=first_metadata['dtype'],
                    layers=1,
                    parallel_strategy="replicated",
                    world_size=0
                )

        return modules

    def _generate_computation_rules(self, arch_config: ArchitectureConfig) -> Dict[str, str]:
        """Generate computation rules based on architecture"""
        rules = {}

        # KV Cache formula
        if arch_config.attention_type == "mla":
            # MLA uses compressed KV cache
            if arch_config.kv_lora_rank:
                rules['kv_cache'] = f"2 * batch_size * seq_len * {arch_config.kv_lora_rank} * num_layers"
        elif arch_config.attention_type in ["mha", "gqa", "mqa"]:
            # Standard KV cache
            kv_heads = arch_config.num_key_value_heads or arch_config.num_attention_heads
            head_dim = arch_config.hidden_size // arch_config.num_attention_heads if arch_config.num_attention_heads else 128
            kv_dim = kv_heads * head_dim if kv_heads else arch_config.hidden_size
            rules['kv_cache'] = f"2 * batch_size * seq_len * {kv_dim} * num_layers"

        # Activation formula (simplified)
        rules['activation'] = f"4 * batch_size * seq_len * hidden_size * num_layers"

        return rules
