#!/usr/bin/env python3
"""
Model detector, config generator, and weight classifier for LLM Memory Estimator

This module combines:
- ModelDetector: Detect model architecture from weights
- ConfigGenerator: Generate model configuration from weights
- WeightClassifier: Classify HuggingFace weights to standard module types
"""

import os
import time
import json
import re
from pathlib import Path
from typing import Dict, Optional, Any, List, Literal
from collections import defaultdict

from .model_config import (
    ModelConfig, ModelIdentity, ArchitectureConfig, WeightInfo, ConfigLoader
)


# ============================================================================
# Weight Classifier
# ============================================================================

class WeightClassifier:
    """Classify HuggingFace weights to standard module types"""

    def __init__(self, rules: Dict[str, Any]):
        self.rules = rules
        self._resolve_inheritance()
        # Parse three-level parallel defaults for PD separation support
        self._parallel_defaults_cache: Dict[str, Dict[str, Dict]] = {}
        self._resolve_parallel_defaults_inheritance()
        # Parse tp_variants for TP variant support
        self._tp_variants_cache: Dict[str, Dict[str, int]] = {}
        self._resolve_tp_variants_inheritance()

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
                            # Deep merge for nested dicts (like computation_rules)
                            if key in merged_rules and isinstance(merged_rules[key], dict) and isinstance(value, dict):
                                merged_rules[key] = {**merged_rules[key], **value}
                            else:
                                merged_rules[key] = value
                    self.rules[model_type] = merged_rules

    def _resolve_parallel_defaults_inheritance(self):
        """Resolve three-level parallel_defaults structure for PD separation

        Parses:
        - parallel_defaults → hybrid (混部/通用场景)
        - parallel_defaults.prefill → prefill (PD分离-prefill)
        - parallel_defaults.decode → decode (PD分离-decode)

        Each level inherits from generic if not explicitly defined in model.
        """
        generic_rules = self.rules.get('generic', {})

        for model_type, model_rules in self.rules.items():
            if not isinstance(model_rules, dict):
                continue

            # Use ConfigLoader's helper to parse the three-level structure
            parsed = ConfigLoader._parse_parallel_defaults(model_rules, generic_rules)
            self._parallel_defaults_cache[model_type] = parsed

    def _resolve_tp_variants_inheritance(self):
        """Resolve tp_variants inheritance

        tp_variants: Dict[str, int] - maps variant name (e.g., 'TP_O_PROJ') to default TP size

        Inheritance rule: model's tp_variants completely replaces (not merges) parent's
        Falls back to 'generic' tp_variants if model doesn't define any.
        """
        for model_type, model_rules in self.rules.items():
            if not isinstance(model_rules, dict):
                self._tp_variants_cache[model_type] = {}
                continue

            # Check if model has tp_variants
            if 'tp_variants' in model_rules:
                self._tp_variants_cache[model_type] = dict(model_rules['tp_variants'])
            elif model_type == 'generic':
                # generic always has tp_variants
                self._tp_variants_cache[model_type] = dict(model_rules.get('tp_variants', {}))
            elif 'inherit' in model_rules:
                # Inherit from parent (which should have been resolved already)
                parent = model_rules['inherit']
                if parent in self._tp_variants_cache:
                    self._tp_variants_cache[model_type] = dict(self._tp_variants_cache[parent])
                else:
                    # Fall back to generic
                    self._tp_variants_cache[model_type] = dict(self._tp_variants_cache.get('generic', {}))
            else:
                # No tp_variants and no inherit, fall back to generic
                self._tp_variants_cache[model_type] = dict(self._tp_variants_cache.get('generic', {}))

    def get_tp_variant_size(self, variant_name: str, model_type: Optional[str] = None) -> Optional[int]:
        """Get the TP size for a given variant name

        Args:
            variant_name: The variant name (e.g., 'TP_O_PROJ', 'TP_MLP')
            model_type: Optional model type for model-specific variants

        Returns:
            TP size for the variant, or None if variant not found
        """
        # Only look up if variant name starts with TP_ (it's a variant)
        if not variant_name.startswith('TP_'):
            return None

        # Try model-specific first, then fall back to generic
        if model_type and model_type in self._tp_variants_cache:
            variants = self._tp_variants_cache[model_type]
            if variant_name in variants:
                return variants[variant_name]

        # Fall back to generic
        if 'generic' in self._tp_variants_cache:
            variants = self._tp_variants_cache['generic']
            if variant_name in variants:
                return variants[variant_name]

        return None

    def classify_weight(self, weight_name: str, model_name: Optional[str] = None,
                       model_type: Optional[str] = None) -> str:
        """Classify a weight name to a module type

        Args:
            weight_name: The weight name to classify
            model_name: Model name (e.g., 'Qwen3-Coder-Next') - highest priority
            model_type: HuggingFace model_type (e.g., 'qwen3') - second priority
        """
        # Priority: model_name > model_type > generic
        if model_name and model_name in self.rules:
            model_rules = self.rules[model_name]
            if isinstance(model_rules, dict):
                result = self._match_rules(weight_name, model_rules)
                if result:
                    return result

        if model_type and model_type in self.rules:
            model_rules = self.rules[model_type]
            if isinstance(model_rules, dict):
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

    def get_parallel_strategy(self, weight_name: str, module_type: str, model_type: Optional[str] = None,
                               stage: Literal["hybrid", "prefill", "decode"] = "hybrid") -> str:
        """Determine parallel strategy based on weight name and module type

        Args:
            weight_name: The weight name (e.g., 'mlp.experts.0.gate_proj.weight')
            module_type: The module type (e.g., 'ffn_moe', 'attention', 'embedding')
            model_type: Optional model type for model-specific rules
            stage: PD separation stage - "hybrid" (混部/通用), "prefill" (PD分离-prefill), "decode" (PD分离-decode)

        Returns:
            Parallel strategy (e.g., 'TP', 'EP', 'replicated')
        """
        # Get parallel_defaults from cached three-level structure
        # Fall back to old-style single-level lookup for backward compatibility
        parallel_defaults = None

        # Try to get from cached three-level structure
        if model_type and model_type in self._parallel_defaults_cache:
            parallel_defaults = self._parallel_defaults_cache[model_type].get(stage, {})
        elif 'generic' in self._parallel_defaults_cache:
            parallel_defaults = self._parallel_defaults_cache['generic'].get(stage, {})

        # For backward compatibility: if stage is 'hybrid' and we still don't have defaults,
        # fall back to old-style single-level parallel_defaults lookup
        # This handles old YAML configs that only have single parallel_defaults
        if not parallel_defaults and stage == "hybrid":
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
            import os

            # Set HF_TOKEN from environment variable if available
            hf_token = os.environ.get('HF_TOKEN')
            if hf_token:
                os.environ['HF_HUB_TOKEN'] = hf_token

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
    def _load_ssh_key(key_filename: str) -> "Optional[paramiko.PKey]":
        """Try to load SSH key, supporting RSA, ECDSA, and ED25519 types"""
        import paramiko
        for key_class in (paramiko.RSAKey, paramiko.ECDSAKey, paramiko.Ed25519Key):
            try:
                return key_class.from_private_key_file(key_filename)
            except Exception:
                continue
        return None

    @staticmethod
    def detect_from_remote(host: str, remote_path: str, username: str,
                          key_filename: Optional[str] = None) -> Dict[str, Any]:
        """Detect model architecture from remote server via SFTP

        Args:
            host: Remote server hostname or IP
            remote_path: Path to the model directory on remote server
            username: SSH username
            key_filename: Path to SSH private key file (optional)
        """
        import paramiko

        # Connect using SSHClient (supports look_for_keys and allow_agent like scp/ssh)
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Diagnostics: Check what keys/agent is available
        print(f"\n=== SSH Diagnostics ===")
        print(f"Host: {host}, Username: {username}")
        print(f"Key filename specified: {key_filename}")

        # Check SSH agent
        try:
            agent = paramiko.Agent()
            agent_keys = agent.get_keys()
            print(f"SSH Agent keys: {len(agent_keys)} found")
            for i, k in enumerate(agent_keys):
                print(f"  Agent key {i}: {k.get_name()}")
        except Exception as agent_err:
            print(f"SSH Agent error: {agent_err}")

        # Check default key files
        home = Path.home()
        default_keys = [
            home / '.ssh' / 'id_rsa',
            home / '.ssh' / 'id_ed25519',
            home / '.ssh' / 'id_ecdsa',
            home / '.ssh' / 'id_ed448',
        ]
        found_keys = [k for k in default_keys if k.exists()]
        print(f"Key files found: {[str(k) for k in found_keys]}")

        # Check SSH config
        ssh_config_path = home / '.ssh' / 'config'
        if ssh_config_path.exists():
            print(f"SSH config exists at: {ssh_config_path}")
            # Try to read host-specific config
            try:
                with open(ssh_config_path, 'r') as f:
                    config_content = f.read()
                    # Check if our host is configured
                    if host in config_content or f"Host {host}" in config_content:
                        print(f"  Host '{host}' found in SSH config!")
            except Exception:
                pass
        else:
            print(f"No SSH config at: {ssh_config_path}")

        print(f"=== End Diagnostics ===\n")

        try:
            # look_for_keys=True searches ~/.ssh/ automatically like scp
            # allow_agent=True tries SSH agent if available
            ssh.connect(
                hostname=host,
                username=username,
                password=None,
                pkey=None,
                look_for_keys=True,
                allow_agent=True
            )
            print("Connected via SSH")
        except Exception as e:
            print(f"\nSSH connect failed: {type(e).__name__}: {e}")
            if key_filename:
                # Fallback to specified key file
                pkey = ModelDetector._load_ssh_key(key_filename)
                if pkey is None:
                    raise RuntimeError(f"Failed to load SSH key: {key_filename}")
                print(f"Using SSH key file: {key_filename}")
                try:
                    ssh.connect(
                        hostname=host,
                        username=username,
                        password=None,
                        pkey=pkey
                    )
                    print("Connected with explicit key")
                except Exception as e2:
                    print(f"Explicit key also failed: {type(e2).__name__}: {e2}")
                    raise RuntimeError(f"SSH connection failed with explicit key: {e2}")
            else:
                raise RuntimeError(f"SSH connection failed: {e}")

        # Get SFTP client from SSH client
        sftp = ssh.open_sftp()

        # Try to find config.json
        config_path = remote_path.rstrip('/') + "/config.json"
        try:
            with sftp.file(config_path, 'rb') as f:
                config_data = f.read()
                return json.loads(config_data.decode('utf-8'))
        except FileNotFoundError:
            raise FileNotFoundError(f"config.json not found at {config_path}")
        finally:
            sftp.close()
            ssh.close()

    @staticmethod
    def get_weights_metadata(model_name_or_path: str, is_local: bool = False,
                            is_remote: bool = False, remote_username: Optional[str] = None,
                            remote_host: Optional[str] = None,
                            remote_key_filename: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Get weights metadata from HuggingFace, local path, or remote server via SFTP

        Args:
            model_name_or_path: HuggingFace model name, local path, or remote path (used as remote_path when is_remote=True)
            is_local: If True, treat model_name_or_path as local filesystem path
            is_remote: If True, treat model_name_or_path as remote_path and use SFTP
            remote_username: SSH username (required when is_remote=True)
            remote_host: SSH host (required when is_remote=True)
            remote_key_filename: Path to SSH private key (optional, defaults to ~/.ssh/id_rsa or ~/.ssh/id_ed25519)
        """
        try:
            if is_local:
                return ModelDetector._get_local_weights_metadata(model_name_or_path)
            elif is_remote:
                if not remote_username or not remote_host:
                    raise ValueError("remote_username and remote_host are required when is_remote=True")
                return ModelDetector._get_remote_weights_metadata(
                    remote_host, model_name_or_path, remote_username, remote_key_filename
                )
            else:
                return ModelDetector._get_huggingface_weights_metadata(model_name_or_path)
        except Exception as e:
            raise RuntimeError(f"Failed to get weights metadata: {e}")

    @staticmethod
    def _get_huggingface_weights_metadata(model_name: str) -> Dict[str, Dict[str, Any]]:
        """
        使用 HTTP Range Requests 获取权重元数据，并集成以下功能：
        1. 本地 JSON 缓存：避免重复请求和断网困扰。
        2. 自动重试机制：应对 [Errno 104] Connection reset。
        3. 镜像源支持：自动切换到国内高速节点。
        """
        # --- 配置参数 ---
        MAX_RETRIES = 3
        # 如果你在国内，建议开启镜像源
        if not os.environ.get('HF_ENDPOINT'):
            os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

        # 1. 准备本地缓存路径（使用用户 home 目录下的标准缓存位置）
        cache_dir = Path.home() / ".cache" / "llm_mem_estimator" / "metadata_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # 将 "org/model" 转换为 "org--model.json"
        safe_name = model_name.replace("/", "--")
        cache_path = cache_dir / f"{safe_name}_weights.json"

        # 2. 检查并读取本地缓存
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    # 简单校验数据有效性
                    if isinstance(cached_data, dict) and len(cached_data) > 0:
                        print(f"加载本地元数据缓存: {cache_path}")
                        return cached_data
            except Exception:
                pass  # 缓存损坏则重新请求

        # 3. 如果无缓存，执行带重试的网络请求
        try:
            from huggingface_hub import get_safetensors_metadata

            metadata = None
            last_error = None

            for attempt in range(MAX_RETRIES):
                try:
                    # 核心请求：只读 Header
                    metadata = get_safetensors_metadata(model_name)
                    if metadata:
                        break
                except Exception as e:
                    last_error = e
                    # 针对网络重置进行等待重试
                    if "104" in str(e) or "reset" in str(e).lower() or "timeout" in str(e).lower():
                        wait = 2 ** attempt
                        print(f"网络波动 ({e})，正在进行第 {attempt + 1}/{MAX_RETRIES} 次重试，等待 {wait}s...")
                        time.sleep(wait)
                    else:
                        raise  # 其他错误直接抛出

            if not metadata:
                raise RuntimeError(
                    f"在 {MAX_RETRIES} 次重试后仍无法获取元数据: {last_error}"
                )

            # 4. 转换结构
            all_tensors = {}
            for file_name, file_info in metadata.files_metadata.items():
                for tensor_name, tensor_info in file_info.tensors.items():
                    all_tensors[tensor_name] = {
                        'shape': list(tensor_info.shape),
                        'dtype': str(tensor_info.dtype)
                    }

            # 5. 写入本地缓存以备后用
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(all_tensors, f, indent=4)

            return all_tensors

        except ImportError:
            raise RuntimeError("请先安装依赖: pip install huggingface_hub")
        except Exception as e:
            raise RuntimeError(f"无法获取元数据: {e}")

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

    @staticmethod
    def _get_remote_weights_metadata(host: str, remote_path: str, username: str,
                                     key_filename: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Get weights metadata from remote server via SFTP

        Only reads safetensors file headers (first 8 + header_size bytes) to get
        tensor metadata (name/shape/dtype), without downloading actual weight data.

        Args:
            host: Remote server hostname or IP
            remote_path: Path to the model directory on remote server
            username: SSH username
            key_filename: Path to SSH private key file (optional, defaults to ~/.ssh/id_rsa or ~/.ssh/id_ed25519)
        """
        import paramiko
        import struct

        # Try default SSH key locations if not specified
        if not key_filename:
            home = Path.home()
            default_keys = [
                home / '.ssh' / 'id_rsa',
                home / '.ssh' / 'id_ed25519',
                home / '.ssh' / 'id_ecdsa',
                home / '.ssh' / 'id_ed448',
                Path(os.environ.get('USERPROFILE', '')) / '.ssh' / 'id_rsa',
                Path(os.environ.get('USERPROFILE', '')) / '.ssh' / 'id_ed25519',
            ]
            for key_path in default_keys:
                if key_path.exists() and key_path.is_file():
                    key_filename = str(key_path)
                    print(f"Using SSH key: {key_path}")
                    break

        # Connect using SSHClient (supports look_for_keys and allow_agent like scp/ssh)
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            try:
                ssh.connect(
                    hostname=host,
                    username=username,
                    password=None,
                    pkey=None,
                    look_for_keys=True,
                    allow_agent=True
                )
                print("Connected via SSH")
            except Exception as e:
                if key_filename:
                    pkey = ModelDetector._load_ssh_key(key_filename)
                    if pkey is None:
                        raise RuntimeError(f"Failed to load SSH key: {key_filename}")
                    print(f"Using SSH key file: {key_filename}")
                    ssh.connect(
                        hostname=host,
                        username=username,
                        password=None,
                        pkey=pkey
                    )
                else:
                    raise RuntimeError(f"SSH connection failed: {e}")

            sftp = ssh.open_sftp()

            # Find all safetensors files in remote directory
            try:
                all_files = sftp.listdir(remote_path)
            except FileNotFoundError:
                raise FileNotFoundError(f"Remote path not found: {remote_path}")

            safetensors_files = [f for f in all_files if f.endswith('.safetensors')]
            if not safetensors_files:
                raise FileNotFoundError(f"No safetensors files found in {remote_path}")

            all_tensors = {}

            for shard_file in safetensors_files:
                remote_file_path = remote_path.rstrip('/') + '/' + shard_file
                try:
                    # Open file in binary mode
                    with sftp.file(remote_file_path, 'rb') as f:
                        # Read first 8 bytes to get header_size (uint64, little-endian)
                        header_size_bytes = f.read(8)
                        if len(header_size_bytes) < 8:
                            raise RuntimeError(f"Cannot read header size from {remote_file_path}")

                        header_size = struct.unpack('<Q', header_size_bytes)[0]

                        # Read JSON header (metadata only, not actual weight data)
                        header_json = f.read(header_size)
                        if len(header_json) < header_size:
                            raise RuntimeError(f"Cannot read full header from {remote_file_path}")

                        # Parse JSON metadata
                        header = json.loads(header_json.decode('utf-8'))

                        # Extract tensor info
                        for tensor_name, tensor_info in header.items():
                            all_tensors[tensor_name] = {
                                'shape': tensor_info['shape'],
                                'dtype': tensor_info['dtype']
                            }
                except Exception as e:
                    raise RuntimeError(f"Failed to read metadata from {remote_file_path}: {e}")

            sftp.close()
            ssh.close()
            return all_tensors

        finally:
            ssh.close()


# ============================================================================
# Config Generator
# ============================================================================

class ConfigGenerator:
    """Generate model configuration from weights"""

    def __init__(self, weight_classifier: WeightClassifier):
        self.classifier = weight_classifier

    def generate_config(self, model_name_or_path: str, is_local: bool = False,
                       is_remote: bool = False, remote_username: Optional[str] = None,
                       remote_host: Optional[str] = None, remote_key_filename: Optional[str] = None,
                       model_type: Optional[str] = None) -> ModelConfig:
        """Generate model configuration from HuggingFace, local, or remote weights

        Args:
            model_name_or_path: HuggingFace model name, local path, or remote path
            is_local: If True, treat model_name_or_path as local filesystem path
            is_remote: If True, treat model_name_or_path as remote_path and use SFTP
            remote_username: SSH username (required when is_remote=True)
            remote_host: SSH host (required when is_remote=True)
            remote_key_filename: Path to SSH private key (optional)
            model_type: HuggingFace model_type
        """
        # Get model config
        if is_local:
            hf_config = ModelDetector.detect_from_local(model_name_or_path)
        elif is_remote:
            if not remote_username or not remote_host:
                raise ValueError("remote_username and remote_host are required when is_remote=True")
            hf_config = ModelDetector.detect_from_remote(remote_host, model_name_or_path, remote_username, remote_key_filename)
        else:
            hf_config = ModelDetector.detect_from_huggingface(model_name_or_path)

        # Get weights metadata and calculate total parameters
        weights_metadata = ModelDetector.get_weights_metadata(
            model_name_or_path, is_local, is_remote, remote_username, remote_host, remote_key_filename
        )

        # Calculate total parameters from safetensors metadata
        total_params = "unknown"
        if is_remote:
            # For remote weights, calculate from weights metadata
            try:
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
        elif not is_local:
            try:
                from huggingface_hub import get_safetensors_metadata
                import os

                # Set HF_TOKEN from environment variable if available
                hf_token = os.environ.get('HF_TOKEN')
                if hf_token:
                    os.environ['HF_HUB_TOKEN'] = hf_token

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

        # For multimodal models, get num_layers from text_config
        text_config = hf_config.get('text_config', {})
        num_layers = text_config.get('num_hidden_layers', 0) if text_config else hf_config.get('num_hidden_layers', 0)

        model_identity = ModelIdentity(
            name=model_name,
            total_params=total_params,
            num_layers=num_layers,
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

        # For multimodal models, LLM config is in text_config
        # If text_config exists, use it as the primary config source
        text_config = hf_config.get('text_config', {})

        # Helper function to get config value using field mappings
        def get_config_value(field_name: str, default: Any = None) -> Any:
            """Get config value using field mappings

            For multimodal models, prioritizes text_config over top-level config.
            """
            mapping_keys = arch_config_mapping.get(field_name, [field_name])
            # First check text_config (for multimodal models)
            if text_config:
                for key in mapping_keys:
                    if key in text_config and text_config[key] is not None:
                        return text_config[key]
            # Then check top-level config
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
        # Use text_config for multimodal models
        config_for_detection = text_config if text_config else hf_config
        architecture_config = ArchitectureConfig(
            hidden_size=get_config_value('hidden_size', 0),
            head_dim=head_dim or 0,
            num_layers=get_config_value('num_layers', 0),
            attention_type=self._detect_attention_type(config_for_detection),
            ffn_type=self._detect_ffn_type(config_for_detection, get_config_value('num_experts')),
            norm_type=self._detect_norm_type(config_for_detection),
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

        # Classify weights (pass model_name and model_type for rule matching)
        # Priority: model_name > model_type > generic
        modules = self._classify_weights(weights_metadata, model_name, model_type, architecture_config, ffn_moe_patterns)

        # Get computation rules from weight_mapping_rules.yaml
        # Priority: model_name > model_type > generic
        raw_computation_rules = model_rules.get('computation_rules', {})

        # Raise error if computation_rules not found
        if not raw_computation_rules:
            raise ValueError(
                f"computation_rules not found for model '{model_name}' (model_type: '{model_type}'). "
                f"Please add computation_rules to weight_mapping_rules.yaml for this model or its model_type."
            )

        # Resolve computation_rules: if it's a dict (e.g., kv_cache/activation by type), select the right formula
        computation_rules = self._resolve_computation_rules(
            raw_computation_rules,
            architecture_config.attention_type,
            architecture_config.ffn_type
        )

        return ModelConfig(
            model_identity=model_identity,
            architecture_config=architecture_config,
            modules=modules,
            computation_rules=computation_rules,
            weight_classifier=self.classifier,
            model_type=model_type
        )

    def _detect_attention_type(self, config: Dict[str, Any]) -> str:
        """Detect attention type from config"""
        # Check for sliding window attention first
        if config.get('sliding_window') is not None:
            return "swa"

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
        else:
            # SwiGLU, GeGLU, and standard FFN all use the same dense activation formula
            # Use "dense" for non-MoE FFN types
            return "dense"

    def _detect_norm_type(self, config: Dict[str, Any]) -> str:
        """Detect normalization type from config"""
        norm_type = config.get('norm_type', config.get('rms_norm_eps'))
        if norm_type or 'rms_norm_eps' in config:
            return "rmsnorm"
        else:
            return "layernorm"

    def _classify_weights(self, weights_metadata: Dict[str, Dict[str, Any]],
                         model_name: str, model_type: str, arch_config: ArchitectureConfig,
                         ffn_moe_patterns: List[str] = None) -> Dict[str, Dict[str, WeightInfo]]:
        """Classify weights into module types

        Args:
            weights_metadata: Dictionary of weight names to metadata
            model_name: Model name (e.g., 'Qwen3-Coder-Next')
            model_type: HuggingFace model_type (e.g., 'qwen3')
            arch_config: Architecture configuration
            ffn_moe_patterns: List of patterns for matching MoE expert weights
        """

        # Step 0: Detect MoE expert count using ffn_moe patterns from weight_mapping_rules
        all_weight_names = list(weights_metadata.keys())
        expert_indices = set()
        expert_pattern_regex = None
        is_moe = False
        num_experts = 0
        matched_moe_weights = []  # Track weights that matched ffn_moe patterns

        if ffn_moe_patterns:
            for weight_name in all_weight_names:
                for pattern in ffn_moe_patterns:
                    # Try to match the pattern
                    if re.match(pattern, weight_name):
                        matched_moe_weights.append(weight_name)
                        # Match found, extract numeric indices from weight name
                        # Use all numeric indices, then take the max as expert count
                        indices = re.findall(r'\.(\d+)\.', weight_name)
                        for idx in indices:
                            expert_indices.add(int(idx))
                        # Build regex pattern for replacement if first match
                        if expert_pattern_regex is None:
                            # Try to find the expert index pattern in the weight name
                            # Supports: .experts.0., .experts0., expert_id_0., etc.
                            match = re.search(r'(\.experts?\d*\.|\w+_id_)\d+(\.)', weight_name)
                            if match:
                                expert_pattern_regex = match.group(1) + '{}' + match.group(2)
                        break

            # Calculate total expert count as max index + 1
            if expert_indices:
                num_experts = max(expert_indices) + 1
                is_moe = num_experts > 0

            # Validate: if MoE weights matched but expert_pattern_regex not found, try to infer from weight shape
            if matched_moe_weights and expert_pattern_regex is None:
                # Try to infer num_experts from weight shape (first dimension)
                # For models like gpt-oss where expert count is in shape [num_experts, ...]
                inferred_num_experts = None
                for weight_name in matched_moe_weights[:10]:  # Check first 10 weights
                    if weight_name in weights_metadata:
                        shape = weights_metadata[weight_name].get('shape', [])
                        if shape and len(shape) >= 1:
                            # First dimension could be num_experts
                            potential_expert_count = shape[0]
                            if potential_expert_count > 1:
                                inferred_num_experts = potential_expert_count
                                break

                if inferred_num_experts:
                    num_experts = inferred_num_experts
                    is_moe = True
                    # For gpt-oss style, expert indices are embedded in shape, not weight name
                    # Use a placeholder that won't match any expert pattern
                    expert_pattern_regex = ".placeholder.{}"
                else:
                    raise ValueError(
                        f"Failed to extract expert indices from MoE weights. "
                        f"Matched {len(matched_moe_weights)} weights but could not find expert index pattern. "
                        f"Example weight: '{matched_moe_weights[0]}'. "
                        f"Please ensure the weight_mapping_rules.yaml 'ffn_moe' patterns correctly match expert indices, "
                        f"or add an 'expert_pattern_regex' to help extract indices."
                    )

        # Step 1: Group weights by their base pattern
        # If MoE detected, also remove expert numbers during grouping
        pattern_groups = defaultdict(list)
        # Track layer indices for each module type
        module_layer_indices = defaultdict(set)

        for weight_name, metadata in weights_metadata.items():
            # Only remove layer numbers first
            base_pattern = re.sub(r'\.layers?\.\d+\.', '.layers.N.', weight_name)
            base_pattern = re.sub(r'model\.layers\.\d+', 'model.layers.N', base_pattern)

            # Extract layer index before pattern normalization
            layer_match = re.search(r'\.layers\.(\d+)\.', weight_name)
            if not layer_match:
                layer_match = re.search(r'model\.layers\.(\d+)', weight_name)
            layer_idx = int(layer_match.group(1)) if layer_match else 0

            # Classify weight to determine module type for layer tracking
            # Priority: model_name > model_type > generic
            module_type_for_layer = self.classifier.classify_weight(weight_name, model_name, model_type)

            # For MoE models, also remove expert numbers to consolidate
            # Match patterns like: .experts.0. or .experts0. or .block_sparse_moe.experts.0.
            if is_moe and expert_pattern_regex:
                # Replace expert index with N
                base_pattern = re.sub(r'(\.experts?\d*)\.\d+(\.)', r'\1.N\2', base_pattern)

            pattern_groups[base_pattern].append((weight_name, metadata))
            # Track layer indices for this module type
            module_layer_indices[module_type_for_layer].add(layer_idx)

        # Update architecture config if MoE detected
        if is_moe:
            arch_config.num_experts = num_experts
            arch_config.ffn_type = "moe"

        # Step 2: Process each group
        modules = {}

        for base_pattern, weight_list in pattern_groups.items():
            # Use the first weight to get metadata
            first_weight_name, first_metadata = weight_list[0]

            # Classify the base pattern
            # Priority: model_name > model_type > generic
            module_type = self.classifier.classify_weight(first_weight_name, model_name, model_type)

            if module_type not in modules:
                modules[module_type] = {}

            # Determine if this is a per-layer weight
            is_per_layer = '.layers.N.' in base_pattern or 'model.layers.N' in base_pattern

            # Check if this group has expert consolidation
            has_experts_in_pattern = is_moe and '.experts.N.' in base_pattern

            if is_per_layer and len(weight_list) > 1:
                # Use actual layer count from weight names, not arch_config.num_layers
                # This correctly handles cases where not all layers have MoE (e.g., DeepSeek-V3 has MoE only in layers 3-61)
                layers_count = len(module_layer_indices.get(module_type, set())) or len(weight_list)

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

    def _resolve_computation_rules(self, raw_rules: Dict[str, Any],
                                     attention_type: str, ffn_type: str) -> Dict[str, Any]:
        """Resolve computation rules by selecting the right formula based on attention/ffn type

        Args:
            raw_rules: Raw computation rules from weight_mapping_rules.yaml
            attention_type: Model's attention type
            ffn_type: Model's ffn type
        """
        resolved = {}

        # recommended_capacity_factor - direct copy
        if 'recommended_capacity_factor' in raw_rules:
            resolved['recommended_capacity_factor'] = raw_rules['recommended_capacity_factor']

        # kv_cache - can be a dict (by attention_type) or a string
        if 'kv_cache' in raw_rules:
            kv_cache = raw_rules['kv_cache']
            if isinstance(kv_cache, dict):
                # Select by attention_type
                resolved['kv_cache'] = kv_cache.get(attention_type, kv_cache.get('default', ''))
            else:
                resolved['kv_cache'] = kv_cache

        # activation - can be a dict (by ffn_type) or a string
        if 'activation' in raw_rules:
            activation = raw_rules['activation']
            if isinstance(activation, dict):
                # Select by ffn_type
                resolved['activation'] = activation.get(ffn_type, activation.get('default', ''))
            else:
                resolved['activation'] = activation

        # system_reserved_gb - direct copy
        if 'system_reserved_gb' in raw_rules:
            resolved['system_reserved_gb'] = raw_rules['system_reserved_gb']

        # gpu_util - direct copy
        if 'gpu_util' in raw_rules:
            resolved['gpu_util'] = raw_rules['gpu_util']

        return resolved
