#!/usr/bin/env python3
"""
Tests for Model Detection and Configuration Generation
Supports testing different models from HuggingFace

Usage:
    # Run all model tests (requires network)
    pytest tests/test_model_detection.py -v

    # Run specific model
    pytest tests/test_model_detection.py -k "Llama" -v

    # Generate outputs only (no validation)
    python tests/test_model_detection.py --model meta-llama/Llama-3.1-8B
"""

import pytest
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_mem_estimator import (
    ModelDetector, ConfigGenerator, WeightClassifier, ConfigLoader,
    MemoryEstimator
)
from llm_mem_estimator.model_config import ModelConfig, ArchitectureConfig
from tests.conftest import get_test_models_config


# 从 test_models.yaml 加载测试模型列表（HuggingFace 模型）
def _load_test_models() -> List[str]:
    """加载测试模型列表（只包含 HuggingFace 模型路径）"""
    try:
        config = get_test_models_config()
        models = config.get("models", [])
        result = []
        for m in models:
            model_type = m.get("type", "huggingface")
            if model_type == "huggingface":
                result.append(m.get("path", ""))
        return result
    except Exception:
        return ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"]


TEST_MODELS = _load_test_models()


def validate_model_config_structure(config: ModelConfig) -> Tuple[bool, List[str]]:
    """验证模型配置结构完整性"""
    errors = []

    # 检查 model_identity
    if not config.model_identity.name:
        errors.append("model_identity.name 不能为空")
    if config.model_identity.num_layers <= 0:
        errors.append(f"model_identity.num_layers={config.model_identity.num_layers} 应大于 0")

    # 检查 architecture_config
    arch = config.architecture_config
    if arch.hidden_size <= 0:
        errors.append(f"hidden_size={arch.hidden_size} 应大于 0")
    if arch.num_layers <= 0:
        errors.append(f"num_layers={arch.num_layers} 应大于 0")

    valid_attention_types = ["mha", "mqa", "gqa", "swa", "mla", "dsa", "gdn"]
    if arch.attention_type not in valid_attention_types:
        errors.append(f"attention_type={arch.attention_type} 不在有效类型中: {valid_attention_types}")

    valid_ffn_types = ["dense", "moe"]
    if arch.ffn_type not in valid_ffn_types:
        errors.append(f"ffn_type={arch.ffn_type} 不在有效类型中: {valid_ffn_types}")

    # 检查模块
    if len(config.modules) == 0:
        errors.append("modules 不能为空")

    # 检查 computation_rules
    if len(config.computation_rules) == 0:
        errors.append("computation_rules 不能为空")

    return len(errors) == 0, errors


def validate_numeric_ranges(config: ModelConfig) -> List[str]:
    """验证数值范围的合理性"""
    warnings = []
    arch = config.architecture_config

    # hidden_size 应该在合理范围 (256 ~ 50000)
    if arch.hidden_size < 256:
        warnings.append(f"hidden_size={arch.hidden_size} 可能过小")
    elif arch.hidden_size > 50000:
        warnings.append(f"hidden_size={arch.hidden_size} 可能过大")

    # vocab_size 检查
    if arch.vocab_size and arch.vocab_size < 1000:
        warnings.append(f"vocab_size={arch.vocab_size} 可能过小")
    if arch.vocab_size and arch.vocab_size > 200000:
        warnings.append(f"vocab_size={arch.vocab_size} 可能过大")

    # GQA/MQA/MLA 时 num_key_value_heads 应该存在
    if arch.attention_type in ["gqa", "mqa"]:
        if not arch.num_key_value_heads:
            warnings.append(f"{arch.attention_type} 需要 num_key_value_heads")
        elif arch.num_key_value_heads > arch.num_attention_heads:
            warnings.append(f"num_key_value_heads={arch.num_key_value_heads} 不应大于 num_attention_heads={arch.num_attention_heads}")

    # MLA 时需要特定字段
    if arch.attention_type == "mla":
        if not arch.kv_lora_rank:
            warnings.append("MLA 需要 kv_lora_rank")
        if not arch.qk_rope_head_dim:
            warnings.append("MLA 需要 qk_rope_head_dim")

    # MoE 时 num_experts 应该存在
    if arch.ffn_type == "moe":
        if not arch.num_experts:
            warnings.append("MoE 需要 num_experts")
        if not arch.num_experts_per_tok:
            warnings.append("MoE 需要 num_experts_per_tok")

    return warnings


def validate_via_calculation(config: ModelConfig) -> Tuple[bool, Dict[str, float], List[str]]:
    """通过计算验证配置有效性"""
    errors = []
    results = {}

    try:
        estimator = MemoryEstimator(config)

        # 使用典型参数进行计算
        result = estimator.estimate_memory(
            batch_size=1,
            seq_len=2048,
            kv_dtype="fp16",
            activation_dtype="fp16",
            tp=1, pp=1, dp=1, cp=1
        )

        results["weights_gb"] = result.weights_memory_gb
        results["kv_cache_gb"] = result.kv_cache_memory_gb
        results["activation_gb"] = result.activation_memory_gb
        results["total_gb"] = result.total_memory_gb

        # 基本检查
        if result.total_memory_gb <= 0:
            errors.append("total_memory_gb 应该大于 0")
        if result.weights_memory_gb <= 0:
            errors.append("weights_memory_gb 应该大于 0")

        # 粗略估算验证：1B 参数 ≈ 2GB (BF16)
        params_str = config.model_identity.total_params
        if params_str.isdigit():
            params_b = int(params_str)
            expected_min_gb = params_b * 1.5 / 1e9  # 最小估计 (BF16 + overhead)
            expected_max_gb = params_b * 4.0 / 1e9  # 最大估计

            if result.weights_memory_gb < expected_min_gb * 0.5:
                warnings = [f"权重内存 {result.weights_memory_gb:.2f}GB 远小于预期 {expected_min_gb:.2f}-{expected_max_gb:.2f}GB (参数 {params_b/1e9:.1f}B)"]
                return len(warnings) == 0, results, warnings
            elif result.weights_memory_gb > expected_max_gb * 2:
                warnings = [f"权重内存 {result.weights_memory_gb:.2f}GB 远大于预期 {expected_min_gb:.2f}-{expected_max_gb:.2f}GB (参数 {params_b/1e9:.1f}B)"]
                return len(warnings) == 0, results, warnings

    except Exception as e:
        errors.append(f"计算验证失败: {str(e)}")

    return len(errors) == 0, results, errors


def generate_weights_metadata_markdown(model_name: str, config_json: Dict[str, Any],
                                       weights_metadata: Dict[str, Any],
                                       classification_result: Dict[str, List[str]]) -> str:
    """生成权重元数据的 Markdown 报告"""
    lines = []
    model_short = model_name.replace("/", "_")

    lines.append(f"# {model_name} 权重元数据\n")
    lines.append(f"**总权重数量**: {len(weights_metadata)}\n")

    # 添加从 config.json 读取的 architecture config
    lines.append("\n## Architecture Config (from config.json)\n")
    lines.append("| 字段 | 值 |")
    lines.append("|------|-----|")

    # 核心字段
    core_fields = [
        ("model_type", "model_type"),
        ("hidden_size", "hidden_size"),
        ("num_hidden_layers", "num_hidden_layers"),
        ("vocab_size", "vocab_size"),
        ("num_attention_heads", "num_attention_heads"),
        ("num_key_value_heads", "num_key_value_heads"),
        ("intermediate_size", "intermediate_size"),
        ("norm_eps", "norm_eps"),
        ("num_experts", "num_experts"),
        ("num_experts_per_tok", "num_experts_per_tok"),
    ]

    for display_name, json_key in core_fields:
        if json_key in config_json:
            val = config_json[json_key]
            if val is not None:
                lines.append(f"| {display_name} | {val} |")

    # 其他字段
    other_fields = ["qk_rope_head_dim", "kv_lora_rank", "q_lora_rank", "v_head_dim", "qk_nope_head_dim"]
    for field in other_fields:
        if field in config_json and config_json[field] is not None:
            lines.append(f"| {field} | {config_json[field]} |")

    # 按分类统计
    lines.append("\n## 按模块类型统计\n")
    lines.append("| 模块类型 | 权重数量 |")
    lines.append("|----------|----------|")
    for module_type, weights in sorted(classification_result.items(), key=lambda x: len(x[1]), reverse=True):
        lines.append(f"| {module_type} | {len(weights)} |")

    # 按模式分组显示权重
    lines.append("\n## 权重详细列表\n")

    # 先显示前几个层，再显示后面的层（避免过长）
    weights_by_layer = defaultdict(list)
    for w_name in weights_metadata.keys():
        # 提取层号
        import re
        layer_match = re.search(r'\.layers?\.(\d+)\.', w_name)
        if layer_match:
            layer = int(layer_match.group(1))
        else:
            layer = -1  # 非层级的权重
        weights_by_layer[layer].append(w_name)

    # embedding 和最后的 norm
    if -1 in weights_by_layer:
        lines.append("### Embedding / Output Norm\n")
        for w_name in sorted(weights_by_layer[-1]):
            info = weights_metadata[w_name]
            shape_str = " x ".join(map(str, info.get("shape", [])))
            dtype = info.get("dtype", "unknown")
            lines.append(f"- `{w_name}`  {shape_str}  ({dtype})")
        lines.append("")

    # 层级权重 - 显示前2层和后2层
    layers = sorted([l for l in weights_by_layer.keys() if l >= 0])
    if layers:
        lines.append("### Layer Weights (示例)\n")

        # 前2层
        for layer in layers[:2]:
            lines.append(f"#### Layer {layer}\n")
            for w_name in sorted(weights_by_layer[layer]):
                info = weights_metadata[w_name]
                shape_str = " x ".join(map(str, info.get("shape", [])))
                dtype = info.get("dtype", "unknown")
                # 获取分类结果
                for mtype, wlist in classification_result.items():
                    if w_name in wlist:
                        lines.append(f"- `{w_name}`  {shape_str}  ({dtype})  → **{mtype}**")
                        break
            lines.append("")

        # 后2层
        for layer in layers[-2:]:
            lines.append(f"#### Layer {layer}\n")
            for w_name in sorted(weights_by_layer[layer]):
                info = weights_metadata[w_name]
                shape_str = " x ".join(map(str, info.get("shape", [])))
                dtype = info.get("dtype", "unknown")
                for mtype, wlist in classification_result.items():
                    if w_name in wlist:
                        lines.append(f"- `{w_name}`  {shape_str}  ({dtype})  → **{mtype}**")
                        break
            lines.append("")

    return "\n".join(lines)


class TestModelDetection:
    """测试不同模型的检测和配置生成"""

    @pytest.fixture
    def classifier(self):
        """Create WeightClassifier instance"""
        rules = ConfigLoader.load_weight_mapping_rules("configs/weight_mapping_rules.yaml")
        return WeightClassifier(rules)

    @pytest.fixture
    def output_dir(self):
        """Create and return output directory"""
        output_dir = Path(__file__).parent.parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        return output_dir

    @pytest.mark.parametrize("model_path", TEST_MODELS)
    def test_detect_model_config(self, model_path):
        """测试从 HuggingFace 获取模型配置"""
        config = ModelDetector.detect_from_huggingface(model_path)

        # 验证关键字段存在
        assert config.get("hidden_size") is not None, f"{model_path}: hidden_size 为空"
        assert config.get("num_hidden_layers") is not None, f"{model_path}: num_hidden_layers 为空"
        assert config.get("model_type") is not None, f"{model_path}: model_type 为空"

    @pytest.mark.parametrize("model_path", TEST_MODELS)
    def test_get_weights_metadata(self, model_path):
        """测试获取模型权重元数据"""
        weights = ModelDetector.get_weights_metadata(model_path, is_local=False)

        # 验证获取到了权重
        assert len(weights) > 0, f"{model_path}: 没有获取到权重"

        # 验证权重格式
        for name, info in list(weights.items())[:5]:
            assert "shape" in info, f"{model_path}: {name} 缺少 shape"
            assert "dtype" in info, f"{model_path}: {name} 缺少 dtype"

    @pytest.mark.parametrize("model_path", TEST_MODELS)
    def test_classify_real_model_weights(self, classifier, model_path):
        """测试真实模型的权重分类"""
        weights = ModelDetector.get_weights_metadata(model_path, is_local=False)

        # 统计分类结果
        classification_result = defaultdict(list)

        for w_name in weights.keys():
            result = classifier.classify_weight(w_name)
            classification_result[result].append(w_name)

        # 验证分类结果
        valid_types = ["embedding", "attention", "ffn_dense", "ffn_moe", "ffn_shared_expert", "norm", "others"]
        for module_type, weight_list in classification_result.items():
            assert module_type in valid_types, f"{model_path}: 无效的模块类型 {module_type}"

        # 输出统计
        print(f"\n{model_path} 权重分类统计:")
        for module_type, weight_list in sorted(classification_result.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"  {module_type}: {len(weight_list)}")

    @pytest.mark.parametrize("model_path", TEST_MODELS)
    def test_generate_config(self, classifier, model_path):
        """测试配置生成"""
        generator = ConfigGenerator(classifier)
        config = generator.generate_config(model_path, is_local=False)

        # 验证生成的结构
        assert config.model_identity.name is not None, f"{model_path}: name 为空"
        assert config.architecture_config.hidden_size > 0, f"{model_path}: hidden_size <= 0"
        assert config.architecture_config.num_layers > 0, f"{model_path}: num_layers <= 0"

    @pytest.mark.parametrize("model_path", TEST_MODELS)
    def test_validate_generated_config_structure(self, classifier, model_path):
        """验证生成的配置文件结构完整性"""
        generator = ConfigGenerator(classifier)
        config = generator.generate_config(model_path, is_local=False)

        is_valid, errors = validate_model_config_structure(config)
        assert is_valid, f"{model_path} 结构验证失败: {errors}"

    @pytest.mark.parametrize("model_path", TEST_MODELS)
    def test_validate_numeric_ranges(self, classifier, model_path):
        """验证数值范围的合理性"""
        generator = ConfigGenerator(classifier)
        config = generator.generate_config(model_path, is_local=False)

        warnings = validate_numeric_ranges(config)
        # 允许警告但不应该有严重问题
        print(f"\n{model_path} 数值范围警告: {warnings}")

    @pytest.mark.parametrize("model_path", TEST_MODELS)
    def test_validate_via_calculation(self, classifier, model_path):
        """通过计算验证配置有效性"""
        generator = ConfigGenerator(classifier)
        config = generator.generate_config(model_path, is_local=False)

        is_valid, results, messages = validate_via_calculation(config)

        print(f"\n{model_path} 计算验证结果:")
        print(f"  weights: {results.get('weights_gb', 0):.2f} GB")
        print(f"  kv_cache: {results.get('kv_cache_gb', 0):.2f} GB")
        print(f"  activation: {results.get('activation_gb', 0):.2f} GB")
        print(f"  total: {results.get('total_gb', 0):.2f} GB")

        # 检查是否有错误
        if not is_valid:
            print(f"  错误: {messages}")
        elif messages:
            print(f"  警告: {messages}")

        # 计算验证应该通过
        assert is_valid, f"{model_path} 计算验证失败: {messages}"


class TestModelDetectionOutputs:
    """生成测试输出文件（供人工确认）"""

    @pytest.fixture
    def classifier(self):
        rules = ConfigLoader.load_weight_mapping_rules("configs/weight_mapping_rules.yaml")
        return WeightClassifier(rules)

    @pytest.fixture
    def output_dir(self):
        output_dir = Path(__file__).parent.parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        return output_dir

    @pytest.mark.parametrize("model_path", TEST_MODELS)
    def test_generate_output_files(self, classifier, output_dir, model_path):
        """生成权重元数据和配置文件供人工确认"""
        model_short = model_path.replace("/", "_")

        # 1. 获取 config.json
        print(f"\n正在处理 {model_path}...")
        config_json = ModelDetector.detect_from_huggingface(model_path)

        # 2. 获取权重元数据
        weights_metadata = ModelDetector.get_weights_metadata(model_path, is_local=False)

        # 3. 分类权重
        classification_result = defaultdict(list)
        for w_name in weights_metadata.keys():
            result = classifier.classify_weight(w_name)
            classification_result[result].append(w_name)

        # 4. 生成权重元数据 Markdown
        model_short = model_path.replace("/", "_")
        md_content = generate_weights_metadata_markdown(model_path, config_json, weights_metadata, classification_result)
        md_file = output_dir / f"{model_short}_weights_metadata.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"  已保存: {md_file}")

        # 4. 生成配置文件 YAML
        generator = ConfigGenerator(classifier)
        config = generator.generate_config(model_path, is_local=False)
        yaml_file = output_dir / f"{model_short}_config.yaml"
        ConfigLoader.save_yaml_config(config, str(yaml_file))
        print(f"  已保存: {yaml_file}")


def main():
    """独立运行脚本生成指定模型的输出文件"""
    import argparse

    parser = argparse.ArgumentParser(description="生成模型权重元数据和配置文件")
    parser.add_argument("--model", type=str, required=True, help="模型名称 (如 Qwen/Qwen2.5-0.5B)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="输出目录")
    args = parser.parse_args()

    model_path = args.model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"处理模型: {model_path}")
    print("=" * 60)

    # 加载分类器
    rules = ConfigLoader.load_weight_mapping_rules("configs/weight_mapping_rules.yaml")
    classifier = WeightClassifier(rules)

    # 获取 config.json
    print("1. 获取 config.json...")
    config_json = ModelDetector.detect_from_huggingface(model_path)
    print(f"   model_type: {config_json.get('model_type')}")
    print(f"   hidden_size: {config_json.get('hidden_size')}")
    print(f"   num_layers: {config_json.get('num_hidden_layers')}")

    # 获取权重元数据
    print("2. 获取权重元数据...")
    weights_metadata = ModelDetector.get_weights_metadata(model_path, is_local=False)
    print(f"   总权重数量: {len(weights_metadata)}")

    # 分类权重
    print("3. 分类权重...")
    classification_result = defaultdict(list)
    for w_name in weights_metadata.keys():
        result = classifier.classify_weight(w_name)
        classification_result[result].append(w_name)

    print("   分类统计:")
    for module_type, weight_list in sorted(classification_result.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"     {module_type}: {len(weight_list)}")

    # 生成权重元数据 Markdown
    print("4. 生成权重元数据 Markdown...")
    model_short = model_path.replace("/", "_")
    md_content = generate_weights_metadata_markdown(model_path, config_json, weights_metadata, classification_result)
    md_file = output_dir / f"{model_short}_weights_metadata.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"   已保存: {md_file}")

    # 生成配置文件 YAML
    print("5. 生成配置文件 YAML...")
    generator = ConfigGenerator(classifier)
    config = generator.generate_config(model_path, is_local=False)
    yaml_file = output_dir / f"{model_short}_config.yaml"
    ConfigLoader.save_yaml_config(config, str(yaml_file))
    print(f"   已保存: {yaml_file}")

    # 验证配置
    print("6. 验证配置...")
    is_valid, errors = validate_model_config_structure(config)
    print(f"   结构验证: {'通过' if is_valid else '失败'}")
    if errors:
        for err in errors:
            print(f"     - {err}")

    warnings = validate_numeric_ranges(config)
    if warnings:
        print(f"   数值警告:")
        for warn in warnings:
            print(f"     - {warn}")

    is_valid_calc, results, messages = validate_via_calculation(config)
    if results:
        print(f"   计算结果:")
        print(f"     权重: {results.get('weights_gb', 0):.2f} GB")
        print(f"     KV Cache: {results.get('kv_cache_gb', 0):.2f} GB")
        print(f"     激活值: {results.get('activation_gb', 0):.2f} GB")
        print(f"     总计: {results.get('total_gb', 0):.2f} GB")

    print("\n完成!")
    print(f"\n请查看以下文件进行人工确认:")
    print(f"  - {md_file}")
    print(f"  - {yaml_file}")


if __name__ == "__main__":
    main()
