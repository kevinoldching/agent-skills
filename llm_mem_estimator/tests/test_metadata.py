#!/usr/bin/env python3
"""
测试脚本：从 HuggingFace 获取模型元数据并分析
用于诊断配置生成问题
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
import re

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_mem_estimator import ModelDetector, WeightClassifier, ConfigLoader


def analyze_weight_patterns(weights_metadata):
    """分析权重名称模式"""
    print("\n" + "="*80)
    print("权重名称模式分析")
    print("="*80)

    # 按模式分组
    patterns = defaultdict(list)

    for weight_name in weights_metadata.keys():
        # 提取基础模式（去掉层号）
        base_pattern = re.sub(r'\.layers?\.\d+\.', '.layers.N.', weight_name)
        base_pattern = re.sub(r'model\.layers\.\d+', 'model.layers.N', base_pattern)
        patterns[base_pattern].append(weight_name)

    print(f"\n发现 {len(patterns)} 种不同的权重模式:")
    print(f"总权重数量: {len(weights_metadata)}")

    # 显示每种模式的示例
    for pattern, examples in sorted(patterns.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"\n模式: {pattern}")
        print(f"  出现次数: {len(examples)}")
        print(f"  示例: {examples[0]}")
        if len(examples) > 1:
            print(f"  最后一个: {examples[-1]}")


def show_weight_details(weights_metadata, limit=20):
    """显示权重详细信息"""
    print("\n" + "="*80)
    print(f"权重详细信息（前 {limit} 个）")
    print("="*80)

    for i, (name, info) in enumerate(list(weights_metadata.items())[:limit]):
        print(f"\n{i+1}. {name}")
        print(f"   Shape: {info['shape']}")
        print(f"   Dtype: {info['dtype']}")


def run_weight_classification(weights_metadata, model_type="unknown"):
    """测试权重分类"""
    print("\n" + "="*80)
    print("权重分类测试")
    print("="*80)

    # 加载分类规则
    script_dir = Path(__file__).parent.parent
    rules_path = script_dir / "configs" / "weight_mapping_rules.yaml"
    rules = ConfigLoader.load_weight_mapping_rules(str(rules_path))
    classifier = WeightClassifier(rules)

    # 统计分类结果
    classification_stats = defaultdict(list)

    for weight_name in weights_metadata.keys():
        module_type = classifier.classify_weight(weight_name, model_type)
        classification_stats[module_type].append(weight_name)

    print(f"\n分类统计:")
    for module_type, weights in sorted(classification_stats.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"\n{module_type}: {len(weights)} 个权重")
        # 显示前3个示例
        for example in weights[:3]:
            print(f"  - {example}")
        if len(weights) > 3:
            print(f"  ... 还有 {len(weights) - 3} 个")


def main():
    if len(sys.argv) < 2:
        print("用法: python test_metadata.py <model_name>")
        print("示例: python test_metadata.py openai/gpt-oss-120b")
        sys.exit(1)

    model_name = sys.argv[1]

    print("="*80)
    print(f"测试模型: {model_name}")
    print("="*80)

    # 1. 获取 config.json
    print("\n[1] 获取 config.json...")
    try:
        config = ModelDetector.detect_from_huggingface(model_name)
        print(f"✓ 成功获取 config.json")
        print(f"\n关键配置信息:")
        print(f"  - model_type: {config.get('model_type', 'unknown')}")
        print(f"  - hidden_size: {config.get('hidden_size')}")
        print(f"  - num_hidden_layers: {config.get('num_hidden_layers')}")
        print(f"  - num_attention_heads: {config.get('num_attention_heads')}")
        print(f"  - num_key_value_heads: {config.get('num_key_value_heads')}")
        print(f"  - vocab_size: {config.get('vocab_size')}")
        if 'num_experts' in config:
            print(f"  - num_experts: {config.get('num_experts')}")
        if 'quantization_config' in config:
            print(f"  - quantization: {config['quantization_config']}")
    except Exception as e:
        print(f"✗ 获取 config.json 失败: {e}")
        sys.exit(1)

    # 2. 获取权重元数据
    print("\n[2] 获取权重元数据（使用 get_safetensors_metadata）...")
    try:
        weights_metadata = ModelDetector.get_weights_metadata(model_name, is_local=False)
        print(f"✓ 成功获取权重元数据")
        print(f"  总权重数量: {len(weights_metadata)}")
    except Exception as e:
        print(f"✗ 获取权重元数据失败: {e}")
        sys.exit(1)

    # 3. 分析权重模式
    analyze_weight_patterns(weights_metadata)

    # 4. 显示权重详细信息
    show_weight_details(weights_metadata, limit=30)

    # 5. 测试权重分类
    test_weight_classification(weights_metadata, config.get('model_type', 'unknown'))

    # 6. 保存原始元数据到文件
    output_file = f"metadata_{model_name.replace('/', '_')}.json"
    print(f"\n[6] 保存原始元数据到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'config': config,
            'weights_metadata': weights_metadata
        }, f, indent=2, ensure_ascii=False)
    print(f"✓ 已保存到 {output_file}")

    print("\n" + "="*80)
    print("测试完成")
    print("="*80)


if __name__ == "__main__":
    main()
