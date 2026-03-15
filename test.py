import re
from collections import defaultdict
from huggingface_hub import get_safetensors_metadata

def analyze_model_weights(repo_id, token=None):
    print(f"📡 正在深度解析 {repo_id} 的权重结构...")
    try:
        metadata = get_safetensors_metadata(repo_id, token=token)
        
        # 1. 获取所有 Tensor 的维度映射
        all_tensors = {}
        f_meta = metadata.files_metadata
        for tensor_name, file_name in metadata.weight_map.items():
            file_obj = f_meta[file_name] if isinstance(f_meta, dict) else next(f for f in f_meta if getattr(f, 'file_name', '') == file_name)
            if tensor_name in file_obj.tensors:
                all_tensors[tensor_name] = file_obj.tensors[tensor_name]

        # 2. 统计归类 (使用正则过滤掉层号和专家号)
        # 结果结构: { "权重功能名": {"shape": [], "count": 1, "is_expert": False} }
        weight_archive = {}
        expert_counts = defaultdict(set) # 用于统计每层有多少专家

        for name, info in all_tensors.items():
            # 提取核心名称，去除层号 (.layers.0.) 和 专家号 (.experts.1.)
            # 例子: model.layers.1.mlp.experts.15.down_proj.weight -> mlp.experts.down_proj
            core_name = re.sub(r'model\.layers\.\d+\.', '', name)
            
            # 检测是否是专家权重并统计
            expert_match = re.search(r'experts\.(\d+)\.', core_name)
            is_expert = False
            if expert_match:
                is_expert = True
                expert_id = expert_match.group(1)
                # 记录这层有哪些专家编号
                layer_id = re.search(r'layers\.(\d+)\.', name).group(1)
                expert_counts[layer_id].add(expert_id)
                # 简化 core_name 用于归类展示
                core_name = re.sub(r'experts\.\d+\.', 'experts.[N].', core_name)

            if core_name not in weight_archive:
                weight_archive[core_name] = {
                    "shape": info.shape,
                    "dtype": info.dtype,
                    "is_expert": is_expert,
                    "total_count": 0
                }
            weight_archive[core_name]["total_count"] += 1

        # 3. 统计专家总数 (取各层专家的平均值或最大值)
        avg_experts = max([len(v) for v in expert_counts.values()]) if expert_counts else 0

        return {
            "total_params": sum(metadata.parameter_count.values()) if isinstance(metadata.parameter_count, dict) else metadata.parameter_count,
            "num_layers": len(expert_counts) if expert_counts else 0, # 或者通过正则重新统计
            "expert_count_per_layer": avg_experts,
            "archive": weight_archive
        }

    except Exception as e:
        print(f"❌ 解析失败: {e}")
        return None

# --- 运行分析 ---
info = analyze_model_weights("Qwen/Qwen3-Coder-Next", token="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")

if info:
    print(f"\n" + "="*50)
    print(f"Qwen3-Coder-Next 模型架构概览")
    print(f"总参数量: {info['total_params']/1e9:.2f} B")
    print(f"每层专家数: {info['expert_count_per_layer']}")
    print("="*50)
    print(f"{'权重组件名':<45} | {'形状 (Shape)':<20} | {'数据类型 (Dtype)':<20} | {'总数'}")
    print("-" * 80)
    
    # 分类输出：基础权重 vs 专家权重
    for name, detail in sorted(info['archive'].items()):
        shape_str = str(detail['shape'])
        dtype_str =str(detail['dtype'])
        print(f"{name:<45} | {shape_str:<20} |{dtype_str:<20}| {detail['total_count']}")