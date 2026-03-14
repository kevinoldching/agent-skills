#!/usr/bin/env python3
"""
Context Length Estimator (V2)
根据模型参数和芯片规格计算可支持的上下文长度
支持细粒度模块分解和多种并行策略
"""

import json
import argparse
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

# ============ 量化配置 ============
# 格式: A{activation}W{weight}C{kv_cache}
# A4=INT4, A8=INT8, W4=INT4, W8=INT8, C4=INT4, C8=INT8, C16=FP16
QUANT_CONFIG = {
    # 激活量化 (bytes per element)
    "A4": 0.5,  # INT4
    "A8": 1.0,  # INT8
    "A16": 2.0, # FP16
    "A32": 4.0, # FP32
    # 权重量化 (bytes per param)
    "W4": 0.5,  # INT4
    "W8": 1.0,  # INT8
    "W16": 2.0, # FP16
    "W32": 4.0, # FP32
    # KV Cache 量化 (bytes per element)
    "C4": 0.5,  # INT4
    "C8": 1.0,  # INT8
    "C16": 2.0, # FP16
    "C32": 4.0, # FP32
}

# 预定义量化组合
QUANT_PRESETS = {
    # 完整格式
    "A8W8C16": {"activation": "A8", "weight": "W8", "kv_cache": "C16"},
    "A8W8C8":  {"activation": "A8", "weight": "W8", "kv_cache": "C8"},
    "A8W4C16": {"activation": "A8", "weight": "W4", "kv_cache": "C16"},
    "A8W4C8":  {"activation": "A8", "weight": "W4", "kv_cache": "C8"},
    "A4W4C16": {"activation": "A4", "weight": "W4", "kv_cache": "C16"},
    "A4W4C8":  {"activation": "A4", "weight": "W4", "kv_cache": "C8"},
    # 简单格式映射
    "fp16":    {"activation": "A16", "weight": "W16", "kv_cache": "C16"},
    "fp32":    {"activation": "A32", "weight": "W32", "kv_cache": "C32"},
    "int8":    {"activation": "A8", "weight": "W8", "kv_cache": "C8"},
    "int4":    {"activation": "A4", "weight": "W4", "kv_cache": "C8"},
    "fp8":     {"activation": "A8", "weight": "W8", "kv_cache": "C8"},
    # 默认FP16
    "default": {"activation": "A16", "weight": "W16", "kv_cache": "C16"},
}


def get_quant_bytes(quant_str: str) -> Tuple[float, float, float]:
    """获取激活、权重、KV Cache的字节数"""
    if quant_str in QUANT_PRESETS:
        cfg = QUANT_PRESETS[quant_str]
        act_bytes = QUANT_CONFIG.get(cfg["activation"], 2.0)
        weight_bytes = QUANT_CONFIG.get(cfg["weight"], 2.0)
        kv_bytes = QUANT_CONFIG.get(cfg["kv_cache"], 2.0)
    else:
        # 兼容旧格式
        act_bytes = QUANT_CONFIG.get(quant_str.upper().replace("FP16", "W16").replace("INT8", "W8").replace("INT4", "W4")[0:2], 2.0)
        weight_bytes = act_bytes
        kv_bytes = 2.0
    return act_bytes, weight_bytes, kv_bytes


# ============ 模型细粒度结构数据库 ============
# 每个模块的参数数量计算基于: in_features * out_features
# 对于 Transformer: hidden_size * (hidden_size * n_heads) 等

MODEL_ARCHITECTURE = {
    # ============ LLaMA 系列 ============
    "llama-7b": {
        "architecture": "llama",
        "hidden_size": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "head_dim": 128,
        "vocab_size": 32000,
        "intermediate_size": 11008,  # FFN 中间层
        "moe": False,
        # 细粒度模块参数 (单位: M)
        "modules": {
            "embedding": {"params": 32000 * 4096, "type": "embedding"},
            "lm_head": {"params": 32000 * 4096, "type": "output"},
            "input_layernorm": {"params": 4096, "type": "layernorm", "num": 1},
            "post_attention_layernorm": {"params": 4096, "type": "layernorm", "num": 32},
            # Attention
            "q_proj": {"params": 4096 * 4096, "type": "linear", "num": 32},
            "k_proj": {"params": 4096 * 4096, "type": "linear", "num": 32},
            "v_proj": {"params": 4096 * 4096, "type": "linear", "num": 32},
            "o_proj": {"params": 4096 * 4096, "type": "linear", "num": 32},
            # MLP
            "gate_proj": {"params": 4096 * 11008, "type": "linear", "num": 32},
            "up_proj": {"params": 4096 * 11008, "type": "linear", "num": 32},
            "down_proj": {"params": 11008 * 4096, "type": "linear", "num": 32},
        }
    },
    "llama-70b": {
        "architecture": "llama",
        "hidden_size": 8192,
        "num_layers": 80,
        "num_heads": 64,
        "head_dim": 128,
        "vocab_size": 32000,
        "intermediate_size": 28672,
        "moe": False,
        "modules": {
            "embedding": {"params": 32000 * 8192, "type": "embedding"},
            "lm_head": {"params": 32000 * 8192, "type": "output"},
            "input_layernorm": {"params": 8192, "type": "layernorm", "num": 1},
            "post_attention_layernorm": {"params": 8192, "type": "layernorm", "num": 80},
            "q_proj": {"params": 8192 * 8192, "type": "linear", "num": 80},
            "k_proj": {"params": 8192 * 8192, "type": "linear", "num": 80},
            "v_proj": {"params": 8192 * 8192, "type": "linear", "num": 80},
            "o_proj": {"params": 8192 * 8192, "type": "linear", "num": 80},
            "gate_proj": {"params": 8192 * 28672, "type": "linear", "num": 80},
            "up_proj": {"params": 8192 * 28672, "type": "linear", "num": 80},
            "down_proj": {"params": 28672 * 8192, "type": "linear", "num": 80},
        }
    },
    "llama3-70b": {
        "architecture": "llama3",
        "hidden_size": 8192,
        "num_layers": 80,
        "num_heads": 64,
        "head_dim": 128,
        "vocab_size": 128256,
        "intermediate_size": 28672,
        "moe": False,
        "modules": {
            "embedding": {"params": 128256 * 8192, "type": "embedding"},
            "lm_head": {"params": 128256 * 8192, "type": "output"},
            "input_layernorm": {"params": 8192, "type": "layernorm", "num": 1},
            "post_attention_layernorm": {"params": 8192, "type": "layernorm", "num": 80},
            "q_proj": {"params": 8192 * 8192, "type": "linear", "num": 80},
            "k_proj": {"params": 8192 * 8192, "type": "linear", "num": 80},
            "v_proj": {"params": 8192 * 8192, "type": "linear", "num": 80},
            "o_proj": {"params": 8192 * 8192, "type": "linear", "num": 80},
            "gate_proj": {"params": 8192 * 28672, "type": "linear", "num": 80},
            "up_proj": {"params": 8192 * 28672, "type": "linear", "num": 80},
            "down_proj": {"params": 28672 * 8192, "type": "linear", "num": 80},
        }
    },
    # ============ Qwen 系列 ============
    "qwen-72b": {
        "architecture": "qwen",
        "hidden_size": 8192,
        "num_layers": 80,
        "num_heads": 64,
        "head_dim": 128,
        "vocab_size": 151936,
        "intermediate_size": 22016,
        "moe": False,
        "modules": {
            "embedding": {"params": 151936 * 8192, "type": "embedding"},
            "lm_head": {"params": 151936 * 8192, "type": "output"},
            "input_layernorm": {"params": 8192, "type": "layernorm", "num": 1},
            "post_attention_layernorm": {"params": 8192, "type": "layernorm", "num": 80},
            "q_proj": {"params": 8192 * 8192, "type": "linear", "num": 80},
            "k_proj": {"params": 8192 * 1024, "type": "linear", "num": 80},  # KV 维度不同
            "v_proj": {"params": 8192 * 1024, "type": "linear", "num": 80},
            "o_proj": {"params": 8192 * 8192, "type": "linear", "num": 80},
            "gate_proj": {"params": 8192 * 22016, "type": "linear", "num": 80},
            "up_proj": {"params": 8192 * 22016, "type": "linear", "num": 80},
            "down_proj": {"params": 22016 * 8192, "type": "linear", "num": 80},
        }
    },
    "qwen3-72b": {
        "architecture": "qwen3",
        "hidden_size": 8192,
        "num_layers": 80,
        "num_heads": 64,
        "head_dim": 128,
        "vocab_size": 151936,
        "intermediate_size": 22016,
        "moe": False,
        "modules": {
            "embedding": {"params": 151936 * 8192, "type": "embedding"},
            "lm_head": {"params": 151936 * 8192, "type": "output"},
            "input_layernorm": {"params": 8192, "type": "layernorm", "num": 1},
            "post_attention_layernorm": {"params": 8192, "type": "layernorm", "num": 80},
            "q_proj": {"params": 8192 * 8192, "type": "linear", "num": 80},
            "k_proj": {"params": 8192 * 1024, "type": "linear", "num": 80},
            "v_proj": {"params": 8192 * 1024, "type": "linear", "num": 80},
            "o_proj": {"params": 8192 * 8192, "type": "linear", "num": 80},
            "gate_proj": {"params": 8192 * 22016, "type": "linear", "num": 80},
            "up_proj": {"params": 8192 * 22016, "type": "linear", "num": 80},
            "down_proj": {"params": 22016 * 8192, "type": "linear", "num": 80},
        }
    },
    # ============ Mistral ============
    "mistral-7b": {
        "architecture": "mistral",
        "hidden_size": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "head_dim": 128,
        "vocab_size": 32000,
        "intermediate_size": 14336,
        "moe": False,
        "modules": {
            "embedding": {"params": 32000 * 4096, "type": "embedding"},
            "lm_head": {"params": 32000 * 4096, "type": "output"},
            "input_layernorm": {"params": 4096, "type": "layernorm", "num": 1},
            "post_attention_layernorm": {"params": 4096, "type": "layernorm", "num": 32},
            "q_proj": {"params": 4096 * 4096, "type": "linear", "num": 32},
            "k_proj": {"params": 4096 * 4096, "type": "linear", "num": 32},
            "v_proj": {"params": 4096 * 4096, "type": "linear", "num": 32},
            "o_proj": {"params": 4096 * 4096, "type": "linear", "num": 32},
            "gate_proj": {"params": 4096 * 14336, "type": "linear", "num": 32},
            "up_proj": {"params": 4096 * 14336, "type": "linear", "num": 32},
            "down_proj": {"params": 14336 * 4096, "type": "linear", "num": 32},
        }
    },
    # ============ Llama3 8B ============
    "llama3-8b": {
        "architecture": "llama3",
        "hidden_size": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "head_dim": 128,
        "vocab_size": 128256,
        "intermediate_size": 14336,
        "moe": False,
        "modules": {
            "embedding": {"params": 128256 * 4096, "type": "embedding"},
            "lm_head": {"params": 128256 * 4096, "type": "output"},
            "input_layernorm": {"params": 4096, "type": "layernorm", "num": 1},
            "post_attention_layernorm": {"params": 4096, "type": "layernorm", "num": 32},
            "q_proj": {"params": 4096 * 4096, "type": "linear", "num": 32},
            "k_proj": {"params": 4096 * 4096, "type": "linear", "num": 32},
            "v_proj": {"params": 4096 * 4096, "type": "linear", "num": 32},
            "o_proj": {"params": 4096 * 4096, "type": "linear", "num": 32},
            "gate_proj": {"params": 4096 * 14336, "type": "linear", "num": 32},
            "up_proj": {"params": 4096 * 14336, "type": "linear", "num": 32},
            "down_proj": {"params": 14336 * 4096, "type": "linear", "num": 32},
        }
    },
    # ============ Qwen2.5 7B ============
    "qwen2.5-7b": {
        "architecture": "qwen2",
        "hidden_size": 3584,
        "num_layers": 28,
        "num_heads": 28,
        "head_dim": 128,
        "vocab_size": 151936,
        "intermediate_size": 18944,
        "moe": False,
        "modules": {
            "embedding": {"params": 151936 * 3584, "type": "embedding"},
            "lm_head": {"params": 151936 * 3584, "type": "output"},
            "input_layernorm": {"params": 3584, "type": "layernorm", "num": 1},
            "post_attention_layernorm": {"params": 3584, "type": "layernorm", "num": 28},
            "q_proj": {"params": 3584 * 3584, "type": "linear", "num": 28},
            "k_proj": {"params": 3584 * 1024, "type": "linear", "num": 28},
            "v_proj": {"params": 3584 * 1024, "type": "linear", "num": 28},
            "o_proj": {"params": 3584 * 3584, "type": "linear", "num": 28},
            "gate_proj": {"params": 3584 * 18944, "type": "linear", "num": 28},
            "up_proj": {"params": 3584 * 18944, "type": "linear", "num": 28},
            "down_proj": {"params": 18944 * 3584, "type": "linear", "num": 28},
        }
    },
    # ============ GLM-4-9B ============
    "glm-4-9b": {
        "architecture": "glm",
        "hidden_size": 4096,
        "num_layers": 40,
        "num_heads": 32,
        "head_dim": 128,
        "vocab_size": 151552,
        "intermediate_size": 13696,
        "moe": False,
        "modules": {
            "embedding": {"params": 151552 * 4096, "type": "embedding"},
            "lm_head": {"params": 151552 * 4096, "type": "output"},
            "input_layernorm": {"params": 4096, "type": "layernorm", "num": 1},
            "post_attention_layernorm": {"params": 4096, "type": "layernorm", "num": 40},
            "q_proj": {"params": 4096 * 4096, "type": "linear", "num": 40},
            "k_proj": {"params": 4096 * 4096, "type": "linear", "num": 40},
            "v_proj": {"params": 4096 * 4096, "type": "linear", "num": 40},
            "o_proj": {"params": 4096 * 4096, "type": "linear", "num": 40},
            "gate_proj": {"params": 4096 * 13696, "type": "linear", "num": 40},
            "up_proj": {"params": 4096 * 13696, "type": "linear", "num": 40},
            "down_proj": {"params": 13696 * 4096, "type": "linear", "num": 40},
        }
    },
    # ============ Mixtral (MoE) ============
    "mixtral-8x7b": {
        "architecture": "mixtral",
        "hidden_size": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "head_dim": 128,
        "vocab_size": 32000,
        "intermediate_size": 14336,
        "moe": True,
        "num_experts": 8,
        "active_experts": 2,
        "shared_experts": 0,  # 纯MoE，无共享专家
        "modules": {
            "embedding": {"params": 32000 * 4096, "type": "embedding"},
            "lm_head": {"params": 32000 * 4096, "type": "output"},
            "input_layernorm": {"params": 4096, "type": "layernorm", "num": 1},
            "post_attention_layernorm": {"params": 4096, "type": "layernorm", "num": 32},
            "q_proj": {"params": 4096 * 4096, "type": "linear", "num": 32},
            "k_proj": {"params": 4096 * 4096, "type": "linear", "num": 32},
            "v_proj": {"params": 4096 * 4096, "type": "linear", "num": 32},
            "o_proj": {"params": 4096 * 4096, "type": "linear", "num": 32},
            # MoE FFN (8 experts) - params is for ONE expert, num_experts=8 means 8 experts total
            "gate_proj": {"params": 4096 * 14336, "type": "moe", "num_experts": 8, "active_experts": 2, "shared_experts": 0, "num": 1},  # num=1 because params is PER expert
            "up_proj": {"params": 4096 * 14336, "type": "moe", "num_experts": 8, "active_experts": 2, "shared_experts": 0, "num": 1},
            "down_proj": {"params": 14336 * 4096, "type": "moe", "num_experts": 8, "active_experts": 2, "shared_experts": 0, "num": 1},
        }
    },
    # ============ GPT-Oss 120B (MoE) ============
    # 参数来源: config.json
    # 特点: 128 experts, 激活4个, sliding window attention + full attention 交替
    "gpt-oss-120b": {
        "architecture": "gpt_oss",
        "hidden_size": 2880,
        "num_layers": 36,
        "num_heads": 64,
        "head_dim": 64,
        "kv_dim": 512,  # num_key_value_heads (8) * head_dim (64)
        "vocab_size": 201088,
        "intermediate_size": 2880,  # SwiGLU: intermediate = hidden_size
        "moe": True,
        "num_experts": 128,
        "active_experts": 4,
        "shared_experts": 0,
        "sliding_window": 128,
        "modules": {
            "embedding": {"params": 201088 * 2880, "type": "embedding"},
            "lm_head": {"params": 201088 * 2880, "type": "output"},
            "input_layernorm": {"params": 2880, "type": "layernorm", "num": 1},
            "post_attention_layernorm": {"params": 2880, "type": "layernorm", "num": 36},
            "q_proj": {"params": 2880 * 2880, "type": "linear", "num": 36},
            "k_proj": {"params": 2880 * 512, "type": "linear", "num": 36},  # hidden_size * kv_dim
            "v_proj": {"params": 2880 * 512, "type": "linear", "num": 36},
            "o_proj": {"params": 2880 * 2880, "type": "linear", "num": 36},
            # MoE FFN (128 experts) - params is for ONE expert
            "gate_proj": {"params": 2880 * 2880, "type": "moe", "num_experts": 128, "active_experts": 4, "shared_experts": 0, "num": 1},
            "up_proj": {"params": 2880 * 2880, "type": "moe", "num_experts": 128, "active_experts": 4, "shared_experts": 0, "num": 1},
            "down_proj": {"params": 2880 * 2880, "type": "moe", "num_experts": 128, "active_experts": 4, "shared_experts": 0, "num": 1},
        }
    },
    # ============ DeepSeek V3 (MoE) ============
    "deepseek-v3": {
        "architecture": "deepseek_v3",
        "hidden_size": 6144,
        "num_layers": 60,
        "num_heads": 48,
        "head_dim": 128,
        "vocab_size": 200000,
        "intermediate_size": 12288,
        "moe": True,
        "num_experts": 256,
        "active_experts": 8,
        "shared_experts": 1,  # 1个共享专家 + 8个路由专家 = 9个激活专家
        "modules": {
            "embedding": {"params": 200000 * 6144, "type": "embedding"},
            "lm_head": {"params": 200000 * 6144, "type": "output"},
            "input_layernorm": {"params": 6144, "type": "layernorm", "num": 1},
            "post_attention_layernorm": {"params": 6144, "type": "layernorm", "num": 60},
            "q_proj": {"params": 6144 * 6144, "type": "linear", "num": 60},
            "k_proj": {"params": 6144 * 1024, "type": "linear", "num": 60},
            "v_proj": {"params": 6144 * 1024, "type": "linear", "num": 60},
            "o_proj": {"params": 6144 * 6144, "type": "linear", "num": 60},
            # MoE FFN - params is for ONE expert
            "gate_proj": {"params": 6144 * 12288, "type": "moe", "num_experts": 256, "active_experts": 8, "shared_experts": 1, "num": 1},
            "up_proj": {"params": 6144 * 12288, "type": "moe", "num_experts": 256, "active_experts": 8, "shared_experts": 1, "num": 1},
            "down_proj": {"params": 12288 * 6144, "type": "moe", "num_experts": 256, "active_experts": 8, "shared_experts": 1, "num": 1},
        }
    },
}


def load_model_from_config(config_path: str) -> Optional[Dict]:
    """
    从 HuggingFace config.json 加载模型参数
    """
    import json
    import os

    if not os.path.exists(config_path):
        return None

    try:
        with open(config_path, 'r') as f:
            config = f.read()

        # 处理可能存在的 pickle 标记或二进制内容
        if config.startswith('{'):
            config_json = json.loads(config)
        else:
            return None

        # 提取模型参数
        hidden_size = config_json.get('hidden_size', 4096)
        num_layers = config_json.get('num_hidden_layers', config_json.get('n_layers', 32))
        num_heads = config_json.get('num_attention_heads', config_json.get('n_heads', 32))
        head_dim = config_json.get('head_dim', hidden_size // num_heads if num_heads else 128)
        vocab_size = config_json.get('vocab_size', 32000)
        intermediate_size = config_json.get('intermediate_size', hidden_size * 4)

        # 判断是否是 MoE
        is_moe = config_json.get('moe', False) or 'num_experts' in config_json
        num_experts = config_json.get('num_experts', 0)
        active_experts = config_json.get('num_active_experts', config_json.get('active_experts', 2))
        shared_experts = config_json.get('shared_experts', 0)

        # 计算 KV 维度 (Qwen 等模型 KV 维度可能不同)
        kv_dim = head_dim * num_heads
        if 'kv_dim' in config_json:
            kv_dim = config_json['kv_dim']

        arch = {
            "architecture": config_json.get('model_type', 'unknown'),
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "vocab_size": vocab_size,
            "intermediate_size": intermediate_size,
            "moe": is_moe,
            "num_experts": num_experts,
            "active_experts": active_experts,
            "shared_experts": shared_experts,
            "kv_dim": kv_dim,
            "modules": {}
        }

        # 自动生成模块参数
        modules = {
            "embedding": {"params": vocab_size * hidden_size, "type": "embedding"},
            "lm_head": {"params": vocab_size * hidden_size, "type": "output"},
            "input_layernorm": {"params": hidden_size, "type": "layernorm", "num": 1},
            "post_attention_layernorm": {"params": hidden_size, "type": "layernorm", "num": num_layers},
            "q_proj": {"params": hidden_size * hidden_size, "type": "linear", "num": num_layers},
            "k_proj": {"params": hidden_size * kv_dim, "type": "linear", "num": num_layers},
            "v_proj": {"params": hidden_size * kv_dim, "type": "linear", "num": num_layers},
            "o_proj": {"params": hidden_size * hidden_size, "type": "linear", "num": num_layers},
        }

        if is_moe and num_experts > 0:
            modules["gate_proj"] = {"params": hidden_size * intermediate_size, "type": "moe",
                                   "num_experts": num_experts, "num": 1,
                                   "active_experts": active_experts, "shared_experts": shared_experts}
            modules["up_proj"] = {"params": hidden_size * intermediate_size, "type": "moe",
                                 "num_experts": num_experts, "num": 1,
                                 "active_experts": active_experts, "shared_experts": shared_experts}
            modules["down_proj"] = {"params": intermediate_size * hidden_size, "type": "moe",
                                   "num_experts": num_experts, "num": 1,
                                   "active_experts": active_experts, "shared_experts": shared_experts}
        else:
            modules["gate_proj"] = {"params": hidden_size * intermediate_size, "type": "linear", "num": num_layers}
            modules["up_proj"] = {"params": hidden_size * intermediate_size, "type": "linear", "num": num_layers}
            modules["down_proj"] = {"params": intermediate_size * hidden_size, "type": "linear", "num": num_layers}

        arch["modules"] = modules
        return arch

    except Exception as e:
        print(f"Warning: Failed to parse config.json: {e}")
        return None


def parse_strategy_string(strategy_str: str) -> Tuple[List[Dict], str]:
    """
    解析策略字符串，支持多个策略用逗号分隔:
    - "TP8EP8" - 单个策略，默认FP16
    - "TP8EP8+INT4" - 单个策略+量化
    - "TP8,TP16EP16" - 多个策略，默认FP16
    - "TP8+INT4,TP16EP16+FP8" - 多个策略+各自量化
    返回: (策略列表, 默认量化配置字符串)
    """
    if not strategy_str:
        return [], "fp16"

    import re

    # 用逗号分割多个策略
    strategy_parts = [s.strip() for s in strategy_str.split(',')]

    strategies = []
    default_quant = "fp16"  # 默认量化

    for idx, part in enumerate(strategy_parts):
        # 检查是否包含量化配置 (用 + 分隔)
        quant_str = "fp16"
        if '+' in part:
            strategy_part, quant_part = part.split('+', 1)
            # 检查是否是量化配置
            if 'W' in quant_part or 'A' in quant_part or 'C' in quant_part:
                quant_str = quant_part
            else:
                quant_map = {
                    "INT4": "int4", "INT8": "int8", "FP8": "fp8", "FP16": "fp16", "FP32": "fp32"
                }
                quant_str = quant_map.get(quant_part, "fp16")
            # 只有第一个策略的量化作为默认
            if idx == 0:
                default_quant = quant_str
        else:
            strategy_part = part

        # 解析策略部分
        matches = re.findall(r'([A-Z]+)(\d+)', strategy_part.upper())

        strategy_dict = {"tp": 1, "pp": 1, "dp": 1, "ep": 1, "sp": False, "cp": 1}

        for prefix, num in matches:
            num = int(num)
            if prefix == "TP":
                strategy_dict["tp"] = num
            elif prefix == "PP":
                strategy_dict["pp"] = num
            elif prefix == "DP":
                strategy_dict["dp"] = num
            elif prefix == "EP":
                strategy_dict["ep"] = num
            elif prefix == "SP":
                strategy_dict["sp"] = True
            elif prefix == "CP":
                strategy_dict["cp"] = num

        # 生成名称
        name_parts = []
        if strategy_dict["tp"] > 1:
            name_parts.append(f"TP{strategy_dict['tp']}")
        if strategy_dict["ep"] > 1:
            name_parts.append(f"EP{strategy_dict['ep']}")
        if strategy_dict["dp"] > 1:
            name_parts.append(f"DP{strategy_dict['dp']}")
        if strategy_dict["sp"]:
            name_parts.append("SP")
        if strategy_dict["cp"] > 1:
            name_parts.append(f"CP{strategy_dict['cp']}")

        strategy_dict["name"] = "".join(name_parts) if name_parts else "SP=1"
        strategies.append(strategy_dict)

    return strategies, default_quant


def get_model_architecture(model_name: str) -> Optional[Dict]:
    """获取模型的细粒度架构"""
    model_key = model_name.lower().replace("-", "").replace("_", "").replace(" ", "")
    for key, data in MODEL_ARCHITECTURE.items():
        if key.replace("-", "").replace("_", "") == model_key:
            return data
    return None


# ============ 兼容: 从基础参数推算模块 ============
def infer_model_architecture(params_b: float, hidden_size: int, num_layers: int,
                            num_heads: int, vocab_size: int, moe: bool = False,
                            num_experts: int = 0, active_experts: int = 0) -> Dict:
    """从基础参数推算模型架构"""
    head_dim = hidden_size // num_heads
    intermediate_size = int(hidden_size * 4)  # 默认 4x hidden_size

    modules = {
        "embedding": {"params": vocab_size * hidden_size, "type": "embedding"},
        "lm_head": {"params": vocab_size * hidden_size, "type": "output"},
        "input_layernorm": {"params": hidden_size, "type": "layernorm", "num": 1},
        "post_attention_layernorm": {"params": hidden_size, "type": "layernorm", "num": num_layers},
        "q_proj": {"params": hidden_size * hidden_size, "type": "linear", "num": num_layers},
        "k_proj": {"params": hidden_size * (num_heads * head_dim), "type": "linear", "num": num_layers},
        "v_proj": {"params": hidden_size * (num_heads * head_dim), "type": "linear", "num": num_layers},
        "o_proj": {"params": hidden_size * hidden_size, "type": "linear", "num": num_layers},
    }

    if moe and num_experts > 0:
        modules["gate_proj"] = {"params": hidden_size * intermediate_size, "type": "moe",
                               "num_experts": num_experts, "active_experts": active_experts,
                               "shared_experts": shared_experts, "num": 1}
        modules["up_proj"] = {"params": hidden_size * intermediate_size, "type": "moe",
                            "num_experts": num_experts, "active_experts": active_experts,
                            "shared_experts": shared_experts, "num": 1}
        modules["down_proj"] = {"params": intermediate_size * hidden_size, "type": "moe",
                              "num_experts": num_experts, "active_experts": active_experts,
                              "shared_experts": shared_experts, "num": 1}
    else:
        modules["gate_proj"] = {"params": hidden_size * intermediate_size, "type": "linear", "num": num_layers}
        modules["up_proj"] = {"params": hidden_size * intermediate_size, "type": "linear", "num": num_layers}
        modules["down_proj"] = {"params": intermediate_size * hidden_size, "type": "linear", "num": num_layers}

    return {
        "architecture": "inferred",
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "vocab_size": vocab_size,
        "intermediate_size": intermediate_size,
        "moe": moe,
        "num_experts": num_experts,
        "active_experts": active_experts,
        "modules": modules
    }


# ============ 芯片规格数据库 ============
CHIPS = {
    # NVIDIA
    "h100-80gb": {"vendor": "NVIDIA", "name": "H100 80GB", "vram_gb": 80, "type": "HBM3"},
    "h100-141gb": {"vendor": "NVIDIA", "name": "H100 141GB", "vram_gb": 141, "type": "HBM3e"},
    "a100-80gb": {"vendor": "NVIDIA", "name": "A100 80GB", "vram_gb": 80, "type": "HBM2e"},
    "rtx-4090": {"vendor": "NVIDIA", "name": "RTX 4090", "vram_gb": 24, "type": "GDDR6X"},
    # 华为昇腾
    "ascend-910b1-64gb": {"vendor": "Huawei", "name": "Ascend 910B1 64GB", "vram_gb": 64, "type": "HBM2e"},
    "ascend-910b2-64gb": {"vendor": "Huawei", "name": "Ascend 910B2 64GB", "vram_gb": 64, "type": "HBM2e"},
    "ascend-910b3-64gb": {"vendor": "Huawei", "name": "Ascend 910B3 64GB", "vram_gb": 64, "type": "HBM2e"},
    "ascend-910b4-32gb": {"vendor": "Huawei", "name": "Ascend 910B4 32GB", "vram_gb": 32, "type": "HBM2e"},
    "ascend-910c-128gb": {"vendor": "Huawei", "name": "Ascend 910C 128GB", "vram_gb": 128, "type": "HBM2e"},
    # 兼容旧名称
    "ascend-910b-64gb": {"vendor": "Huawei", "name": "Ascend 910B 64GB", "vram_gb": 64, "type": "HBM2e"},
}


@dataclass
class ParallelConfig:
    """并行策略配置"""
    tp: int = 1      # Tensor Parallelism
    pp: int = 1      # Pipeline Parallelism
    dp: int = 1      # Data Parallelism
    ep: int = 1      # Expert Parallelism (MoE)
    sp: bool = False # Sequence Parallelism
    cp: int = 1      # Context Parallelism

    @property
    def total_gpus(self) -> int:
        return self.tp * self.pp * self.dp * self.ep


@dataclass
class QuantConfig:
    """量化配置"""
    activation: str = "A8"   # A4 or A8
    weight: str = "W8"       # W4, W8, W16
    kv_cache: str = "C16"    # C4, C8, C16


def get_chip_spec(chip_name: str) -> Optional[Dict]:
    """获取芯片规格"""
    chip_key = chip_name.lower().replace("-", "").replace("_", "").replace(" ", "")
    for key, data in CHIPS.items():
        if key.replace("-", "").replace("_", "") == chip_key:
            return data
    return None


def calculate_module_memory(
    module_name: str,
    module_spec: Dict,
    hidden_size: int,
    num_layers: int,
    quant: QuantConfig,
    parallel: ParallelConfig,
    is_moe: bool = False,
    num_experts: int = 0,
    seq_len: int = 1,
    batch_size: int = 1,
) -> Dict[str, float]:
    """
    计算单个模块的显存占用

    返回:
    {
        "weight_gb": 权重显存 GB,
        "activation_gb": 激活显存 GB,
        "kv_cache_gb": KV Cache 显存 GB,
        "total_gb": 总显存 GB,
    }
    """
    act_bytes = QUANT_CONFIG.get(quant.activation, 1.0)
    weight_bytes = QUANT_CONFIG.get(quant.weight, 1.0)
    kv_bytes = QUANT_CONFIG.get(quant.kv_cache, 2.0)

    m_type = module_spec.get("type", "linear")
    m_num = module_spec.get("num", 1)

    # 权重计算
    total_params = module_spec.get("params", 0) * m_num

    if is_moe and m_type == "moe":
        # MoE 模块: 需要考虑 EP
        experts_per_layer = module_spec.get("num_experts", num_experts)
        # 每层所有专家的总参数量
        layer_params = module_spec.get("params", 0) * experts_per_layer
        # EP 后每卡只加载部分专家
        actual_experts = max(1, experts_per_layer // parallel.ep)
        total_weight = layer_params / parallel.ep * num_layers
    else:
        # 普通模块: 考虑 TP
        if m_type in ["linear", "embedding", "output"]:
            # TP 影响线性层
            total_weight = total_params / parallel.tp
        else:
            total_weight = total_params

    weight_gb = total_weight * weight_bytes / (1024 ** 3)

    # 激活计算
    activation_gb = 0.0

    if m_type == "linear":
        # 线性层激活: output * batch * seq
        # TP 后每卡的输出大小
        out_features = module_spec.get("params", 0) // hidden_size if hidden_size > 0 else hidden_size
        per_gpu_out = out_features // parallel.tp if parallel.tp > 1 else out_features
        activation_gb = per_gpu_out * batch_size * seq_len * act_bytes / (1024 ** 3) * m_num

        # SP 影响序列维度的激活
        if parallel.sp and m_type == "linear":
            activation_gb = activation_gb / parallel.cp if parallel.cp > 1 else activation_gb
    elif m_type == "embedding":
        activation_gb = hidden_size * batch_size * seq_len * act_bytes / (1024 ** 3)
    elif m_type == "layernorm":
        activation_gb = hidden_size * batch_size * seq_len * act_bytes / (1024 ** 3) * m_num

    # KV Cache 计算 (只针对 attention 相关的 k_proj, v_proj)
    kv_cache_gb = 0.0
    if module_name in ["k_proj", "v_proj"]:
        # KV Cache = 2 * batch * seq * hidden * layers * kv_bytes
        # CP 分割序列
        seq_per_cp = seq_len // parallel.cp if parallel.cp > 1 else seq_len
        kv_cache_gb = 2 * batch_size * seq_per_cp * hidden_size * m_num * kv_bytes / (1024 ** 3)

    return {
        "weight_gb": weight_gb,
        "activation_gb": activation_gb,
        "kv_cache_gb": kv_cache_gb,
        "total_gb": weight_gb + activation_gb + kv_cache_gb,
    }


def calculate_full_memory(
    model_name: str,
    hidden_size: int,
    num_layers: int,
    vocab_size: int,
    chip_vram_gb: float,
    quant: QuantConfig,
    parallel: ParallelConfig,
    gpu_util: float = 0.9,
    batch_size: int = 1,
    seq_len: int = 1,
    is_moe: bool = False,
    num_experts: int = 0,
    active_experts: int = 0,
) -> Dict[str, Any]:
    """计算完整模型的显存占用"""

    # 获取或推算模型架构
    arch = get_model_architecture(model_name)
    if arch is None:
        arch = infer_model_architecture(
            params_b=0, hidden_size=hidden_size, num_layers=num_layers,
            num_heads=0, vocab_size=vocab_size, moe=is_moe,
            num_experts=num_experts, active_experts=active_experts
        )

    # 可用显存
    available_vram = chip_vram_gb * gpu_util

    # 分类模块
    weight_modules = {}  # 权重模块
    attn_modules = {}   # Attention 模块 (影响 KV Cache)
    other_modules = {}  # 其他模块

    for mod_name, mod_spec in arch["modules"].items():
        if mod_name in ["k_proj", "v_proj"]:
            attn_modules[mod_name] = mod_spec
        else:
            weight_modules[mod_name] = mod_spec

    # 计算各模块显存
    results = {
        "modules": {},
        "summary": {
            "total_weight_gb": 0.0,
            "total_activation_gb": 0.0,
            "total_kv_cache_gb": 0.0,
            "total_gb": 0.0,
        }
    }

    # 计算权重模块
    for mod_name, mod_spec in weight_modules.items():
        mem = calculate_module_memory(
            mod_name, mod_spec, hidden_size, num_layers, quant, parallel,
            is_moe, num_experts, seq_len, batch_size
        )
        results["modules"][mod_name] = mem
        results["summary"]["total_weight_gb"] += mem["weight_gb"]
        results["summary"]["total_activation_gb"] += mem["activation_gb"]

    # 计算 Attention 模块 (包含 KV Cache)
    for mod_name, mod_spec in attn_modules.items():
        mem = calculate_module_memory(
            mod_name, mod_spec, hidden_size, num_layers, quant, parallel,
            is_moe, num_experts, seq_len, batch_size
        )
        results["modules"][mod_name] = mem
        results["summary"]["total_weight_gb"] += mem["weight_gb"]
        results["summary"]["total_kv_cache_gb"] += mem["kv_cache_gb"]

    results["summary"]["total_gb"] = (
        results["summary"]["total_weight_gb"] +
        results["summary"]["total_activation_gb"] +
        results["summary"]["total_kv_cache_gb"]
    )

    results["available_vram_gb"] = available_vram
    results["can_fit"] = results["summary"]["total_gb"] <= available_vram

    # 估算最大 seq_len
    if results["summary"]["total_kv_cache_gb"] > 0:
        kv_per_token = results["summary"]["total_kv_cache_gb"] / seq_len if seq_len > 0 else 0
        remaining = available_vram - results["summary"]["total_weight_gb"]
        if kv_per_token > 0 and remaining > 0:
            results["max_seq_len"] = int(remaining / kv_per_token)
        else:
            results["max_seq_len"] = 0
    else:
        results["max_seq_len"] = 0

    return results


def format_detailed_report(
    model_name: str,
    hidden_size: int,
    num_layers: int,
    vocab_size: int,
    chip_name: str,
    chip_vram_gb: float,
    quant: QuantConfig,
    parallel: ParallelConfig,
    gpu_util: float = 0.9,
    is_moe: bool = False,
    num_experts: int = 0,
    seq_len: int = 1,
    ref_seq_len: int = 2048,
    user_strategies: Optional[List[Dict]] = None,
) -> str:
    """生成详细报告"""

    # 确保类型正确
    chip_vram_gb = float(chip_vram_gb)

    # 获取架构
    arch = get_model_architecture(model_name)
    if arch is None:
        arch = infer_model_architecture(0, hidden_size, num_layers, 0, vocab_size, is_moe, num_experts)

    quant_str = f"{quant.activation}{quant.weight}{quant.kv_cache}"

    report = f"""# 显存估算详细报告

---

## 1. 模型结构参数

| 参数 | 值 |
|------|-----|
| **模型名称** | {model_name} |
| **架构** | {arch.get('architecture', 'N/A')} |
| **隐藏层大小** | {hidden_size} |
| **层数** | {num_layers} |
| **词汇表大小** | {vocab_size} |
| **中间层大小** | {arch.get('intermediate_size', 'N/A')} |
| **MoE** | {'是' if is_moe else '否'} |
| {'**专家数量**' if is_moe else ''} | {num_experts if is_moe else 'N/A'} |

---

## 2. 硬件配置

| 参数 | 值 |
|------|-----|
| **芯片** | {chip_name} |
| **显存总量** | {chip_vram_gb} GB |
| **GPU利用率** | {gpu_util * 100}% |
| **可用显存** | {chip_vram_gb * gpu_util:.1f} GB |

---

## 3. 并行策略

"""

    # 生成并行策略说明
    if user_strategies and len(user_strategies) > 1:
        # 用户指定了多个策略
        report += "| 策略 | 说明 | GPU数 |\n"
        report += "|------|------|------|\n"
        for s in user_strategies:
            strategy_name = s.get("name", "Unknown")
            tp = s.get("tp", 1)
            ep = s.get("ep", 1)
            dp = s.get("dp", 1)
            total_gpus = tp * ep * dp

            # 生成说明
            desc_parts = []
            if tp > 1:
                desc_parts.append(f"TP={tp}")
            if ep > 1:
                desc_parts.append(f"EP={ep}")
            if dp > 1:
                desc_parts.append(f"DP={dp}")
            if s.get("sp"):
                desc_parts.append("SP")
            if s.get("cp", 1) > 1:
                desc_parts.append(f"CP={s.get('cp')}")

            desc = ", ".join(desc_parts) if desc_parts else "无并行"

            report += f"| {strategy_name} | {desc} | {total_gpus} GPUs |\n"
    else:
        # 单策略显示详细表格
        report += f"""| 并行策略 | 值 |
|----------|-----|
| **TP (Tensor Parallel)** | {parallel.tp} |
| **PP (Pipeline Parallel)** | {parallel.pp} |
| **DP (Data Parallel)** | {parallel.dp} |
| **EP (Expert Parallel)** | {parallel.ep} |
| **SP (Sequence Parallel)** | {'是' if parallel.sp else '否'} |
| **CP (Context Parallel)** | {parallel.cp} |
| **总GPU数** | {parallel.total_gpus} |
"""

    report += f"""
---

## 4. 量化配置

| 类型 | 格式 | 字节数 |
|------|------|--------|
| **激活** | {quant.activation} | {QUANT_CONFIG.get(quant.activation, 0)} |
| **权重** | {quant.weight} | {QUANT_CONFIG.get(quant.weight, 0)} |
| **KV Cache** | {quant.kv_cache} | {QUANT_CONFIG.get(quant.kv_cache, 0)} |

---

## 5. 模块级显存占用 ({quant_str}) - 不同并行策略对比
"""

    # 定义要对比的并行策略
    # 如果用户指定了策略，使用用户策略；否则使用默认策略
    if user_strategies and len(user_strategies) > 0:
        parallel_configs = user_strategies
    else:
        parallel_configs = [
            {"tp": 1, "dp": 1, "ep": 1, "sp": False, "cp": 1, "name": "SP=1"},
            {"tp": 2, "dp": 1, "ep": 1, "sp": False, "cp": 1, "name": "TP2"},
            {"tp": 4, "dp": 1, "ep": 1, "sp": False, "cp": 1, "name": "TP4"},
            {"tp": 8, "dp": 1, "ep": 1, "sp": False, "cp": 1, "name": "TP8"},
            {"tp": 8, "dp": 1, "ep": 1, "sp": False, "cp": 4, "name": "TP8+CP4"},
            {"tp": 8, "dp": 1, "ep": 1, "sp": True, "cp": 4, "name": "TP8+SP+CP4"},
        ]

        # 添加 MoE 专用策略
        if is_moe and num_experts > 0:
            parallel_configs.extend([
                {"tp": 8, "dp": 1, "ep": 8, "sp": False, "cp": 1, "name": "TP8+EP8"},
                {"tp": 8, "dp": 2, "ep": 8, "sp": False, "cp": 1, "name": "TP8+EP8+DP2"},
            ])

    # 默认量化类型
    quant_types = [quant_str]

    # 动态生成表格头部 - 格式: 策略名 (量化类型)
    header_row1 = "| 模块 | 类型 | 层数 | 维度 | 参数量(M) | 显存占用总量(GB) |"
    header_row2 = "|------|------|------|------|-----------|------------------|"
    for p in parallel_configs:
        header_row1 += f" {p['name']} ({quant_str}) |"
        header_row2 += "----------------------|"
    report += header_row1 + "\n" + header_row2 + "\n"

    act_bytes = QUANT_CONFIG.get(quant.activation, 1.0)
    weight_bytes = QUANT_CONFIG.get(quant.weight, 1.0)
    kv_bytes = QUANT_CONFIG.get(quant.kv_cache, 2.0)

    # 计算每个模块在不同并行策略下的显存
    modules_data = {}
    totals = {p["name"]: 0.0 for p in parallel_configs}

    # 计算总参数量（用于总计行）
    total_params_m = 0

    for mod_name, mod_spec in arch["modules"].items():
        m_type = mod_spec.get("type", "linear")
        m_num = mod_spec.get("num", 1)
        params = mod_spec.get("params", 0)

        # 获取模块的完整维度 (in_features, out_features)
        kv_dim = arch.get("kv_dim", 1024)  # Qwen系列KV维度不同
        intermediate = arch.get("intermediate_size", hidden_size * 4)

        if "embedding" in mod_name:
            mod_type = "embedding"
            mod_dim = f"({vocab_size}, {hidden_size})"
        elif mod_name == "lm_head":
            mod_type = "embedding"
            mod_dim = f"({hidden_size}, {vocab_size})"
        elif mod_name == "q_proj":
            mod_type = "attn_qkv"
            mod_dim = f"({hidden_size}, {hidden_size})"
        elif mod_name == "k_proj":
            mod_type = "attn_qkv"
            mod_dim = f"({hidden_size}, {kv_dim})"
        elif mod_name == "v_proj":
            mod_type = "attn_qkv"
            mod_dim = f"({hidden_size}, {kv_dim})"
        elif mod_name == "o_proj":
            mod_type = "attn_o"
            mod_dim = f"({hidden_size}, {hidden_size})"
        elif mod_name == "gate_proj":
            mod_type = "mlp"
            mod_dim = f"({hidden_size}, {intermediate})"
        elif mod_name == "up_proj":
            mod_type = "mlp"
            mod_dim = f"({hidden_size}, {intermediate})"
        elif mod_name == "down_proj":
            mod_type = "mlp"
            mod_dim = f"({intermediate}, {hidden_size})"
        elif "layernorm" in mod_name:
            mod_type = "layernorm"
            mod_dim = f"({hidden_size},)"
        else:
            mod_type = "other"
            mod_dim = f"({hidden_size},)"

        # 计算每个并行策略下的显存
        memory_per_parallel = {}
        for p in parallel_configs:
            # 排除 name 字段，只传入并行参数
            parallel_params = {k: v for k, v in p.items() if k != "name"}
            parallel = ParallelConfig(**parallel_params)

            # 权重
            if is_moe and m_type == "moe":
                # MoE 模块: params是单层单个专家的参数量
                # 全部专家的权重需要加载到显存，然后根据EP切分
                total_exp = mod_spec.get("num_experts", 1)
                # EP后每卡实际加载的专家数 = 全部专家数 / EP
                actual_experts = max(1, total_exp // parallel.ep) if parallel.ep > 1 else total_exp
                # m_num = 1 (因为params是单层), 需要额外乘以num_layers
                # 单卡专家权重 = 总参数 × 量化字节数 / EP (不再除以TP)
                weight = params * actual_experts * num_layers * weight_bytes / (1024 ** 3) * parallel.dp
            elif m_type in ["linear", "embedding", "output"]:
                weight = params * m_num * weight_bytes / parallel.tp / (1024 ** 3) * parallel.dp
            else:
                weight = params * m_num * weight_bytes / (1024 ** 3) * parallel.dp

            # 激活
            if m_type == "embedding":
                activation = hidden_size * seq_len * act_bytes / (1024 ** 3)
                if parallel.sp:
                    activation = activation / parallel.cp
            elif m_type == "layernorm":
                activation = hidden_size * seq_len * m_num * act_bytes / (1024 ** 3)
                if parallel.sp:
                    activation = activation / parallel.cp
            else:
                activation = hidden_size * seq_len * m_num * act_bytes / (1024 ** 3)
                if parallel.sp:
                    activation = activation / parallel.cp

            # KV Cache
            if mod_name in ["k_proj", "v_proj"]:
                seq_per_cp = seq_len // parallel.cp if parallel.cp > 1 else seq_len
                kv = 2 * hidden_size * seq_per_cp * m_num * kv_bytes / (1024 ** 3)
            else:
                kv = 0.0

            total = weight + activation + kv
            memory_per_parallel[p["name"]] = total
            totals[p["name"]] += total

        # MoE 模块显示实际层数，非MoE使用m_num
        display_layers = num_layers if (is_moe and m_type == "moe") else m_num

        # 参数量计算：MoE 需要乘以全部专家数（权重需要全部加载到显存）
        if is_moe and m_type == "moe":
            total_experts = mod_spec.get("num_experts", 1)
            total_params = params * total_experts * display_layers
        else:
            total_params = params * display_layers

        modules_data[mod_name] = {
            "type": mod_type,
            "layers": display_layers,
            "dim": mod_dim,
            "params_m": total_params / 1e6,
            "memory": memory_per_parallel,
            "weight": {p["name"]: memory_per_parallel[p["name"]] * 0.9 for p in parallel_configs},  # 估算权重占比90%
            "activation": {p["name"]: memory_per_parallel[p["name"]] * 0.05 for p in parallel_configs},  # 估算激活占比5%
            "kv": {p["name"]: memory_per_parallel[p["name"]] * 0.05 for p in parallel_configs},  # 估算KV占比5%
        }
        total_params_m += total_params / 1e6

    # 按类型和名称排序输出
    type_order = {"embedding": 0, "attn_qkv": 1, "attn_o": 2, "mlp": 3, "layernorm": 4, "other": 5}
    sorted_modules = sorted(modules_data.items(), key=lambda x: (type_order.get(x[1]["type"], 6), x[0]))

    # 计算每个模块的显存占用总量（单卡，所有并行策略的权重之和）
    for mod_name, mod_info in sorted_modules:
        # 显存占用总量 = 该模块所有并行策略的权重显存之和（用于展示单卡负载）
        total_memory = sum(mod_info['memory'].values())
        row = f"| {mod_name} | {mod_info['type']} | {mod_info['layers']} | {mod_info['dim']} | {mod_info['params_m']:.2f} | {total_memory:.2f} |"
        for p in parallel_configs:
            row += f" {mod_info['memory'][p['name']]:.2f} |"
        report += row + "\n"

    # 总计行
    report += f"| **总计** | - | - | - | **{total_params_m:.2f}** | - |"
    for p in parallel_configs:
        report += f" **{totals[p['name']]:.2f}** |"
    report += "\n"

    # 估算 max seq_len
    available = chip_vram_gb * gpu_util

    # 使用第一个并行策略计算最大seq_len
    first_strategy_name = parallel_configs[0]["name"] if parallel_configs else "SP=1"
    first_total = totals.get(first_strategy_name, 0)
    if first_total > 0 and first_total <= available:
        max_seq = int((available - first_total * 0.5) / (first_total / seq_len)) if seq_len > 0 else 0
    else:
        max_seq = 0

    # 估算各并行策略下的最大 seq_len
    report += f"""
### 各并行策略最大 Seq Len 估算

*(gpu_util={gpu_util*100:.0f}%, 显存总量={chip_vram_gb}GB)*

| 并行策略 | 权重(GB) | 激活峰值(GB) | 可用KV Cache(GB) | 最大SeqLen |
|----------|----------|-------------|------------------|------------|
"""
    for p in parallel_configs:
        total_mem = totals[p["name"]]
        strategy_name = p["name"]

        # 计算权重显存（固定成本，不随seq变化）
        weight_mem = 0
        for mod_name, mod_info in modules_data.items():
            weight_mem += mod_info["weight"].get(strategy_name, 0)
        weight_mem = max(0, weight_mem)

        # 激活估计 = 峰值激活占用量
        # 使用参考序列长度 2048 来估算峰值激活（长上下文场景）
        # 对于 MoE 模型，激活峰值主要来自 MoE 层的中间计算
        # 激活 ≈ 2 * intermediate_size * ref_seq_len * num_layers * act_bytes
        # 系数2考虑: MoE的gate_proj + up_proj 中间激活
        tp = p.get("tp", 1)
        ep = p.get("ep", 1)
        cp = p.get("cp", 1)

        # 获取中间层大小（用于激活计算）
        intermediate = arch.get("intermediate_size", hidden_size)

        # ref_seq_len 由参数传入
        act_mem = 2 * intermediate * ref_seq_len * num_layers * act_bytes / tp / ep / (1024 ** 3)

        # 每token的KV Cache成本 = 2 * hidden_size * kv_bytes * num_layers (所有层)
        # 这是因为每个token的KV需要存储所有层的
        per_token_kv = 2 * hidden_size * kv_bytes * num_layers / (1024 ** 3)

        # 可用KV Cache = 总内存 * gpu_util - 权重 - 激活峰值
        total_available = chip_vram_gb * gpu_util
        available_kv = total_available - weight_mem - act_mem
        available_kv = max(0, available_kv)

        # 最大SeqLen = 可用KV显存 / 每token的KV成本
        if per_token_kv > 0:
            max_s = int(available_kv / per_token_kv)
        else:
            max_s = 0
        max_s = max(0, max_s)

        report += f"| {p['name']} | {weight_mem:.2f} | {act_mem:.2f} | {available_kv:.2f} | ~{max_s:,} |\n"

    # 添加计算步骤示例 (使用第一个并行策略)
    first_strategy = parallel_configs[0] if parallel_configs else {"name": "SP=1", "tp": 1}
    example_name = first_strategy.get("name", "SP=1")
    example_tp = first_strategy.get("tp", 1)

    # 获取示例模块的显存
    emb_mem = modules_data.get('embedding', {}).get('memory', {}).get(example_name, 0)
    q_mem = modules_data.get('q_proj', {}).get('memory', {}).get(example_name, 0)
    mlp_mem = modules_data.get('gate_proj', {}).get('memory', {}).get(example_name, 0)

    report += f"""
---

## 6. 计算步骤说明

### 显存占用计算公式

```
权重显存(GB) = 参数总量 × 量化字节数 / TP / EP / DP / 1024³
激活显存(GB) = hidden_size × seq_len × batch_size × 激活量化字节数 × 层数 / TP / EP / 1024³
KV Cache(GB) = 2 × hidden_size × seq_len × batch_size × KV量化字节数 × 层数 / CP / 1024³
```

### 计算示例 (以 {example_name} 为例)

| 类别 | 计算公式 | 结果 |
|------|---------|------|
| embedding 权重 | {modules_data.get('embedding', {}).get('params_m', 0):.2f}M × {weight_bytes} bytes / {example_tp} TP | {emb_mem:.2f} GB |
| q_proj 权重 | {modules_data.get('q_proj', {}).get('params_m', 0):.2f}M × {weight_bytes} bytes / {example_tp} TP | {q_mem:.2f} GB |
| MLP 权重 | {modules_data.get('gate_proj', {}).get('params_m', 0):.2f}M × {weight_bytes} bytes / {example_tp} TP | {mlp_mem:.2f} GB |
| 激活 (峰值) | {hidden_size} × {seq_len} × {num_layers} × {act_bytes} bytes / {example_tp} TP | ~{hidden_size * seq_len * num_layers * act_bytes / example_tp / (1024**3):.2f} GB |
| KV Cache (单token) | 2 × {hidden_size} × {kv_bytes} bytes | ~{2 * hidden_size * kv_bytes / (1024**3):.4f} GB |

### 最大序列长度估算公式

```
KV Cache = 显存总量 × gpu_util - 权重 - 激活
最大SeqLen = KV Cache / (2 × hidden_size × KV量化字节数)
```

---

*注: 表格中的数值为每张GPU的显存占用(GB)，包含权重+激活+KV Cache*

*Generated by Context Length Estimator V2*
"""
    return report


def generate_comparison_table(
    model_name: str,
    hidden_size: int,
    num_layers: int,
    vocab_size: int,
    chip_name: str,
    chip_vram_gb: float,
    gpu_util: float = 0.9,
    is_moe: bool = False,
    num_experts: int = 0,
    base_seq_len: int = 4096,
) -> str:
    """生成不同量化+并行策略的对比表格"""

    # 量化组合
    quant_combos = ["A8W8C16", "A8W8C8", "A8W4C16", "A8W4C8", "A4W4C16", "A4W4C8"]

    # 并行策略组合 - 扩展为更详细的对比
    parallel_combos = [
        {"tp": 1, "dp": 1, "ep": 1, "sp": False, "cp": 1, "name": "SP=1"},
        {"tp": 2, "dp": 1, "ep": 1, "sp": False, "cp": 1, "name": "TP2"},
        {"tp": 4, "dp": 1, "ep": 1, "sp": False, "cp": 1, "name": "TP4"},
        {"tp": 8, "dp": 1, "ep": 1, "sp": False, "cp": 1, "name": "TP8"},
        {"tp": 8, "dp": 1, "ep": 1, "sp": True, "cp": 1, "name": "TP8+SP"},
        {"tp": 8, "dp": 1, "ep": 1, "sp": False, "cp": 4, "name": "TP8+CP4"},
        {"tp": 8, "dp": 1, "ep": 1, "sp": True, "cp": 4, "name": "TP8+SP+CP4"},
    ]

    # 添加 MoE 专用策略
    if is_moe and num_experts > 0:
        parallel_combos.extend([
            {"tp": 8, "dp": 1, "ep": 8, "sp": False, "cp": 1, "name": "TP8+EP8"},
            {"tp": 8, "dp": 2, "ep": 8, "sp": False, "cp": 1, "name": "TP8+EP8+DP2"},
            {"tp": 8, "dp": 8, "ep": 1, "sp": False, "cp": 1, "name": "TP8+DP8"},
        ])

    report = f"""
---

## 7. 不同量化+并行策略对比 (seq_len={base_seq_len})

### 7.1 每卡显存占用总表

| 量化 | 并行 | 总GPU | 权重/卡(GB) | 激活/卡(GB) | KV/卡(GB) | 总计/卡(GB) | 可运行 |
|------|------|-------|-------------|-------------|------------|--------------|--------|
"""

    # 获取架构
    arch = get_model_architecture(model_name)
    if arch is None:
        arch = infer_model_architecture(0, hidden_size, num_layers, 0, vocab_size, is_moe, num_experts)

    available = chip_vram_gb * gpu_util
    results = []

    for quant_str in quant_combos:
        quant = QuantConfig()
        if quant_str in QUANT_PRESETS:
            cfg = QUANT_PRESETS[quant_str]
            quant.activation = cfg["activation"]
            quant.weight = cfg["weight"]
            quant.kv_cache = cfg["kv_cache"]
        else:
            quant.activation = "A8"
            quant.weight = "W8"
            quant.kv_cache = "C16"

        act_bytes = QUANT_CONFIG.get(quant.activation, 1.0)
        weight_bytes = QUANT_CONFIG.get(quant.weight, 1.0)
        kv_bytes = QUANT_CONFIG.get(quant.kv_cache, 2.0)

        for p in parallel_combos:
            parallel = ParallelConfig(
                tp=p["tp"], dp=p["dp"], ep=p["ep"],
                sp=p["sp"], cp=p["cp"]
            )

            # 计算每卡的显存
            total_weight_per_gpu = 0.0
            total_activation_per_gpu = 0.0
            total_kv_per_gpu = 0.0

            for mod_name, mod_spec in arch["modules"].items():
                m_type = mod_spec.get("type", "linear")
                m_num = mod_spec.get("num", 1)
                params = mod_spec.get("params", 0)

                # 权重计算 - 考虑并行策略
                if is_moe and m_type == "moe":
                    experts_per_layer = mod_spec.get("num_experts", num_experts)
                    actual_experts = max(1, experts_per_layer // parallel.ep)
                    # EP: 每卡只加载部分专家; DP: 每卡有完整副本
                    weight_per_gpu = params * actual_experts * m_num * weight_bytes / (1024 ** 3) * parallel.dp
                elif m_type in ["linear", "embedding", "output"]:
                    # TP: 权重分布在多卡上; DP: 每卡有完整副本
                    weight_per_gpu = params * m_num * weight_bytes / parallel.tp / (1024 ** 3) * parallel.dp
                else:
                    weight_per_gpu = params * m_num * weight_bytes / (1024 ** 3) * parallel.dp

                # 激活计算
                if m_type == "embedding":
                    activation_per_gpu = hidden_size * base_seq_len * act_bytes / (1024 ** 3)
                    # SP: 激活分布在序列维度
                    if parallel.sp:
                        activation_per_gpu = activation_per_gpu / parallel.cp
                elif m_type == "layernorm":
                    activation_per_gpu = hidden_size * base_seq_len * m_num * act_bytes / (1024 ** 3)
                    if parallel.sp:
                        activation_per_gpu = activation_per_gpu / parallel.cp
                else:
                    # 线性层激活
                    activation_per_gpu = hidden_size * base_seq_len * m_num * act_bytes / (1024 ** 3)
                    if parallel.sp:
                        activation_per_gpu = activation_per_gpu / parallel.cp

                # KV Cache 计算
                if mod_name in ["k_proj", "v_proj"]:
                    # CP: KV Cache 分布在序列维度
                    seq_per_cp = base_seq_len // parallel.cp if parallel.cp > 1 else base_seq_len
                    kv_per_gpu = 2 * hidden_size * seq_per_cp * m_num * kv_bytes / (1024 ** 3)
                else:
                    kv_per_gpu = 0.0

                total_weight_per_gpu += weight_per_gpu
                total_activation_per_gpu += activation_per_gpu
                total_kv_per_gpu += kv_per_gpu

            total_per_gpu = total_weight_per_gpu + total_activation_per_gpu + total_kv_per_gpu
            total_gpus = parallel.total_gpus
            can_run = "✅" if total_per_gpu <= available else "❌"

            report += f"| {quant_str} | {p['name']} | {total_gpus} | {total_weight_per_gpu:.2f} | {total_activation_per_gpu:.2f} | {total_kv_per_gpu:.2f} | {total_per_gpu:.2f} | {can_run} |\n"

    # 添加详细分解表
    report += f"""
### 7.2 详细分解表 (以 A4W4C8 + TP8 为例)

| 模块类型 | 权重/卡(GB) | 激活/卡(GB) | KV/卡(GB) | 说明 |
|----------|-------------|-------------|-----------|------|
"""

    # 选取一个配置进行详细分解
    sample_quant = "A4W4C8"
    sample_parallel = {"tp": 8, "dp": 1, "ep": 1, "sp": False, "cp": 1}

    if sample_quant in QUANT_PRESETS:
        cfg = QUANT_PRESETS[sample_quant]
        act_bytes = QUANT_CONFIG.get(cfg["activation"], 1.0)
        weight_bytes = QUANT_CONFIG.get(cfg["weight"], 1.0)
        kv_bytes = QUANT_CONFIG.get(cfg["kv_cache"], 2.0)

    parallel = ParallelConfig(**sample_parallel)

    # 模块分类统计
    module_stats = {
        "embedding/lm_head": {"weight": 0, "activation": 0, "kv": 0},
        "attention (qkv)": {"weight": 0, "activation": 0, "kv": 0},
        "attention (o)": {"weight": 0, "activation": 0, "kv": 0},
        "MLP (gate/up/down)": {"weight": 0, "activation": 0, "kv": 0},
        "layernorm": {"weight": 0, "activation": 0, "kv": 0},
    }

    for mod_name, mod_spec in arch["modules"].items():
        m_type = mod_spec.get("type", "linear")
        m_num = mod_spec.get("num", 1)
        params = mod_spec.get("params", 0)

        # 权重
        if is_moe and m_type == "moe":
            experts_per_layer = mod_spec.get("num_experts", num_experts)
            actual_experts = max(1, experts_per_layer // parallel.ep)
            weight = params * actual_experts * m_num * weight_bytes / (1024 ** 3)
        elif m_type in ["linear", "embedding", "output"]:
            weight = params * m_num * weight_bytes / parallel.tp / (1024 ** 3)
        else:
            weight = params * m_num * weight_bytes / (1024 ** 3)

        # 激活
        if m_type == "embedding":
            activation = hidden_size * base_seq_len * act_bytes / (1024 ** 3)
        elif m_type == "layernorm":
            activation = hidden_size * base_seq_len * m_num * act_bytes / (1024 ** 3)
        else:
            activation = hidden_size * base_seq_len * m_num * act_bytes / (1024 ** 3)

        # KV
        if mod_name in ["k_proj", "v_proj"]:
            kv = 2 * hidden_size * base_seq_len * m_num * kv_bytes / (1024 ** 3)
        else:
            kv = 0.0

        # 分类
        if "embedding" in mod_name or "lm_head" in mod_name:
            module_stats["embedding/lm_head"]["weight"] += weight
            module_stats["embedding/lm_head"]["activation"] += activation
        elif mod_name in ["q_proj", "k_proj", "v_proj"]:
            module_stats["attention (qkv)"]["weight"] += weight
            module_stats["attention (qkv)"]["activation"] += activation
            module_stats["attention (qkv)"]["kv"] += kv
        elif mod_name == "o_proj":
            module_stats["attention (o)"]["weight"] += weight
            module_stats["attention (o)"]["activation"] += activation
        elif "proj" in mod_name:
            module_stats["MLP (gate/up/down)"]["weight"] += weight
            module_stats["MLP (gate/up/down)"]["activation"] += activation
        elif "layernorm" in mod_name:
            module_stats["layernorm"]["weight"] += weight
            module_stats["layernorm"]["activation"] += activation

    for cat, vals in module_stats.items():
        report += f"| {cat} | {vals['weight']:.4f} | {vals['activation']:.4f} | {vals['kv']:.4f} | - |\n"

    report += f"""
---

*注:*
- *权重/卡 = 模型权重 / TP / DP × EP影响*
- *激活/卡 = 激活值 / SP / CP*
- *KV/卡 = KV Cache / CP*
- *总计/卡 必须 < 可用显存 ({available:.1f}GB)*
"""
    return report


def main():
    parser = argparse.ArgumentParser(description="Context Length Estimator V2")
    parser.add_argument("--model", "-m", type=str, help="模型名称")
    parser.add_argument("--chip", "-c", type=str, help="芯片名称")

    # 基础参数
    parser.add_argument("--hidden", type=int, help="隐藏层大小")
    parser.add_argument("--layers", "-l", type=int, help="层数")
    parser.add_argument("--vocab", type=int, help="词汇表大小")
    parser.add_argument("--params", "-p", type=float, help="参数量(B)")

    # 并行策略
    parser.add_argument("--tp", type=int, default=1, help="Tensor Parallelism (默认1)")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline Parallelism (默认1)")
    parser.add_argument("--dp", type=int, default=1, help="Data Parallelism (默认1)")
    parser.add_argument("--ep", type=int, default=1, help="Expert Parallelism (默认1)")
    parser.add_argument("--sp", action="store_true", help="启用 Sequence Parallelism")
    parser.add_argument("--cp", type=int, default=1, help="Context Parallelism (默认1)")
    parser.add_argument("--strategies", "-S", type=str,
                       help="并行策略字符串，如 'TP8EP8', 'TP8EP8+W8A8C16', 'TP8EP16+INT4'")

    # 量化
    parser.add_argument("--quant", "-q", type=str, default="fp16",
                       help="量化配置: A8W8C16, A8W8C8, A8W4C16, A8W4C8, A4W4C16, A4W4C8")

    # 其他
    parser.add_argument("--vram", type=float, help="显存大小 (GB)")
    parser.add_argument("--gpu-util", "-g", type=float, default=0.9, help="GPU利用率 (默认0.9)")
    parser.add_argument("--batch", "-b", type=int, default=1, help="批大小 (默认1)")
    parser.add_argument("--seq", "-s", type=int, default=1, help="序列长度 (默认1)")
    parser.add_argument("--ref-seq", "-r", type=int, default=2048, help="参考序列长度，用于激活峰值估算 (默认2048)")
    parser.add_argument("--num-experts", "-e", type=int, help="MoE专家数量")
    parser.add_argument("--config", type=str, help="从 HuggingFace config.json 加载模型参数")
    parser.add_argument("--output", "-o", choices=["text", "json"], default="text", help="输出格式")

    args = parser.parse_args()

    # 获取模型信息
    model_name = args.model or "unknown"

    # 优先从 config.json 加载（如果指定）
    if args.config:
        arch = load_model_from_config(args.config)
        if arch:
            model_name = args.config.split('/')[-2] if '/' in args.config else "custom"
    else:
        arch = get_model_architecture(model_name)

    if arch:
        hidden_size = args.hidden or arch.get("hidden_size", 4096)
        num_layers = args.layers or arch.get("num_layers", 32)
        vocab_size = args.vocab or arch.get("vocab_size", 32000)
        is_moe = arch.get("moe", False)
        num_experts = args.num_experts or arch.get("num_experts", 0)
    else:
        hidden_size = args.hidden or 4096
        num_layers = args.layers or 32
        vocab_size = args.vocab or 32000
        is_moe = args.num_experts is not None and args.num_experts > 0
        num_experts = args.num_experts or 0

    # 获取芯片信息
    chip_name = args.chip or "unknown"
    chip_data = get_chip_spec(chip_name)

    if chip_data:
        chip_vram_gb = float(args.vram or chip_data.get("vram_gb", 80))
        chip_name = chip_data.get("name", chip_name)
    else:
        chip_vram_gb = float(args.vram or 80)

    # 解析量化配置
    # 如果用户指定了 --strategies，优先使用其中的量化配置
    quant_str = args.quant
    if args.strategies:
        _, strategy_quant = parse_strategy_string(args.strategies)
        if strategy_quant:
            quant_str = strategy_quant

    quant = QuantConfig()
    if quant_str in QUANT_PRESETS:
        cfg = QUANT_PRESETS[quant_str]
        quant.activation = cfg["activation"]
        quant.weight = cfg["weight"]
        quant.kv_cache = cfg["kv_cache"]
    else:
        # 简单解析
        quant.activation = "A16"
        quant.weight = "W16"
        quant.kv_cache = "C16"

    # 并行配置 - 如果指定了 --strategies，解析用户策略
    user_strategies = None
    if args.strategies:
        user_strategies, _ = parse_strategy_string(args.strategies)

    # 如果没有指定策略，使用默认的
    if not user_strategies:
        parallel = ParallelConfig(
            tp=args.tp, pp=args.pp, dp=args.dp, ep=args.ep,
            sp=args.sp, cp=args.cp
        )
    else:
        # 使用用户指定的第一个策略
        s = user_strategies[0]
        parallel = ParallelConfig(
            tp=s.get("tp", 1), pp=s.get("pp", 1), dp=s.get("dp", 1),
            ep=s.get("ep", 1), sp=s.get("sp", False), cp=s.get("cp", 1)
        )

    # 生成报告 - 传递用户策略列表
    report = format_detailed_report(
        model_name, hidden_size, num_layers, vocab_size,
        chip_name, float(chip_vram_gb), quant, parallel,
        args.gpu_util, is_moe, num_experts, args.seq, args.ref_seq,
        user_strategies=user_strategies
    )

    # 对比表已合并到 format_detailed_report 中

    print(report)


if __name__ == "__main__":
    main()
