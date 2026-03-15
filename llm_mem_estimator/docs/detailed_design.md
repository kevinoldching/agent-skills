# LLM 显存占用估算器 - 详细设计

## 1. 系统架构

### 1.1 整体架构（简化版）

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI / Skill 接口                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      MemoryEstimator                         │
│  (主控制器: 协调配置加载、显存计算、结果汇总)                 │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ ConfigLoader │    │FormulaEvaluator│  │ReportGenerator│
│ (配置加载)   │    │ (公式计算)   │    │ (报告生成)    │
└──────────────┘    └──────────────┘    └──────────────┘
```

**设计理念**: 配置驱动，无需复杂的模块类层次结构

### 1.2 数据流

```
YAML配置文件 → ConfigLoader → ModelConfig对象
                                    │
                                    ▼
              MemoryEstimator 遍历权重计算显存
                                    │
                                    ▼
              FormulaEvaluator 解析公式计算 KV Cache/激活值
                                    │
                                    ▼
              ReportGenerator 生成 Markdown 报告
```

## 2. 核心模块设计

### 2.1 ConfigLoader (配置加载器)

**职责**:
- 加载并解析 YAML 配置文件
- 验证配置完整性
- 加载硬件配置 (chips.json)
- 生成模型配置文件（从 HuggingFace 或本地权重）

**接口**:
```python
class ConfigLoader:
    @staticmethod
    def load_model_config(yaml_path: str) -> ModelConfig:
        """加载模型配置文件"""
        pass

    @staticmethod
    def load_chip_config(chip_name: str) -> ChipConfig:
        """加载芯片配置"""
        pass

    @staticmethod
    def validate_config(config: ModelConfig) -> None:
        """验证配置完整性，失败则抛出异常"""
        pass

    @staticmethod
    def generate_config_from_weights(repo_id: str, token: str = None) -> ModelConfig:
        """
        从 HuggingFace 或本地权重目录生成配置文件

        实现步骤：
        1. 使用 get_safetensors_metadata 获取权重元数据
        2. 分析权重名称，归类到 embedding/attention/ffn/norm/others
        3. 读取 config.json 获取架构参数
        4. 生成标准的 YAML 配置文件
        """
        pass
```

### 2.2 ModelConfig (模型配置数据类)

**职责**: 存储模型配置信息

**数据结构**:
```python
@dataclass
class ModelConfig:
    # 模型标识
    model_identity: ModelIdentity

    # 架构配置（扁平化结构）
    architecture_config: Dict[str, Any]

    # 模块定义
    modules: ModulesConfig

    # 计算规则
    computation_rules: ComputationRules

@dataclass
class ModelIdentity:
    name: str
    total_params_b: float
    num_layers: int
    quantization: str

@dataclass
class ModulesConfig:
    """模块配置，包含 embedding, norm, attention, ffn, others"""
    embedding: Dict[str, WeightInfo]
    norm: Dict[str, WeightInfo]
    attention: AttentionConfig
    ffn: FFNConfig
    others: Dict[str, WeightInfo]

@dataclass
class WeightInfo:
    """权重信息"""
    shape: List[int]
    dtype: str
    layers: int = 1  # 该权重在多少层中出现（默认为1）
    parallel_strategy: str = "replicated"
    world_size: int = 0  # 并行切分大小（0表示不切分）

@dataclass
class AttentionConfig:
    """Attention 配置"""
    type: str  # MHA, MQA, GQA, SWA, MLA, DSA, GDN, UNKNOWN
    num_layers: int
    components: Dict[str, Dict[str, WeightInfo]]

@dataclass
class FFNConfig:
    """FFN 配置，支持多种子类型"""
    router_expert: Optional[ExpertConfig] = None
    shared_expert: Optional[ExpertConfig] = None
    dense_mlp: Optional[DenseMLPConfig] = None

@dataclass
class ExpertConfig:
    """Expert 配置（router_expert 或 shared_expert）"""
    layer_count: int
    count_per_layer: int
    weights: Dict[str, WeightInfo]  # 灵活存储所有权重，如 gate_proj_weight, down_proj_weight 等

@dataclass
class DenseMLPConfig:
    """Dense MLP 配置"""
    layer_count: int
    count_per_layer: int
    weights: Dict[str, WeightInfo]  # 灵活存储所有权重，如 gate_proj_weight, down_proj_weight 等

@dataclass
class ComputationRules:
    """计算规则"""
    activation_vram: ActivationVRAMConfig
    kv_cache_vram: KVCacheVRAMConfig

@dataclass
class ActivationVRAMConfig:
    ideal_capacity_factor: float
    recommended_capacity_factor: float
    worst_case_capacity_factor: float
    formula: str  # 公式字符串

@dataclass
class KVCacheVRAMConfig:
    formula: str  # 公式字符串

### 2.3 FormulaEvaluator (公式计算器)

**职责**: 解析和计算 computation_rules 中的公式字符串

**接口**:
```python
class FormulaEvaluator:
    @staticmethod
    def evaluate(formula: str, context: Dict[str, Any]) -> float:
        """
        计算公式

        Args:
            formula: 公式字符串，如 "(kv_lora_rank + qk_rope_head_dim) * num_layers * 2 * precision"
            context: 变量上下文，包含公式中需要的所有变量

        Returns:
            计算结果

        示例:
            formula = "(kv_lora_rank + qk_rope_head_dim) * num_layers * 2 * precision"
            context = {
                'kv_lora_rank': 512,
                'qk_rope_head_dim': 64,
                'num_layers': 62,
                'precision': 1
            }
            result = FormulaEvaluator.evaluate(formula, context)
            # result = (512 + 64) * 62 * 2 * 1 = 71424
        """
        # 安全的公式计算，只允许基本的数学运算
        # 使用 ast.parse 解析公式，避免 eval 的安全风险
        pass
```

### 2.4 MemoryEstimator (主控制器)

**职责**: 协调配置加载、显存计算、结果汇总

**接口**:
```python
class MemoryEstimator:
    def __init__(self, model_config: ModelConfig, parallel_config: ParallelConfig):
        self.model_config = model_config
        self.parallel_config = parallel_config

    def estimate(self, batch_size: int, seq_len: int,
                chip_config: ChipConfig) -> MemoryResult:
        """
        主估算函数

        计算步骤：
        1. 计算权重显存（遍历 modules）
        2. 计算 KV Cache 显存（使用公式）
        3. 计算激活值显存（使用公式）
        4. 计算系统保留显存
        5. 计算其他显存
        6. 汇总结果
        """
        # 1. 计算权重显存
        weight_memory = self._calculate_weight_memory()

        # 2. 计算 KV Cache 显存
        kv_cache_memory = self._calculate_kv_cache(batch_size, seq_len)

        # 3. 计算激活值显存
        activation_memory = self._calculate_activation_memory(batch_size, seq_len)

        # 4. 计算系统保留显存
        system_reserved = self._calculate_system_reserved(chip_config.vram_gb)

        # 5. 计算其他显存
        other_memory = self._calculate_other_memory(chip_config.vram_gb)

        # 6. 汇总
        total_memory = (weight_memory + kv_cache_memory + activation_memory +
                       system_reserved + other_memory)

        return MemoryResult(
            total_memory_gb=total_memory,
            weight_memory_gb=weight_memory,
            kv_cache_memory_gb=kv_cache_memory,
            activation_memory_gb=activation_memory,
            system_reserved_gb=system_reserved,
            other_memory_gb=other_memory,
            available_memory_gb=chip_config.vram_gb - total_memory,
            breakdown=self._generate_breakdown()  # 详细分解
        )

    def _calculate_weight_memory(self) -> float:
        """
        计算权重显存占用

        遍历 modules 下的所有权重：
        1. embedding: embed_tokens, lm_head, shared_head
        2. norm: 各种 norm 层
        3. attention: q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj
        4. ffn: router_expert, shared_expert, dense_mlp
        5. others: eh_proj 等

        对每个权重：
        - 计算参数量 = product(shape) * count
        - 获取数据类型字节数
        - 应用并行策略
        """
        total_weight_gb = 0.0

        # 遍历 embedding
        for weight_name, weight_info in self.model_config.modules.embedding.items():
            memory_gb = self._calculate_single_weight(weight_info)
            total_weight_gb += memory_gb

        # 遍历 norm
        for weight_name, weight_info in self.model_config.modules.norm.items():
            memory_gb = self._calculate_single_weight(weight_info)
            total_weight_gb += memory_gb

        # 遍历 attention
        attention = self.model_config.modules.attention
        for component_name, component_weights in attention.components.items():
            for weight_name, weight_info in component_weights.items():
                memory_gb = self._calculate_single_weight(weight_info)
                # attention 有 num_layers 层
                total_weight_gb += memory_gb * attention.num_layers

        # 遍历 ffn
        ffn = self.model_config.modules.ffn
        if ffn.router_expert:
            total_weight_gb += self._calculate_expert_memory(ffn.router_expert)
        if ffn.shared_expert:
            total_weight_gb += self._calculate_expert_memory(ffn.shared_expert)
        if ffn.dense_mlp:
            total_weight_gb += self._calculate_dense_mlp_memory(ffn.dense_mlp)

        # 遍历 others
        for weight_name, weight_info in self.model_config.modules.others.items():
            memory_gb = self._calculate_single_weight(weight_info)
            total_weight_gb += memory_gb

        return total_weight_gb

    def _calculate_single_weight(self, weight_info: WeightInfo) -> float:
        """计算单个权重的显存占用"""
        # 计算参数量
        params = 1
        for dim in weight_info.shape:
            params *= dim

        # 获取数据类型字节数
        dtype_bytes = self._get_dtype_bytes(weight_info.dtype)

        # 计算显存 (GB)，乘以 layers（该权重在多少层中出现）
        memory_gb = params * dtype_bytes * weight_info.layers / (1024**3)

        # 应用并行策略
        memory_gb = self._apply_parallel_strategy(
            memory_gb,
            weight_info.parallel_strategy,
            weight_info.world_size
        )

        return memory_gb

    def _calculate_expert_memory(self, expert_config: ExpertConfig) -> float:
        """计算 Expert 的显存占用"""
        total_memory = 0.0

        # 遍历所有权重（如 gate_proj_weight, down_proj_weight, up_proj_weight 等）
        for weight_name, weight_info in expert_config.weights.items():
            memory_gb = self._calculate_single_weight(weight_info)
            # expert 有 layer_count 层，每层有 count_per_layer 个
            total_memory += memory_gb * expert_config.layer_count * expert_config.count_per_layer

        return total_memory

    def _calculate_dense_mlp_memory(self, dense_config: DenseMLPConfig) -> float:
        """计算 Dense MLP 的显存占用"""
        total_memory = 0.0

        # 遍历所有权重（如 gate_proj_weight, down_proj_weight, up_proj_weight 等）
        for weight_name, weight_info in dense_config.weights.items():
            memory_gb = self._calculate_single_weight(weight_info)
            total_memory += memory_gb * dense_config.layer_count * dense_config.count_per_layer

        return total_memory

    def _calculate_kv_cache(self, batch_size: int, seq_len: int) -> float:
        """
        计算 KV Cache 显存占用

        使用 computation_rules.kv_cache_vram.formula 计算
        """
        formula = self.model_config.computation_rules.kv_cache_vram.formula

        # 构建上下文
        context = {
            'batch_size': batch_size,
            'seq_len': seq_len,
            **self.model_config.architecture_config,
            'precision': self._get_precision()
        }

        # 计算公式
        kv_cache_bytes = FormulaEvaluator.evaluate(formula, context)
        kv_cache_gb = kv_cache_bytes / (1024**3)

        # 应用并行策略
        kv_cache_gb = kv_cache_gb / self.parallel_config.tp / self.parallel_config.cp

        return kv_cache_gb

    def _calculate_activation_memory(self, batch_size: int, seq_len: int) -> float:
        """
        计算激活值显存占用

        使用 computation_rules.activation_vram.formula 计算
        """
        formula = self.model_config.computation_rules.activation_vram.formula
        capacity_factor = self.model_config.computation_rules.activation_vram.recommended_capacity_factor

        # 构建上下文
        context = {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'S': seq_len,  # 别名
            'capacity_factor': capacity_factor,
            'recommended_capacity_factor': capacity_factor,
            **self.model_config.architecture_config,
            'precision': 2  # 激活值通常使用 BF16
        }

        # 计算公式
        activation_bytes = FormulaEvaluator.evaluate(formula, context)
        activation_gb = activation_bytes / (1024**3)

        # 应用并行策略
        activation_gb = activation_gb / self.parallel_config.ep

        return activation_gb

    def _calculate_system_reserved(self, vram_gb: float) -> float:
        """计算系统保留显存，默认 5%"""
        return vram_gb * 0.05

    def _calculate_other_memory(self, vram_gb: float) -> float:
        """计算其他显存（通信 buffer, 框架开销等），默认 5%"""
        return vram_gb * 0.05

    def _get_dtype_bytes(self, dtype: str) -> float:
        """获取数据类型的字节数"""
        dtype_map = {
            'BF16': 2, 'FP16': 2,
            'F8_E4M3': 1, 'INT8': 1, 'INT4': 0.5,
            'F32': 4, 'FP32': 4
        }
        return dtype_map.get(dtype, 2)

    def _get_precision(self) -> int:
        """获取 KV Cache 的精度（字节数）"""
        quantization = self.model_config.model_identity.quantization
        if 'C8' in quantization or 'FP8' in quantization:
            return 1
        return 2

    def _apply_parallel_strategy(self, memory_gb: float, strategy: str, world_size: int = 0) -> float:
        """
        根据并行策略调整显存占用

        Args:
            memory_gb: 原始显存占用
            strategy: 并行策略 (replicated, tp_col, tp_row, expert_sharded, pp)
            world_size: 并行世界大小（0表示使用全局配置）
        """
        if strategy == 'replicated':
            return memory_gb
        elif strategy in ['tp_col', 'tp_row']:
            tp_size = world_size if world_size > 0 else self.parallel_config.tp
            return memory_gb / tp_size
        elif strategy == 'expert_sharded':
            ep_size = world_size if world_size > 0 else self.parallel_config.ep
            return memory_gb / ep_size
        elif strategy == 'pp':
            return memory_gb / self.parallel_config.pp
        else:
            return memory_gb

    def _generate_breakdown(self) -> Dict[str, Any]:
        """生成详细的显存分解"""
        # 返回各模块的详细显存占用
        pass

    def estimate_max_sequence_length(self, batch_size: int, chip_config: ChipConfig) -> int:
        """估算给定 batch_size 下的最大序列长度（二分搜索）"""
        left, right = 1024, 1000000
        max_seq_len = 0

        while left <= right:
            mid = (left + right) // 2
            result = self.estimate(batch_size, mid, chip_config)

            if result.total_memory_gb <= chip_config.vram_gb:
                max_seq_len = mid
                left = mid + 1
            else:
                right = mid - 1

        return max_seq_len

    def estimate_max_batch_size(self, seq_len: int, chip_config: ChipConfig) -> int:
        """估算给定 seq_len 下的最大 batch size（二分搜索）"""
        left, right = 1, 1024
        max_batch_size = 0

        while left <= right:
            mid = (left + right) // 2
            result = self.estimate(mid, seq_len, chip_config)

            if result.total_memory_gb <= chip_config.vram_gb:
                max_batch_size = mid
                left = mid + 1
            else:
                right = mid - 1

        return max_batch_size

### 2.5 ParallelConfig (并行配置)

```python
@dataclass
class ParallelConfig:
    tp: int = 1   # Tensor Parallel
    pp: int = 1   # Pipeline Parallel
    dp: int = 1   # Data Parallel
    ep: int = 1   # Expert Parallel
    sp: int = 1   # Sequence Parallel
    cp: int = 1   # Context Parallel

    @property
    def total_gpus(self) -> int:
        return self.tp * self.pp * self.dp

    def validate(self) -> None:
        """验证并行配置的合理性"""
        if any(x <= 0 for x in [self.tp, self.pp, self.dp, self.ep, self.sp, self.cp]):
            raise ParallelConfigError("All parallel dimensions must be positive")
```

### 2.6 MemoryResult (结果数据类)

```python
@dataclass
class MemoryResult:
    total_memory_gb: float
    weight_memory_gb: float
    kv_cache_memory_gb: float
    activation_memory_gb: float
    system_reserved_gb: float
    other_memory_gb: float
    available_memory_gb: float
    breakdown: Dict[str, Any]  # 详细分解

    def to_dict(self) -> Dict[str, float]:
        """转换为字典格式"""
        return {
            'total_memory_gb': round(self.total_memory_gb, 2),
            'weight_memory_gb': round(self.weight_memory_gb, 2),
            'kv_cache_memory_gb': round(self.kv_cache_memory_gb, 2),
            'activation_memory_gb': round(self.activation_memory_gb, 2),
            'system_reserved_gb': round(self.system_reserved_gb, 2),
            'other_memory_gb': round(self.other_memory_gb, 2),
            'available_memory_gb': round(self.available_memory_gb, 2)
        }
```

### 2.7 ReportGenerator (报告生成器)

**职责**: 生成 Markdown 格式的详细报告

**接口**:
```python
class ReportGenerator:
    @staticmethod
    def generate_report(result: MemoryResult,
                       model_config: ModelConfig,
                       parallel_config: ParallelConfig,
                       chip_config: ChipConfig,
                       batch_size: int,
                       seq_len: int) -> str:
        """
        生成 Markdown 报告

        报告内容：
        1. 模型信息
        2. 硬件配置
        3. 并行策略
        4. 显存占用总览（表格）
        5. 各组件详细分解（表格）
        6. 最大序列长度和 batch size 估算
        """
        report = []

        # 标题
        report.append(f"# {model_config.model_identity.name} 显存占用估算报告\n")

        # 模型信息
        report.append("## 1. 模型信息\n")
        report.append(f"- **模型名称**: {model_config.model_identity.name}")
        report.append(f"- **总参数量**: {model_config.model_identity.total_params_b}B")
        report.append(f"- **层数**: {model_config.model_identity.num_layers}")
        report.append(f"- **量化方式**: {model_config.model_identity.quantization}\n")

        # 硬件配置
        report.append("## 2. 硬件配置\n")
        report.append(f"- **芯片型号**: {chip_config.name}")
        report.append(f"- **单卡显存**: {chip_config.vram_gb} GB")
        report.append(f"- **显存带宽**: {chip_config.bandwidth_gb_s} GB/s\n")

        # 并行策略
        report.append("## 3. 并行策略\n")
        report.append(f"- **TP (Tensor Parallel)**: {parallel_config.tp}")
        report.append(f"- **PP (Pipeline Parallel)**: {parallel_config.pp}")
        report.append(f"- **DP (Data Parallel)**: {parallel_config.dp}")
        report.append(f"- **EP (Expert Parallel)**: {parallel_config.ep}")
        report.append(f"- **总 GPU 数**: {parallel_config.total_gpus}\n")

        # 显存占用总览
        report.append("## 4. 显存占用总览\n")
        report.append(f"**配置**: batch_size={batch_size}, seq_len={seq_len}\n")
        report.append("| 组件 | 显存占用 (GB) | 占比 |")
        report.append("|------|--------------|------|")

        total = result.total_memory_gb
        report.append(f"| 模型权重 | {result.weight_memory_gb:.2f} | {result.weight_memory_gb/total*100:.1f}% |")
        report.append(f"| KV Cache | {result.kv_cache_memory_gb:.2f} | {result.kv_cache_memory_gb/total*100:.1f}% |")
        report.append(f"| 激活值 | {result.activation_memory_gb:.2f} | {result.activation_memory_gb/total*100:.1f}% |")
        report.append(f"| 系统保留 | {result.system_reserved_gb:.2f} | {result.system_reserved_gb/total*100:.1f}% |")
        report.append(f"| 其他 | {result.other_memory_gb:.2f} | {result.other_memory_gb/total*100:.1f}% |")
        report.append(f"| **总计** | **{result.total_memory_gb:.2f}** | **100%** |")
        report.append(f"| 可用显存 | {result.available_memory_gb:.2f} | - |\n")

        # 详细分解
        report.append("## 5. 各组件详细分解\n")
        report.append(ReportGenerator._generate_breakdown_table(result.breakdown))

        return "\n".join(report)

    @staticmethod
    def _generate_breakdown_table(breakdown: Dict[str, Any]) -> str:
        """生成详细分解表格"""
        # 实现详细分解表格生成
        pass
```

### 2.8 ChipConfig (芯片配置)

```python
@dataclass
class ChipConfig:
    name: str
    vram_gb: float
    bandwidth_gb_s: float

    @staticmethod
    def from_json(chip_name: str, chips_json_path: str = "configs/chips.json") -> 'ChipConfig':
        """从 chips.json 加载芯片配置"""
        with open(chips_json_path, 'r') as f:
            chips_data = json.load(f)

        if chip_name not in chips_data:
            raise ValueError(f"Unknown chip: {chip_name}")

        chip_data = chips_data[chip_name]
        return ChipConfig(
            name=chip_name,
            vram_gb=chip_data['vram_gb'],
            bandwidth_gb_s=chip_data['bandwidth_gb_s']
        )
```

### 2.9 WeightClassifier (权重分类器)

**职责**: 将模型权重名称分类到标准的 YAML 结构（embedding, attention, ffn, norm, others）

**接口**:
```python
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class ClassificationResult:
    """权重分类结果"""
    category: str        # embedding, attention, ffn, norm, others
    subcategory: str     # q_proj, gate_proj, etc.
    layer_idx: Optional[int] = None
    expert_idx: Optional[int] = None

class WeightClassifier:
    """
    权重分类器，基于规则将权重名称映射到标准结构

    支持：
    1. 通用规则（适用于大多数 Transformer 模型）
    2. 模型特定规则（如 DeepSeek MLA）
    3. 自动检测模型类型
    """

    def __init__(self, model_type: str = 'generic',
                 rules_path: str = 'configs/weight_mapping_rules.yaml'):
        """
        初始化分类器

        Args:
            model_type: 模型类型 (generic, deepseek, llama, qwen, etc.)
            rules_path: 规则配置文件路径
        """
        self.model_type = model_type
        self.rules = self._load_rules(rules_path)

    def classify(self, weight_name: str) -> ClassificationResult:
        """
        分类权重名称

        Args:
            weight_name: 权重名称，如 "model.layers.0.self_attn.q_proj.weight"

        Returns:
            ClassificationResult: 分类结果

        示例:
            classifier = WeightClassifier(model_type='llama')
            result = classifier.classify("model.layers.0.self_attn.q_proj.weight")
            # result.category = 'attention'
            # result.subcategory = 'q_proj'
            # result.layer_idx = 0
        """
        # 提取层号和专家编号
        layer_idx = self._extract_layer_idx(weight_name)
        expert_idx = self._extract_expert_idx(weight_name)

        # 尝试匹配规则（特定规则优先）
        for category in ['embedding', 'attention', 'ffn', 'norm']:
            if category in self.rules:
                result = self._match_category(weight_name, category, self.rules[category])
                if result:
                    return ClassificationResult(
                        category=category,
                        subcategory=result,
                        layer_idx=layer_idx,
                        expert_idx=expert_idx
                    )

        # 默认归类到 others
        return ClassificationResult(
            category='others',
            subcategory=self._simplify_name(weight_name),
            layer_idx=layer_idx
        )

    def _extract_layer_idx(self, weight_name: str) -> Optional[int]:
        """提取层号"""
        patterns = [
            r'\.layers?\.(\d+)\.',
            r'\.h\.(\d+)\.',
            r'\.block\.(\d+)\.',
        ]
        for pattern in patterns:
            match = re.search(pattern, weight_name)
            if match:
                return int(match.group(1))
        return None

    def _extract_expert_idx(self, weight_name: str) -> Optional[int]:
        """提取专家编号"""
        match = re.search(r'\.experts?\.(\d+)\.', weight_name)
        if match:
            return int(match.group(1))
        return None
```

**规则配置文件** (`configs/weight_mapping_rules.yaml`):
```yaml
# 通用规则（适用于大多数 Transformer 模型）
generic:
  embedding:
    embed_tokens:
      patterns:
        - ".*embed_tokens.*"
        - ".*wte.*"              # GPT style
        - ".*word_embeddings.*"  # BERT style
    lm_head:
      patterns:
        - ".*lm_head.*"
        - ".*output\\.weight"

  attention:
    q_proj:
      patterns: [".*\\.q_proj.*", ".*\\.query.*"]
    k_proj:
      patterns: [".*\\.k_proj.*", ".*\\.key.*"]
    v_proj:
      patterns: [".*\\.v_proj.*", ".*\\.value.*"]
    o_proj:
      patterns: [".*\\.o_proj.*", ".*\\.dense.*"]

  ffn:
    gate_proj:
      patterns: [".*\\.gate_proj.*", ".*\\.w1.*", ".*\\.fc1.*"]
    up_proj:
      patterns: [".*\\.up_proj.*", ".*\\.w3.*"]
    down_proj:
      patterns: [".*\\.down_proj.*", ".*\\.w2.*", ".*\\.fc2.*"]

  norm:
    patterns:
      - ".*input_layernorm.*"
      - ".*post_attention_layernorm.*"
      - ".*norm.*"
      - ".*ln_.*"

# DeepSeek 特定规则（MLA 架构）
deepseek:
  attention:
    q_a_proj:
      patterns: [".*\\.q_a_proj\\.weight"]
    q_a_proj_scale:
      patterns: [".*\\.q_a_proj\\.weight_scale_inv"]
    q_a_layernorm:
      patterns: [".*\\.q_a_layernorm.*"]
    q_b_proj:
      patterns: [".*\\.q_b_proj\\.weight"]
    kv_a_proj_with_mqa:
      patterns: [".*\\.kv_a_proj_with_mqa\\.weight"]
    kv_b_proj:
      patterns: [".*\\.kv_b_proj.*"]
    o_proj:
      patterns: [".*\\.o_proj.*"]

  ffn:
    router_expert:
      patterns: [".*\\.experts\\.(\\d+)\\..*"]
      exclude_patterns: [".*shared_experts.*"]
    shared_expert:
      patterns: [".*\\.shared_experts.*"]
    dense_mlp:
      patterns: [".*\\.mlp\\.(gate_proj|up_proj|down_proj).*"]
```

### 2.10 ModelDetector (模型类型检测器)

**职责**: 自动检测模型类型

**接口**:
```python
class ModelDetector:
    @staticmethod
    def detect_model_type(config_json: Dict, weight_names: List[str]) -> str:
        """
        自动检测模型类型

        优先级：
        1. config.json 中的 model_type 或 architectures
        2. 权重名称特征
        3. 默认 generic

        Args:
            config_json: 模型的 config.json 内容
            weight_names: 所有权重名称列表

        Returns:
            模型类型字符串 (generic, deepseek, llama, qwen, etc.)
        """
        # 1. 检查 config.json
        if 'model_type' in config_json:
            model_type = config_json['model_type'].lower()
            if 'deepseek' in model_type:
                return 'deepseek'
            elif 'llama' in model_type:
                return 'llama'
            elif 'qwen' in model_type:
                return 'qwen'

        if 'architectures' in config_json:
            arch = config_json['architectures'][0].lower()
            if 'deepseek' in arch:
                return 'deepseek'
            elif 'llama' in arch:
                return 'llama'
            elif 'qwen' in arch:
                return 'qwen'

        # 2. 检查权重名称特征
        weight_features = {
            'deepseek': ['kv_a_proj_with_mqa', 'q_a_proj', 'shared_experts'],
            'llama': ['self_attn', 'mlp.gate_proj'],
            'qwen': ['c_attn', 'c_proj'],
        }

        for model_type, features in weight_features.items():
            if any(any(feature in name for name in weight_names) for feature in features):
                return model_type

        # 3. 默认
        return 'generic'
```

### 2.11 配置生成流程

**在 ConfigLoader 中实现配置生成**:

```python
class ConfigLoader:
    @staticmethod
    def generate_config_from_weights(repo_id: str, token: str = None) -> ModelConfig:
        """
        从 HuggingFace 或本地权重目录生成配置文件

        实现步骤：
        1. 获取权重 metadata
        2. 加载 config.json
        3. 自动检测模型类型
        4. 创建权重分类器
        5. 分类所有权重
        6. 统计层数、expert 数量等
        7. 生成标准 YAML 配置

        Args:
            repo_id: HuggingFace repo ID 或本地路径
            token: HuggingFace token（可选）

        Returns:
            ModelConfig: 生成的模型配置
        """
        from huggingface_hub import get_safetensors_metadata
        from collections import defaultdict

        # 1. 获取 metadata
        metadata = get_safetensors_metadata(repo_id, token=token)
        all_tensors = {}
        f_meta = metadata.files_metadata
        for tensor_name, file_name in metadata.weight_map.items():
            file_obj = f_meta[file_name] if isinstance(f_meta, dict) else \
                       next(f for f in f_meta if getattr(f, 'file_name', '') == file_name)
            if tensor_name in file_obj.tensors:
                all_tensors[tensor_name] = file_obj.tensors[tensor_name]

        # 2. 加载 config.json
        config_json = ConfigLoader._load_config_json(repo_id, token)

        # 3. 自动检测模型类型
        model_type = ModelDetector.detect_model_type(config_json, list(all_tensors.keys()))
        print(f"检测到模型类型: {model_type}")

        # 4. 创建分类器
        classifier = WeightClassifier(model_type=model_type)

        # 5. 分类所有权重
        categorized_weights = defaultdict(lambda: defaultdict(list))
        for weight_name, weight_info in all_tensors.items():
            result = classifier.classify(weight_name)
            categorized_weights[result.category][result.subcategory].append({
                'name': weight_name,
                'shape': list(weight_info.shape),
                'dtype': str(weight_info.dtype),
                'layer_idx': result.layer_idx,
                'expert_idx': result.expert_idx
            })

        # 6. 生成配置
        yaml_config = ConfigLoader._build_yaml_config(
            categorized_weights,
            config_json,
            model_type
        )

        return yaml_config

    @staticmethod
    def _build_yaml_config(categorized_weights: Dict,
                          config_json: Dict,
                          model_type: str) -> ModelConfig:
        """
        从分类后的权重构建 YAML 配置

        处理逻辑：
        1. embedding: 直接映射，layers=1
        2. norm: 统计每个权重出现的层数，设置 layers 字段
        3. attention: 统计层数，设置 num_layers
        4. ffn: 区分 router_expert, shared_expert, dense_mlp
           - 统计 layer_count 和 count_per_layer
           - 为每种权重类型创建 WeightInfo（layers=1）
        5. others: 直接映射
        """
        # 实现细节...
        pass
```

## 3. 错误处理策略

### 3.1 配置验证错误

```python
class ConfigValidationError(Exception):
    """配置验证失败"""
    pass

class ConfigLoader:
    @staticmethod
    def validate_config(config: ModelConfig) -> None:
        """验证配置完整性"""
        # 1. 验证必需字段
        if not config.model_identity.name:
            raise ConfigValidationError("model_identity.name is required")

        # 2. 验证 attention 类型
        valid_attention_types = ['MHA', 'MQA', 'GQA', 'SWA', 'MLA', 'DSA', 'GDN', 'UNKNOWN']
        if config.modules.attention.type not in valid_attention_types:
            raise ConfigValidationError(f"Invalid attention type: {config.modules.attention.type}")

        # 3. 验证权重形状
        for weight_info in config.modules.embedding.values():
            if not weight_info.shape or len(weight_info.shape) == 0:
                raise ConfigValidationError("Weight shape cannot be empty")

        # 4. 验证公式字段
        if not config.computation_rules.kv_cache_vram.formula:
            raise ConfigValidationError("kv_cache_vram.formula is required")
```

### 3.2 公式计算错误

```python
class FormulaEvaluationError(Exception):
    """公式计算失败"""
    pass

class FormulaEvaluator:
    @staticmethod
    def evaluate(formula: str, context: Dict[str, Any]) -> float:
        try:
            # 使用 ast.parse 安全解析公式
            # 只允许基本的数学运算符: +, -, *, /, //, %, **
            # 不允许函数调用、属性访问等危险操作
            pass
        except Exception as e:
            raise FormulaEvaluationError(f"Failed to evaluate formula '{formula}': {e}")
```

### 3.3 并行配置错误

```python
class ParallelConfigError(Exception):
    """并行配置错误"""
    pass
```

### 3.4 芯片配置错误

```python
class ChipConfigError(Exception):
    """芯片配置不存在或无效"""
    pass
```

## 4. 测试策略

### 4.1 单元测试

**测试文件**: `tests/test_config_loader.py`
```python
def test_load_model_config():
    """测试加载模型配置"""
    config = ConfigLoader.load_model_config("configs/models/deepseek_r1.yaml")
    assert config.model_identity.name == "DeepSeek-R1"
    assert config.model_identity.total_params_b == 671

def test_validate_config_missing_fields():
    """测试配置验证 - 缺失字段"""
    config = ModelConfig(...)  # 缺失必需字段
    with pytest.raises(ConfigValidationError):
        ConfigLoader.validate_config(config)
```

**测试文件**: `tests/test_formula_evaluator.py`
```python
def test_evaluate_simple_formula():
    """测试简单公式计算"""
    formula = "a + b * c"
    context = {'a': 10, 'b': 5, 'c': 2}
    result = FormulaEvaluator.evaluate(formula, context)
    assert result == 20

def test_evaluate_complex_formula():
    """测试复杂公式（DeepSeek-R1 KV Cache）"""
    formula = "(kv_lora_rank + qk_rope_head_dim) * num_layers * 2 * precision"
    context = {
        'kv_lora_rank': 512,
        'qk_rope_head_dim': 64,
        'num_layers': 62,
        'precision': 1
    }
    result = FormulaEvaluator.evaluate(formula, context)
    assert result == 71424
```

**测试文件**: `tests/test_memory_estimator.py`
```python
def test_calculate_weight_memory():
    """测试权重显存计算"""
    config = ConfigLoader.load_model_config("configs/models/deepseek_r1.yaml")
    parallel_config = ParallelConfig(tp=8, ep=8)
    estimator = MemoryEstimator(config, parallel_config)

    weight_memory = estimator._calculate_weight_memory()
    assert weight_memory > 0
    assert weight_memory < 1000  # 合理范围

def test_estimate_full():
    """测试完整估算流程"""
    config = ConfigLoader.load_model_config("configs/models/deepseek_r1.yaml")
    parallel_config = ParallelConfig(tp=8, ep=8)
    chip_config = ChipConfig.from_json("H100_80GB")
    estimator = MemoryEstimator(config, parallel_config)

    result = estimator.estimate(batch_size=1, seq_len=8192, chip_config=chip_config)

    assert result.total_memory_gb > 0
    assert result.total_memory_gb <= chip_config.vram_gb
    assert result.weight_memory_gb > 0
    assert result.kv_cache_memory_gb > 0
```

### 4.2 集成测试

**测试文件**: `tests/test_integration.py`
```python
def test_end_to_end_deepseek_r1():
    """端到端测试 - DeepSeek-R1"""
    # 1. 加载配置
    config = ConfigLoader.load_model_config("configs/models/deepseek_r1.yaml")
    parallel_config = ParallelConfig(tp=8, ep=8)
    chip_config = ChipConfig.from_json("H100_80GB")

    # 2. 估算显存
    estimator = MemoryEstimator(config, parallel_config)
    result = estimator.estimate(batch_size=1, seq_len=8192, chip_config=chip_config)

    # 3. 生成报告
    report = ReportGenerator.generate_report(
        result, config, parallel_config, chip_config, 1, 8192
    )

    # 4. 验证报告内容
    assert "DeepSeek-R1" in report
    assert "H100_80GB" in report
    assert "模型权重" in report
```

**测试文件**: `tests/test_weight_classifier.py`
```python
def test_classify_llama_weights():
    """测试 Llama 模型权重分类"""
    classifier = WeightClassifier(model_type='llama')

    # 测试 attention 权重
    result = classifier.classify("model.layers.0.self_attn.q_proj.weight")
    assert result.category == 'attention'
    assert result.subcategory == 'q_proj'
    assert result.layer_idx == 0

    # 测试 FFN 权重
    result = classifier.classify("model.layers.5.mlp.gate_proj.weight")
    assert result.category == 'ffn'
    assert result.subcategory == 'gate_proj'
    assert result.layer_idx == 5

def test_classify_deepseek_weights():
    """测试 DeepSeek 模型权重分类"""
    classifier = WeightClassifier(model_type='deepseek')

    # 测试 MLA attention 权重
    result = classifier.classify("model.layers.0.self_attn.q_a_proj.weight")
    assert result.category == 'attention'
    assert result.subcategory == 'q_a_proj'

    # 测试 expert 权重
    result = classifier.classify("model.layers.10.mlp.experts.5.gate_proj.weight")
    assert result.category == 'ffn'
    assert result.subcategory == 'router_expert'
    assert result.layer_idx == 10
    assert result.expert_idx == 5

def test_model_detector():
    """测试模型类型检测"""
    config_json = {'model_type': 'LlamaForCausalLM'}
    weight_names = ['model.layers.0.self_attn.q_proj.weight']

    model_type = ModelDetector.detect_model_type(config_json, weight_names)
    assert model_type == 'llama'
```

### 4.3 配置文件测试

**测试文件**: `tests/test_configs.py`
```python
def test_all_model_configs_valid():
    """测试所有模型配置文件的有效性"""
    config_dir = Path("configs/models")
    for config_file in config_dir.glob("*.yaml"):
        config = ConfigLoader.load_model_config(str(config_file))
        ConfigLoader.validate_config(config)  # 不应抛出异常
```

## 5. 使用示例

### 5.1 基本使用

```python
from llm_mem_estimator import MemoryEstimator, ConfigLoader, ParallelConfig, ChipConfig

# 1. 加载配置
model_config = ConfigLoader.load_model_config("configs/models/deepseek_r1.yaml")
parallel_config = ParallelConfig(tp=8, ep=8, pp=1, dp=1)
chip_config = ChipConfig.from_json("H100_80GB")

# 2. 创建估算器
estimator = MemoryEstimator(model_config, parallel_config)

# 3. 估算显存占用
result = estimator.estimate(
    batch_size=1,
    seq_len=8192,
    chip_config=chip_config
)

# 4. 打印结果
print(f"总显存占用: {result.total_memory_gb:.2f} GB")
print(f"权重显存: {result.weight_memory_gb:.2f} GB")
print(f"KV Cache: {result.kv_cache_memory_gb:.2f} GB")
print(f"激活值: {result.activation_memory_gb:.2f} GB")
```

### 5.2 估算最大序列长度

```python
# 估算在给定 batch_size 下的最大序列长度
max_seq_len = estimator.estimate_max_sequence_length(
    batch_size=1,
    chip_config=chip_config
)
print(f"最大序列长度: {max_seq_len}")
```

### 5.3 估算最大 Batch Size

```python
# 估算在给定 seq_len 下的最大 batch size
max_batch_size = estimator.estimate_max_batch_size(
    seq_len=8192,
    chip_config=chip_config
)
print(f"最大 Batch Size: {max_batch_size}")
```

### 5.4 生成报告并保存

```python
from llm_mem_estimator import ReportGenerator

# 生成报告
report = ReportGenerator.generate_report(
    result=result,
    model_config=model_config,
    parallel_config=parallel_config,
    chip_config=chip_config,
    batch_size=1,
    seq_len=8192
)

# 保存报告（使用模型名称命名）
output_filename = f"{model_config.model_identity.name}_memory_report.md"
with open(output_filename, "w") as f:
    f.write(report)
print(f"报告已保存到: {output_filename}")
# 输出: 报告已保存到: DeepSeek-R1_memory_report.md
```
```

### 5.5 从 HuggingFace 生成配置

```python
# 从 HuggingFace 模型生成配置文件
config = ConfigLoader.generate_config_from_weights(
    repo_id="deepseek-ai/DeepSeek-R1",
    token="your_hf_token"
)

# 保存配置
import yaml
with open("configs/models/deepseek_r1_generated.yaml", "w") as f:
    yaml.dump(config, f)
```

## 6. CLI 接口设计

### 6.1 命令行参数

```bash
python scripts/calculate_mem.py \
    --model configs/models/deepseek_r1.yaml \
    --chip H100_80GB \
    --tp 8 \
    --ep 8 \
    --pp 1 \
    --dp 1 \
    --batch-size 1 \
    --seq-len 8192
# 自动生成报告: DeepSeek-R1_memory_report.md
```

或指定输出路径：
```bash
python scripts/calculate_mem.py \
    --model configs/models/deepseek_r1.yaml \
    --chip H100_80GB \
    --tp 8 \
    --ep 8 \
    --batch-size 1 \
    --seq-len 8192 \
    --output /path/to/custom_report.md
```

### 6.2 参数说明

| 参数 | 说明 | 默认值 | 必需 |
|------|------|--------|------|
| `--model` | 模型配置文件路径 | - | 是 |
| `--chip` | 芯片型号 | - | 是 |
| `--tp` | Tensor Parallel 大小 | 1 | 否 |
| `--ep` | Expert Parallel 大小 | 1 | 否 |
| `--pp` | Pipeline Parallel 大小 | 1 | 否 |
| `--dp` | Data Parallel 大小 | 1 | 否 |
| `--sp` | Sequence Parallel 大小 | 1 | 否 |
| `--cp` | Context Parallel 大小 | 1 | 否 |
| `--batch-size` | Batch Size | 1 | 否 |
| `--seq-len` | 序列长度 | 8192 | 否 |
| `--output` | 输出报告路径（不指定则使用 `<model_name>_memory_report.md`） | `<model_name>_memory_report.md` | 否 |
| `--estimate-max-seq` | 估算最大序列长度 | False | 否 |
| `--estimate-max-batch` | 估算最大 Batch Size | False | 否 |

### 6.3 CLI 实现

```python
# scripts/calculate_mem.py
import argparse
from llm_mem_estimator import MemoryEstimator, ConfigLoader, ParallelConfig, ChipConfig, ReportGenerator

def main():
    parser = argparse.ArgumentParser(description="LLM Memory Estimator")
    parser.add_argument("--model", required=True, help="Model config YAML path")
    parser.add_argument("--chip", required=True, help="Chip name (e.g., H100_80GB)")
    parser.add_argument("--tp", type=int, default=1, help="Tensor Parallel size")
    parser.add_argument("--ep", type=int, default=1, help="Expert Parallel size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline Parallel size")
    parser.add_argument("--dp", type=int, default=1, help="Data Parallel size")
    parser.add_argument("--sp", type=int, default=1, help="Sequence Parallel size")
    parser.add_argument("--cp", type=int, default=1, help="Context Parallel size")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=8192, help="Sequence length")
    parser.add_argument("--output", help="Output report path (default: stdout)")
    parser.add_argument("--estimate-max-seq", action="store_true", help="Estimate max sequence length")
    parser.add_argument("--estimate-max-batch", action="store_true", help="Estimate max batch size")

    args = parser.parse_args()

    # 加载配置
    model_config = ConfigLoader.load_model_config(args.model)
    parallel_config = ParallelConfig(
        tp=args.tp, ep=args.ep, pp=args.pp,
        dp=args.dp, sp=args.sp, cp=args.cp
    )
    chip_config = ChipConfig.from_json(args.chip)

    # 创建估算器
    estimator = MemoryEstimator(model_config, parallel_config)

    # 估算显存
    result = estimator.estimate(args.batch_size, args.seq_len, chip_config)

    # 生成报告
    report = ReportGenerator.generate_report(
        result, model_config, parallel_config, chip_config,
        args.batch_size, args.seq_len
    )

    # 输出报告
    if args.output:
        output_path = args.output
    else:
        # 默认使用模型名称命名: <model_name>_memory_report.md
        model_name = model_config.model_identity.name.replace(" ", "_")
        output_path = f"{model_name}_memory_report.md"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"报告已保存到: {output_path}")

    # 估算最大序列长度
    if args.estimate_max_seq:
        max_seq = estimator.estimate_max_sequence_length(args.batch_size, chip_config)
        print(f"\n最大序列长度 (batch_size={args.batch_size}): {max_seq}")

    # 估算最大 Batch Size
    if args.estimate_max_batch:
        max_batch = estimator.estimate_max_batch_size(args.seq_len, chip_config)
        print(f"\n最大 Batch Size (seq_len={args.seq_len}): {max_batch}")

if __name__ == "__main__":
    main()
```

## 7. 文件组织结构

```
llm_mem_estimator/
├── SKILL.md                    # Skill 接口定义
├── CLAUDE.md                   # 项目说明文档
├── README.md                   # 用户文档
├── setup.py                    # 安装配置
├── requirements.txt            # 依赖列表
├── docs/
│   ├── plan.md                 # 实现计划
│   └── detailed_design.md      # 详细设计（本文档）
├── configs/
│   ├── models/                 # 模型配置文件
│   │   ├── deepseek_r1.yaml
│   │   ├── llama3_70b.yaml
│   │   └── qwen2_72b.yaml
│   ├── chips.json              # 芯片配置
│   └── weight_mapping_rules.yaml  # 权重分类规则
├── llm_mem_estimator/          # 主包
│   ├── __init__.py
│   ├── config_loader.py        # ConfigLoader
│   ├── model_config.py         # ModelConfig 数据类
│   ├── formula_evaluator.py   # FormulaEvaluator
│   ├── memory_estimator.py    # MemoryEstimator
│   ├── report_generator.py    # ReportGenerator
│   ├── parallel_config.py     # ParallelConfig
│   ├── chip_config.py         # ChipConfig
│   ├── weight_classifier.py   # WeightClassifier
│   ├── model_detector.py      # ModelDetector
│   └── exceptions.py          # 自定义异常
├── scripts/
│   └── calculate_mem.py       # CLI 入口
└── tests/
    ├── __init__.py
    ├── test_config_loader.py
    ├── test_formula_evaluator.py
    ├── test_memory_estimator.py
    ├── test_weight_classifier.py
    ├── test_integration.py
    └── test_configs.py
```

## 8. 实现步骤

### 阶段 1: 基础设施 (1-2 天)

1. **创建项目结构**
   - 创建目录结构
   - 创建 `__init__.py` 文件
   - 创建 `setup.py` 和 `requirements.txt`

2. **实现数据类**
   - `model_config.py`: ModelConfig, WeightInfo, AttentionConfig, FFNConfig 等
   - `parallel_config.py`: ParallelConfig
   - `chip_config.py`: ChipConfig
   - `exceptions.py`: 自定义异常类

3. **创建芯片配置文件**
   - `configs/chips.json`: 常见芯片规格

### 阶段 2: 核心功能 (3-4 天)

4. **实现 ConfigLoader**
   - `config_loader.py`: 加载和验证 YAML 配置
   - 实现配置验证逻辑

5. **实现 FormulaEvaluator**
   - `formula_evaluator.py`: 安全的公式解析和计算
   - 使用 `ast.parse` 实现安全计算

6. **实现 MemoryEstimator**
   - `memory_estimator.py`: 核心显存计算逻辑
   - 实现权重、KV Cache、激活值计算
   - 实现并行策略应用
   - 实现最大序列长度和 Batch Size 估算

### 阶段 3: 权重分类和配置生成 (2-3 天)

7. **创建权重分类规则**
   - `configs/weight_mapping_rules.yaml`: 通用规则和模型特定规则
   - 支持 generic, deepseek, llama, qwen 等模型类型

8. **实现 WeightClassifier**
   - `weight_classifier.py`: 权重分类器
   - 实现规则加载和匹配逻辑
   - 实现层号和专家编号提取

9. **实现 ModelDetector**
   - `model_detector.py`: 模型类型自动检测
   - 基于 config.json 和权重名称特征

10. **实现配置生成功能**
    - 在 `config_loader.py` 中实现 `generate_config_from_weights`
    - 集成 HuggingFace `get_safetensors_metadata`
    - 实现权重统计和 YAML 生成逻辑

### 阶段 4: 报告生成 (1-2 天)

11. **实现 ReportGenerator**
   - `report_generator.py`: Markdown 报告生成
   - 实现表格格式化
   - 实现详细分解展示

### 阶段 5: CLI 接口 (1-2 天)

12. **实现 CLI 接口**
   - `scripts/calculate_mem.py`: 命令行入口
   - 参数解析和验证

### 阶段 6: 测试和文档 (2-3 天)

13. **编写单元测试**
    - 为每个模块编写测试
    - 特别关注权重分类器和配置生成的测试
    - 确保测试覆盖率 > 80%

14. **编写集成测试**
    - 端到端测试
    - 配置文件验证测试
    - 配置生成测试

15. **创建示例配置文件**
    - DeepSeek-R1
    - Llama 3 70B
    - Qwen2 72B

16. **编写文档**
    - README.md: 用户指南
    - SKILL.md: Skill 接口定义
    - 使用示例和最佳实践

### 阶段 7: 优化和发布 (1-2 天)

17. **性能优化**
    - 优化公式计算性能
    - 优化二分搜索算法

18. **代码审查和重构**
    - 代码风格统一
    - 类型注解完善

19. **发布准备**
    - 版本号管理
    - 发布说明

**总计**: 约 12-18 天

## 9. 关键设计决策

### 9.1 为什么不使用模块类层次结构？

**原因**:
1. **配置已包含所有信息**: YAML 配置文件已经包含了权重的 shape, dtype, layers, parallel_strategy 等所有必要信息
2. **避免重复**: 创建 AttentionBase, FFNBase 等类会导致配置信息和类定义重复
3. **简化实现**: 直接遍历配置文件计算显存更简单、更直接
4. **易于扩展**: 添加新的模块类型只需修改配置文件，无需修改代码

### 9.2 为什么使用 layers 字段而不是 count 字段？

**原因**:
1. **语义清晰**: `layers` 明确表示"该权重在多少层中出现"，而 `count` 容易混淆（共享次数 vs 层数）
2. **计算一致**: 所有权重的显存计算都是 `shape × dtype × layers`，逻辑统一
3. **避免混淆**:
   - 共享权重（如 embed_tokens）：layers=1，因为在显存中只有一份
   - 多层权重（如 input_layernorm）：layers=62，因为每层都有一份
4. **简化设计**: 不需要区分"共享次数"和"层数"两种不同的语义

### 9.3 为什么使用公式字符串？

**原因**:
1. **灵活性**: 不同模型的 KV Cache 和激活值计算公式可能不同
2. **可配置**: 用户可以在配置文件中自定义公式，无需修改代码
3. **可读性**: 公式字符串比硬编码的计算逻辑更易读
4. **可验证**: 公式可以在配置文件中直接查看和验证

### 9.4 为什么使用二分搜索估算最大序列长度？

**原因**:
1. **效率**: 二分搜索的时间复杂度为 O(log n)，比线性搜索快得多
2. **精度**: 可以快速找到最接近显存上限的序列长度
3. **通用性**: 适用于任何模型和硬件配置

### 9.5 为什么使用规则配置文件进行权重分类？

**原因**:
1. **可扩展**: 添加新模型只需在 YAML 中添加规则，无需修改代码
2. **可维护**: 规则集中管理，易于修改和调试
3. **灵活**: 支持通用规则和模型特定规则，支持继承和覆盖
4. **透明**: 规则可视化，用户可以理解和自定义分类逻辑
5. **容错**: 未匹配的权重自动归类到 others，不会导致程序失败

## 10. 未来扩展

### 10.1 支持更多模型架构

- Transformer-XL
- Longformer
- BigBird
- Reformer

### 10.2 支持更多并行策略

- ZeRO (Zero Redundancy Optimizer)
- FSDP (Fully Sharded Data Parallel)
- Megatron-LM 风格的 3D 并行

### 10.3 支持更多量化方式

- GPTQ
- AWQ
- SmoothQuant
- LLM.int8()

### 10.4 可视化支持

- 生成显存占用图表
- 交互式 Web 界面
- 实时显存监控

### 10.5 自动调优

- 自动搜索最优并行策略
- 自动调整 batch size 和序列长度
- 成本优化建议
