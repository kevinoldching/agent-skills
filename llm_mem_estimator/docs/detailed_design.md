# LLM 显存占用估算器 - 详细设计

## 1. 系统架构

### 1.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI / Skill 接口                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      MemoryEstimator                         │
│  (主控制器: 协调配置加载、模块计算、结果汇总)                 │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ ConfigLoader │    │ ModuleFactory│    │ ResultFormatter│
│ (配置解析)   │    │ (模块工厂)   │    │ (结果格式化)  │
└──────────────┘    └──────────────┘    └──────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ AttentionBase│    │   FFNBase    │    │   NormBase   │
│ (注意力模块) │    │ (FFN模块)    │    │ (归一化模块) │
└──────────────┘    └──────────────┘    └──────────────┘
        │                     │                     │
        ├─ MHA               ├─ StandardFFN        ├─ LayerNorm
        ├─ MQA               ├─ SwiGLU             ├─ RMSNorm
        ├─ GQA               └─ MoE                └─ DeepNorm
        ├─ SWA
        ├─ MLA
        ├─ DSA
        └─ GDN
```

### 1.2 数据流

```
YAML配置文件 → ConfigLoader → ModelConfig对象
                                    │
                                    ▼
              ModuleFactory 创建各类模块实例
                                    │
                                    ▼
              MemoryEstimator 调用各模块计算显存
                                    │
                                    ▼
              ResultFormatter 格式化输出结果
```

## 2. 核心模块设计

### 2.1 ConfigLoader (配置加载器)

**职责**:
- 加载并解析 YAML 配置文件
- 验证配置完整性
- 加载硬件配置 (chips.json)

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
    def validate_config(config: ModelConfig) -> bool:
        """验证配置完整性"""
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

    # 架构配置
    architecture_config: ArchitectureConfig

    # 模块定义
    modules: Dict[str, Any]

    # 计算规则
    computation_rules: ComputationRules

@dataclass
class ModelIdentity:
    name: str
    total_params_b: float
    num_layers: int
    quantization: str

@dataclass
class ArchitectureConfig:
    common: Dict[str, Any]
    attention: Dict[str, Any]
    moe: Optional[Dict[str, Any]] = None

@dataclass
class ComputationRules:
    activation_vram: Dict[str, Any]
    kv_cache_vram: Dict[str, Any]
```

### 2.3 ModuleFactory (模块工厂)

**职责**:
- 根据配置创建对应的模块实例
- 管理模块注册表

**接口**:
```python
class ModuleFactory:
    _attention_registry: Dict[str, Type[AttentionBase]] = {}
    _ffn_registry: Dict[str, Type[FFNBase]] = {}
    _norm_registry: Dict[str, Type[NormBase]] = {}

    @classmethod
    def register_attention(cls, name: str, attention_class: Type[AttentionBase]):
        """注册 Attention 模块"""
        cls._attention_registry[name] = attention_class

    @classmethod
    def create_attention(cls, attention_type: str, config: Dict) -> AttentionBase:
        """创建 Attention 模块实例"""
        if attention_type not in cls._attention_registry:
            raise ValueError(f"Unknown attention type: {attention_type}")
        return cls._attention_registry[attention_type](config)

    @classmethod
    def create_ffn(cls, ffn_type: str, config: Dict) -> FFNBase:
        """创建 FFN 模块实例"""
        pass

    @classmethod
    def create_norm(cls, norm_type: str, config: Dict) -> NormBase:
        """创建 Norm 模块实例"""
        pass
```

### 2.4 AttentionBase (注意力模块基类)

**职责**: 定义 Attention 模块的通用接口

**接口**:
```python
class AttentionBase(ABC):
    def __init__(self, config: Dict):
        self.config = config

    @abstractmethod
    def calculate_kv_cache(self, batch_size: int, seq_len: int,
                          num_layers: int, dtype_bytes: float,
                          parallel_config: ParallelConfig) -> float:
        """计算 KV Cache 显存占用 (GB)"""
        pass

    @abstractmethod
    def calculate_weight_memory(self, parallel_config: ParallelConfig) -> float:
        """计算权重显存占用 (GB)"""
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """验证配置参数完整性"""
        pass

### 2.5 具体 Attention 实现类

#### 2.5.1 MHA (Multi-Head Attention)

```python
class MHAAttention(AttentionBase):
    def calculate_kv_cache(self, batch_size: int, seq_len: int,
                          num_layers: int, dtype_bytes: float,
                          parallel_config: ParallelConfig) -> float:
        """
        KV Cache = 2 * batch_size * seq_len * num_kv_heads * head_dim * num_layers * dtype_bytes
        考虑并行: / TP / CP
        """
        num_kv_heads = self.config['num_kv_heads']
        head_dim = self.config['head_dim']

        kv_cache_bytes = 2 * batch_size * seq_len * num_kv_heads * head_dim * num_layers * dtype_bytes
        kv_cache_gb = kv_cache_bytes / (1024**3)

        # 应用并行策略
        kv_cache_gb = kv_cache_gb / parallel_config.tp / parallel_config.cp

        return kv_cache_gb

    def validate_config(self) -> bool:
        required_fields = ['num_attention_heads', 'num_kv_heads', 'head_dim']
        return all(field in self.config for field in required_fields)
```

#### 2.5.2 MLA (Multi-Latent Attention)

```python
class MLAAttention(AttentionBase):
    def calculate_kv_cache(self, batch_size: int, seq_len: int,
                          num_layers: int, dtype_bytes: float,
                          parallel_config: ParallelConfig) -> float:
        """
        MLA 使用 LoRA 压缩:
        KV Cache = (kv_lora_rank + qk_rope_head_dim) * batch_size * seq_len * num_layers * 2 * dtype_bytes
        """
        kv_lora_rank = self.config['kv_lora_rank']
        qk_rope_head_dim = self.config['qk_rope_head_dim']

        kv_dim = kv_lora_rank + qk_rope_head_dim
        kv_cache_bytes = kv_dim * batch_size * seq_len * num_layers * 2 * dtype_bytes
        kv_cache_gb = kv_cache_bytes / (1024**3)

        # 应用并行策略
        kv_cache_gb = kv_cache_gb / parallel_config.tp / parallel_config.cp

        return kv_cache_gb

    def validate_config(self) -> bool:
        required_fields = ['kv_lora_rank', 'qk_rope_head_dim', 'num_attention_heads']
        return all(field in self.config for field in required_fields)
```

#### 2.5.3 其他 Attention 类型

类似地实现:
- `MQAAttention`: num_kv_heads = 1
- `GQAAttention`: num_kv_heads < num_attention_heads
- `SWAAttention`: 考虑 sliding_window_size
- `DSAAttention`: DeepSeek Attention with index
- `GDNAttention`: Generalized Delta Networks

### 2.6 FFNBase (FFN 模块基类)

```python
class FFNBase(ABC):
    def __init__(self, config: Dict):
        self.config = config

    @abstractmethod
    def calculate_weight_memory(self, parallel_config: ParallelConfig) -> float:
        """计算 FFN 权重显存占用 (GB)"""
        pass

    @abstractmethod
    def calculate_activation_memory(self, batch_size: int, seq_len: int,
                                   dtype_bytes: float,
                                   parallel_config: ParallelConfig) -> float:
        """计算 FFN 激活值显存占用 (GB)"""
        pass

### 2.7 具体 FFN 实现类

#### 2.7.1 StandardFFN

```python
class StandardFFN(FFNBase):
    def calculate_activation_memory(self, batch_size: int, seq_len: int,
                                   dtype_bytes: float,
                                   parallel_config: ParallelConfig) -> float:
        """
        激活值 = batch_size * seq_len * intermediate_size * dtype_bytes
        """
        intermediate_size = self.config['intermediate_size']
        activation_bytes = batch_size * seq_len * intermediate_size * dtype_bytes
        activation_gb = activation_bytes / (1024**3)

        # 应用并行策略
        activation_gb = activation_gb / parallel_config.tp

        return activation_gb
```

#### 2.7.2 MoEFFN

```python
class MoEFFN(FFNBase):
    def calculate_activation_memory(self, batch_size: int, seq_len: int,
                                   dtype_bytes: float,
                                   parallel_config: ParallelConfig) -> float:
        """
        MoE 激活值计算考虑 capacity_factor:
        激活值 = seq_len * topk * hidden_size * capacity_factor * dtype_bytes
        """
        hidden_size = self.config['hidden_size']
        topk = self.config['topk']
        capacity_factor = self.config.get('capacity_factor', 1.25)

        activation_bytes = seq_len * topk * hidden_size * capacity_factor * dtype_bytes
        activation_gb = activation_bytes / (1024**3)

        # 应用并行策略
        activation_gb = activation_gb / parallel_config.ep

        return activation_gb

    def calculate_weight_memory(self, parallel_config: ParallelConfig) -> float:
        """
        MoE 权重 = (num_router_experts * expert_size + num_shared_experts * expert_size) * dtype_bytes
        """
        num_router_experts = self.config['num_router_experts']
        num_shared_experts = self.config['num_shared_experts']
        moe_intermediate_size = self.config['moe_intermediate_size']
        hidden_size = self.config['hidden_size']
        dtype_bytes = self.config['dtype_bytes']

        # Router experts: gate_up_proj + down_proj
        router_expert_params = num_router_experts * (2 * moe_intermediate_size * hidden_size + hidden_size * moe_intermediate_size)

        # Shared experts
        shared_expert_params = num_shared_experts * (2 * moe_intermediate_size * hidden_size + hidden_size * moe_intermediate_size)

        total_params = router_expert_params + shared_expert_params
        weight_bytes = total_params * dtype_bytes
        weight_gb = weight_bytes / (1024**3)

        # 应用并行策略: router experts 使用 EP, shared experts replicated
        router_weight_gb = (router_expert_params * dtype_bytes / (1024**3)) / parallel_config.ep
        shared_weight_gb = shared_expert_params * dtype_bytes / (1024**3)

        return router_weight_gb + shared_weight_gb

### 2.8 NormBase (归一化模块基类)

```python
class NormBase(ABC):
    def __init__(self, config: Dict):
        self.config = config

    @abstractmethod
    def calculate_weight_memory(self, parallel_config: ParallelConfig) -> float:
        """计算 Norm 权重显存占用 (GB)"""
        pass

class LayerNorm(NormBase):
    def calculate_weight_memory(self, parallel_config: ParallelConfig) -> float:
        """LayerNorm: 2 * hidden_size (weight + bias)"""
        hidden_size = self.config['hidden_size']
        dtype_bytes = self.config['dtype_bytes']
        params = 2 * hidden_size
        return params * dtype_bytes / (1024**3)

class RMSNorm(NormBase):
    def calculate_weight_memory(self, parallel_config: ParallelConfig) -> float:
        """RMSNorm: hidden_size (只有 weight)"""
        hidden_size = self.config['hidden_size']
        dtype_bytes = self.config['dtype_bytes']
        params = hidden_size
        return params * dtype_bytes / (1024**3)
```

### 2.9 ParallelConfig (并行配置)

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

    def validate(self) -> bool:
        """验证并行配置的合理性"""
        return all(x > 0 for x in [self.tp, self.pp, self.dp, self.ep, self.sp, self.cp])
```

### 2.10 MemoryEstimator (主控制器)

```python
class MemoryEstimator:
    def __init__(self, model_config: ModelConfig, parallel_config: ParallelConfig):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.module_factory = ModuleFactory()

    def estimate(self, batch_size: int, seq_len: int,
                chip_config: ChipConfig) -> MemoryResult:
        """
        主估算函数
        """
        # 1. 计算权重显存
        weight_memory = self._calculate_weight_memory()

        # 2. 计算 KV Cache 显存
        kv_cache_memory = self._calculate_kv_cache(batch_size, seq_len)

        # 3. 计算激活值显存
        activation_memory = self._calculate_activation_memory(batch_size, seq_len)

        # 4. 计算系统保留显存
        system_reserved = self._calculate_system_reserved(chip_config.vram_gb)

        # 5. 计算其他显存 (通信 buffer, 框架开销等)
        other_memory = self._calculate_other_memory(chip_config.vram_gb)

        # 6. 汇总
        total_memory = weight_memory + kv_cache_memory + activation_memory + system_reserved + other_memory

        return MemoryResult(
            total_memory_gb=total_memory,
            weight_memory_gb=weight_memory,
            kv_cache_memory_gb=kv_cache_memory,
            activation_memory_gb=activation_memory,
            system_reserved_gb=system_reserved,
            other_memory_gb=other_memory,
            available_memory_gb=chip_config.vram_gb - total_memory
        )

    def _calculate_weight_memory(self) -> float:
        """
        计算权重显存占用
        遍历 modules 配置,根据 parallel_strategy 计算每个模块的权重显存
        """
        total_weight_gb = 0.0

        for module_name, module_config in self.model_config.modules.items():
            if isinstance(module_config, dict):
                for param_name, param_config in module_config.items():
                    if 'shape' in param_config and 'dtype' in param_config:
                        # 计算参数量
                        shape = param_config['shape']
                        params = 1
                        for dim in shape:
                            params *= dim

                        # 获取数据类型字节数
                        dtype_bytes = self._get_dtype_bytes(param_config['dtype'])

                        # 计算显存 (GB)
                        memory_gb = params * dtype_bytes / (1024**3)

                        # 应用并行策略
                        parallel_strategy = param_config.get('parallel_strategy', 'replicated')
                        memory_gb = self._apply_parallel_strategy(memory_gb, parallel_strategy)

                        total_weight_gb += memory_gb

        return total_weight_gb

    def _calculate_kv_cache(self, batch_size: int, seq_len: int) -> float:
        """
        计算 KV Cache 显存占用
        根据 computation_rules.kv_cache_vram.formula 计算
        """
        # 从配置中获取 attention 类型
        attention_config = self.model_config.architecture_config.attention

        # 根据 attention 类型创建对应的 attention 模块
        # 这里简化处理,实际应该从配置中读取 attention_type
        # 假设使用 MLA
        if 'kv_lora_rank' in attention_config:
            # MLA attention
            kv_lora_rank = attention_config['kv_lora_rank']
            qk_rope_head_dim = attention_config['qk_rope_head_dim']
            num_layers = self.model_config.model_identity.num_layers

            # 获取 dtype_bytes (如果使用 C8 量化则为 1, 否则为 2)
            dtype_bytes = 1 if 'C8' in self.model_config.model_identity.quantization else 2

            kv_dim = kv_lora_rank + qk_rope_head_dim
            kv_cache_bytes = kv_dim * batch_size * seq_len * num_layers * 2 * dtype_bytes
            kv_cache_gb = kv_cache_bytes / (1024**3)

            # 应用并行策略
            kv_cache_gb = kv_cache_gb / self.parallel_config.tp / self.parallel_config.cp

            return kv_cache_gb
        else:
            # 标准 attention (MHA/GQA)
            num_kv_heads = attention_config.get('num_key_value_heads', attention_config['num_heads'])
            head_dim = attention_config.get('head_dim',
                                           self.model_config.architecture_config.common['hidden_size'] // attention_config['num_heads'])
            num_layers = self.model_config.model_identity.num_layers
            dtype_bytes = 2  # BF16/FP16

            kv_cache_bytes = 2 * batch_size * seq_len * num_kv_heads * head_dim * num_layers * dtype_bytes
            kv_cache_gb = kv_cache_bytes / (1024**3)

            # 应用并行策略
            kv_cache_gb = kv_cache_gb / self.parallel_config.tp / self.parallel_config.cp

            return kv_cache_gb

    def _calculate_activation_memory(self, batch_size: int, seq_len: int) -> float:
        """
        计算激活值显存占用
        根据 computation_rules.activation_vram.formula 计算
        """
        # 从配置中获取 capacity_factor
        activation_config = self.model_config.computation_rules.activation_vram
        capacity_factor = activation_config.get('recommended_capacity_factor', 1.25)

        # 检查是否是 MoE 模型
        if self.model_config.architecture_config.moe:
            # MoE 模型
            moe_config = self.model_config.architecture_config.moe
            topk = moe_config['num_experts_per_tok']
            hidden_size = self.model_config.architecture_config.common['hidden_size']
            dtype_bytes = 2  # BF16

            activation_bytes = seq_len * topk * hidden_size * capacity_factor * dtype_bytes
            activation_gb = activation_bytes / (1024**3)

            # 应用并行策略
            activation_gb = activation_gb / self.parallel_config.ep

            return activation_gb
        else:
            # 非 MoE 模型,使用简化公式
            hidden_size = self.model_config.architecture_config.common['hidden_size']
            intermediate_size = self.model_config.architecture_config.common['intermediate_size']
            dtype_bytes = 2

            # 峰值激活 = batch_size * seq_len * intermediate_size * dtype_bytes
            activation_bytes = batch_size * seq_len * intermediate_size * dtype_bytes
            activation_gb = activation_bytes / (1024**3)

            # 应用并行策略
            activation_gb = activation_gb / self.parallel_config.tp

            return activation_gb

    def _calculate_system_reserved(self, vram_gb: float) -> float:
        """
        计算系统保留显存
        默认按显存容量的 5% 估算
        """
        reserve_ratio = 0.05
        return vram_gb * reserve_ratio

    def _calculate_other_memory(self, vram_gb: float) -> float:
        """
        计算其他显存 (通信 buffer, 框架开销等)
        默认按显存容量的 5% 估算
        """
        other_ratio = 0.05
        return vram_gb * other_ratio

    def _get_dtype_bytes(self, dtype: str) -> float:
        """获取数据类型的字节数"""
        dtype_map = {
            'BF16': 2,
            'FP16': 2,
            'F8_E4M3': 1,
            'INT8': 1,
            'INT4': 0.5,
            'F32': 4,
            'FP32': 4
        }
        return dtype_map.get(dtype, 2)  # 默认 2 字节

    def _apply_parallel_strategy(self, memory_gb: float, strategy: str) -> float:
        """
        根据并行策略调整显存占用
        """
        if strategy == 'replicated':
            return memory_gb
        elif strategy == 'tp_col' or strategy == 'tp_row':
            return memory_gb / self.parallel_config.tp
        elif strategy == 'expert_sharded':
            return memory_gb / self.parallel_config.ep
        elif strategy == 'pp':
            return memory_gb / self.parallel_config.pp
        else:
            return memory_gb

    def estimate_max_sequence_length(self, batch_size: int,
                                     chip_config: ChipConfig) -> int:
        """
        估算给定 batch_size 下的最大序列长度
        使用二分搜索找到最大可支持的序列长度
        """
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

    def estimate_max_batch_size(self, seq_len: int,
                                chip_config: ChipConfig) -> int:
        """
        估算给定 seq_len 下的最大 batch size
        使用二分搜索找到最大可支持的 batch size
        """
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
```

### 2.11 MemoryResult (结果数据类)

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

    def __str__(self) -> str:
        """格式化输出"""
        return f"""
Memory Estimation Result:
========================
Total Memory:        {self.total_memory_gb:.2f} GB
  - Weights:         {self.weight_memory_gb:.2f} GB
  - KV Cache:        {self.kv_cache_memory_gb:.2f} GB
  - Activations:     {self.activation_memory_gb:.2f} GB
  - System Reserved: {self.system_reserved_gb:.2f} GB
  - Other:           {self.other_memory_gb:.2f} GB
Available Memory:    {self.available_memory_gb:.2f} GB
"""

### 2.12 ChipConfig (芯片配置数据类)

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

## 3. 错误处理策略

### 3.1 配置验证错误

```python
class ConfigValidationError(Exception):
    """配置验证错误"""
    pass

class ConfigLoader:
    @staticmethod
    def validate_config(config: ModelConfig) -> None:
        """验证配置完整性,如果验证失败则抛出异常"""
        # 验证 model_identity
        if not config.model_identity.name:
            raise ConfigValidationError("model_identity.name is required")

        # 验证 architecture_config
        if not config.architecture_config.common:
            raise ConfigValidationError("architecture_config.common is required")

        required_common_fields = ['num_layers', 'hidden_size']
        for field in required_common_fields:
            if field not in config.architecture_config.common:
                raise ConfigValidationError(f"architecture_config.common.{field} is required")

        # 验证 attention 配置
        if not config.architecture_config.attention:
            raise ConfigValidationError("architecture_config.attention is required")

        # 验证 modules
        if not config.modules:
            raise ConfigValidationError("modules configuration is required")
```

### 3.2 并行配置验证错误

```python
class ParallelConfigError(Exception):
    """并行配置错误"""
    pass

class ParallelConfig:
    def validate(self) -> None:
        """验证并行配置的合理性"""
        if any(x <= 0 for x in [self.tp, self.pp, self.dp, self.ep, self.sp, self.cp]):
            raise ParallelConfigError("All parallel dimensions must be positive")

        # 验证 GPU 总数是否合理
        total_gpus = self.tp * self.pp * self.dp
        if total_gpus > 10000:
            raise ParallelConfigError(f"Total GPUs ({total_gpus}) seems unreasonable")
```

### 3.3 显存不足错误

```python
class InsufficientMemoryError(Exception):
    """显存不足错误"""
    pass

class MemoryEstimator:
    def estimate(self, batch_size: int, seq_len: int,
                chip_config: ChipConfig) -> MemoryResult:
        """主估算函数"""
        # ... 计算逻辑 ...

        if total_memory > chip_config.vram_gb:
            raise InsufficientMemoryError(
                f"Required memory ({total_memory:.2f} GB) exceeds available memory ({chip_config.vram_gb} GB)"
            )

        return result

## 4. 测试策略

### 4.1 单元测试

#### 4.1.1 ConfigLoader 测试

```python
def test_load_model_config():
    """测试加载模型配置"""
    config = ConfigLoader.load_model_config("configs/models/deepseek-r1.yaml")
    assert config.model_identity.name == "DeepSeek-R1"
    assert config.model_identity.num_layers == 62

def test_validate_config_missing_fields():
    """测试配置验证 - 缺少必需字段"""
    with pytest.raises(ConfigValidationError):
        invalid_config = ModelConfig(...)
        ConfigLoader.validate_config(invalid_config)
```

#### 4.1.2 Attention 模块测试

```python
def test_mha_kv_cache_calculation():
    """测试 MHA 的 KV Cache 计算"""
    config = {
        'num_attention_heads': 64,
        'num_kv_heads': 64,
        'head_dim': 128
    }
    mha = MHAAttention(config)
    parallel_config = ParallelConfig(tp=1, cp=1)

    kv_cache_gb = mha.calculate_kv_cache(
        batch_size=1,
        seq_len=4096,
        num_layers=80,
        dtype_bytes=2,
        parallel_config=parallel_config
    )

    # 验证计算结果
    expected = 2 * 1 * 4096 * 64 * 128 * 80 * 2 / (1024**3)
    assert abs(kv_cache_gb - expected) < 0.01

def test_mla_kv_cache_calculation():
    """测试 MLA 的 KV Cache 计算"""
    config = {
        'kv_lora_rank': 512,
        'qk_rope_head_dim': 64,
        'num_attention_heads': 128
    }
    mla = MLAAttention(config)
    parallel_config = ParallelConfig(tp=1, cp=1)

    kv_cache_gb = mla.calculate_kv_cache(
        batch_size=1,
        seq_len=4096,
        num_layers=62,
        dtype_bytes=1,  # FP8
        parallel_config=parallel_config
    )

    # 验证计算结果
    kv_dim = 512 + 64
    expected = kv_dim * 1 * 4096 * 62 * 2 * 1 / (1024**3)
    assert abs(kv_cache_gb - expected) < 0.01
```

#### 4.1.3 MemoryEstimator 测试

```python
def test_memory_estimation_deepseek_r1():
    """测试 DeepSeek-R1 的显存估算"""
    model_config = ConfigLoader.load_model_config("configs/models/deepseek-r1.yaml")
    parallel_config = ParallelConfig(tp=8, ep=8)
    chip_config = ChipConfig.from_json("h100-80gb")

    estimator = MemoryEstimator(model_config, parallel_config)
    result = estimator.estimate(batch_size=1, seq_len=4096, chip_config=chip_config)

    # 验证结果
    assert result.total_memory_gb > 0
    assert result.total_memory_gb <= chip_config.vram_gb
    assert result.weight_memory_gb > 0
    assert result.kv_cache_memory_gb > 0
```

### 4.2 集成测试

```python
def test_end_to_end_estimation():
    """端到端测试: 从配置文件到结果输出"""
    # 加载配置
    model_config = ConfigLoader.load_model_config("configs/models/deepseek-r1.yaml")
    parallel_config = ParallelConfig(tp=8, ep=8)
    chip_config = ChipConfig.from_json("h100-80gb")

    # 创建估算器
    estimator = MemoryEstimator(model_config, parallel_config)

    # 执行估算
    result = estimator.estimate(batch_size=1, seq_len=4096, chip_config=chip_config)

    # 验证结果格式
    result_dict = result.to_dict()
    assert 'total_memory_gb' in result_dict
    assert 'weight_memory_gb' in result_dict

    # 验证结果输出
    result_str = str(result)
    assert 'Memory Estimation Result' in result_str
```

### 4.3 性能测试

```python
def test_estimation_performance():
    """测试估算性能"""
    import time

    model_config = ConfigLoader.load_model_config("configs/models/deepseek-r1.yaml")
    parallel_config = ParallelConfig(tp=8, ep=8)
    chip_config = ChipConfig.from_json("h100-80gb")

    estimator = MemoryEstimator(model_config, parallel_config)

    start_time = time.time()
    result = estimator.estimate(batch_size=1, seq_len=4096, chip_config=chip_config)
    end_time = time.time()

    # 估算应该在 1 秒内完成
    assert end_time - start_time < 1.0

## 5. 使用示例

### 5.1 基本使用

```python
from llm_mem_estimator import ConfigLoader, MemoryEstimator, ParallelConfig, ChipConfig

# 1. 加载模型配置
model_config = ConfigLoader.load_model_config("configs/models/deepseek-r1.yaml")

# 2. 设置并行配置
parallel_config = ParallelConfig(tp=8, ep=8, pp=1, dp=1)

# 3. 加载芯片配置
chip_config = ChipConfig.from_json("h100-80gb")

# 4. 创建估算器
estimator = MemoryEstimator(model_config, parallel_config)

# 5. 执行估算
result = estimator.estimate(batch_size=1, seq_len=4096, chip_config=chip_config)

# 6. 输出结果
print(result)
print(f"Total Memory: {result.total_memory_gb:.2f} GB")
```

### 5.2 估算最大序列长度

```python
# 估算给定 batch_size 下的最大序列长度
max_seq_len = estimator.estimate_max_sequence_length(batch_size=1, chip_config=chip_config)
print(f"Max sequence length for batch_size=1: {max_seq_len}")
```

### 5.3 估算最大 Batch Size

```python
# 估算给定 seq_len 下的最大 batch size
max_batch_size = estimator.estimate_max_batch_size(seq_len=4096, chip_config=chip_config)
print(f"Max batch size for seq_len=4096: {max_batch_size}")
```

### 5.4 CLI 使用

```bash
# 基本估算
python scripts/calculate_mem.py \
    --model configs/models/deepseek-r1.yaml \
    --chip h100-80gb \
    --batch-size 1 \
    --seq-len 4096 \
    --tp 8 \
    --ep 8

# 估算最大序列长度
python scripts/calculate_mem.py \
    --model configs/models/deepseek-r1.yaml \
    --chip h100-80gb \
    --batch-size 1 \
    --tp 8 \
    --ep 8 \
    --estimate-max-seq-len

# 估算最大 batch size
python scripts/calculate_mem.py \
    --model configs/models/deepseek-r1.yaml \
    --chip h100-80gb \
    --seq-len 4096 \
    --tp 8 \
    --ep 8 \
    --estimate-max-batch-size
```

## 6. CLI 接口设计

### 6.1 命令行参数

```python
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='LLM Memory Estimator')

    # 模型配置
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model YAML config file')

    # 硬件配置
    parser.add_argument('--chip', type=str, required=True,
                       help='Chip name (e.g., h100-80gb, a100-80gb)')

    # 序列配置
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--seq-len', type=int, default=4096,
                       help='Sequence length')

    # 并行策略
    parser.add_argument('--tp', type=int, default=1,
                       help='Tensor Parallel size')
    parser.add_argument('--pp', type=int, default=1,
                       help='Pipeline Parallel size')
    parser.add_argument('--dp', type=int, default=1,
                       help='Data Parallel size')
    parser.add_argument('--ep', type=int, default=1,
                       help='Expert Parallel size')
    parser.add_argument('--sp', type=int, default=1,
                       help='Sequence Parallel size')
    parser.add_argument('--cp', type=int, default=1,
                       help='Context Parallel size')

    # 估算模式
    parser.add_argument('--estimate-max-seq-len', action='store_true',
                       help='Estimate maximum sequence length')
    parser.add_argument('--estimate-max-batch-size', action='store_true',
                       help='Estimate maximum batch size')

    # 输出格式
    parser.add_argument('--output-format', type=str, default='text',
                       choices=['text', 'json', 'yaml'],
                       help='Output format')

    return parser.parse_args()
```

### 6.2 主函数

```python
def main():
    args = parse_args()

    try:
        # 加载配置
        model_config = ConfigLoader.load_model_config(args.model)
        parallel_config = ParallelConfig(
            tp=args.tp, pp=args.pp, dp=args.dp,
            ep=args.ep, sp=args.sp, cp=args.cp
        )
        chip_config = ChipConfig.from_json(args.chip)

        # 创建估算器
        estimator = MemoryEstimator(model_config, parallel_config)

        # 执行估算
        if args.estimate_max_seq_len:
            max_seq_len = estimator.estimate_max_sequence_length(
                batch_size=args.batch_size,
                chip_config=chip_config
            )
            print(f"Maximum sequence length: {max_seq_len}")
        elif args.estimate_max_batch_size:
            max_batch_size = estimator.estimate_max_batch_size(
                seq_len=args.seq_len,
                chip_config=chip_config
            )
            print(f"Maximum batch size: {max_batch_size}")
        else:
            result = estimator.estimate(
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                chip_config=chip_config
            )

            # 输出结果
            if args.output_format == 'json':
                import json
                print(json.dumps(result.to_dict(), indent=2))
            elif args.output_format == 'yaml':
                import yaml
                print(yaml.dump(result.to_dict()))
            else:
                print(result)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()

## 7. 文件组织结构

```
llm_mem_estimator/
├── SKILL.md                      # Skill 接口定义
├── CLAUDE.md                     # Claude Code 项目指南
├── README.md                     # 项目说明文档
├── docs/
│   ├── plan.md                   # 实现计划
│   └── detailed_design.md        # 详细设计 (本文档)
├── configs/
│   ├── models/                   # 模型配置文件
│   │   ├── deepseek-r1.yaml
│   │   ├── llama-70b.yaml
│   │   ├── qwen-72b.yaml
│   │   └── ...
│   └── chips.json                # 芯片配置文件
├── scripts/
│   └── calculate_mem.py          # 核心计算脚本 (CLI 入口)
├── llm_mem_estimator/            # Python 包
│   ├── __init__.py
│   ├── config_loader.py          # 配置加载器
│   ├── models.py                 # 数据模型 (ModelConfig, ChipConfig, etc.)
│   ├── modules/                  # 模块实现
│   │   ├── __init__.py
│   │   ├── attention.py          # Attention 模块
│   │   ├── ffn.py                # FFN 模块
│   │   └── norm.py               # Norm 模块
│   ├── estimator.py              # MemoryEstimator 主控制器
│   ├── factory.py                # ModuleFactory 模块工厂
│   └── exceptions.py             # 自定义异常
├── tests/                        # 测试文件
│   ├── __init__.py
│   ├── test_config_loader.py
│   ├── test_attention.py
│   ├── test_ffn.py
│   ├── test_estimator.py
│   └── test_integration.py
└── requirements.txt              # Python 依赖
```

## 8. 实现步骤

### 阶段 1: 基础框架搭建 (1-2 天)

1. **创建项目结构**
   - 创建目录结构
   - 初始化 Python 包
   - 创建 requirements.txt

2. **实现数据模型**
   - 实现 ModelConfig, ChipConfig, ParallelConfig, MemoryResult
   - 实现自定义异常类

3. **实现 ConfigLoader**
   - 实现 YAML 配置文件加载
   - 实现配置验证逻辑
   - 实现 chips.json 加载

### 阶段 2: 模块实现 (2-3 天)

1. **实现 Attention 模块**
   - 实现 AttentionBase 基类
   - 实现 MHA, MQA, GQA 基础 Attention
   - 实现 MLA, DSA 高级 Attention
   - 实现 SWA, GDN 特殊 Attention

2. **实现 FFN 模块**
   - 实现 FFNBase 基类
   - 实现 StandardFFN, SwiGLU
   - 实现 MoEFFN

3. **实现 Norm 模块**
   - 实现 NormBase 基类
   - 实现 LayerNorm, RMSNorm, DeepNorm

4. **实现 ModuleFactory**
   - 实现模块注册机制
   - 实现模块创建逻辑

### 阶段 3: 核心估算器实现 (2-3 天)

1. **实现 MemoryEstimator**
   - 实现权重显存计算
   - 实现 KV Cache 显存计算
   - 实现激活值显存计算
   - 实现系统保留和其他显存计算
   - 实现最大序列长度估算
   - 实现最大 batch size 估算

2. **实现并行策略处理**
   - 实现 _apply_parallel_strategy 方法
   - 验证并行策略的正确性

### 阶段 4: CLI 和配置文件 (1-2 天)

1. **实现 CLI 接口**
   - 实现命令行参数解析
   - 实现主函数
   - 实现输出格式化 (text, json, yaml)

2. **创建配置文件**
   - 创建 chips.json
   - 创建 DeepSeek-R1 配置文件
   - 创建 Llama-70B 配置文件
   - 创建其他常见模型配置文件

### 阶段 5: 测试和文档 (1-2 天)

1. **编写测试**
   - 编写单元测试
   - 编写集成测试
   - 编写性能测试

2. **完善文档**
   - 完善 README.md
   - 完善 SKILL.md
   - 添加使用示例

### 阶段 6: 集成和优化 (1 天)

1. **集成到 Agent Skill**
   - 实现 Skill 接口
   - 测试 Skill 调用

2. **性能优化**
   - 优化计算性能
   - 优化内存占用

## 9. 注意事项

### 9.1 设计原则

1. **不自动推导参数**: 程序不自动推导任何参数,用户必须提供完整的 YAML 配置文件
2. **模块化设计**: Attention、FFN、Norm 等模块独立实现,便于扩展
3. **配置驱动**: 每个模型存储于独立的 YAML 配置文件
4. **向前向后兼容**: 配置文件支持添加任意自定义字段,程序解析时忽略未知字段

### 9.2 性能考虑

1. **计算效率**: 估算应该在 1 秒内完成
2. **内存占用**: 程序本身的内存占用应该很小 (< 100MB)
3. **并发支持**: 支持多个估算任务并发执行

### 9.3 可扩展性

1. **新增 Attention 类型**: 只需继承 AttentionBase 并注册
2. **新增 FFN 类型**: 只需继承 FFNBase 并注册
3. **新增 Norm 类型**: 只需继承 NormBase 并注册
4. **新增模型**: 只需添加 YAML 配置文件

### 9.4 错误处理

1. **配置验证**: 在加载配置时进行完整性验证
2. **并行配置验证**: 验证并行配置的合理性
3. **显存不足提示**: 当估算结果超过硬件容量时给出明确提示

## 10. 总结

本详细设计文档定义了 LLM 显存占用估算器的完整架构和实现方案:

1. **系统架构**: 采用模块化设计,包含 ConfigLoader、ModuleFactory、MemoryEstimator 等核心组件
2. **模块设计**: 定义了 Attention、FFN、Norm 三大模块的基类和具体实现
3. **数据模型**: 定义了 ModelConfig、ChipConfig、ParallelConfig、MemoryResult 等数据结构
4. **计算逻辑**: 详细说明了权重、KV Cache、激活值等显存占用的计算方法
5. **错误处理**: 定义了配置验证、并行配置验证、显存不足等错误处理策略
6. **测试策略**: 包含单元测试、集成测试、性能测试
7. **CLI 接口**: 提供了完整的命令行接口设计
8. **实现步骤**: 分 6 个阶段,预计 8-13 天完成

该设计方案具有良好的可扩展性、可维护性和性能,能够满足工业级大模型显存估算的需求。
```

```

```

```

```
```
```
