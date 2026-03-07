# vLLM NPU平台扩展规范

## 1) 层级放置规范

- 新增代码应遵循以下目录结构：

  组件                 位置
  -------------------- ---------------------------------------------------------
  NPU Base layer       `src/omni_npu/layers/`
  NPU 高性能layer       `src/omni_npu/v1/layers/`
  Attention 后端实现   `src/omni_npu/attention/backends/`
  模型实现             `src/omni_npu/v1/models/`
  通用 Patch           `src/omni_npu/vllm_patches/patches/common/`
  模型 Patch           `src/omni_npu/vllm_patches/patches/models/{modelname}/`

- 说明：
  - **Base Layer**：面向内部 NPU 优化，接口需与 vLLM 兼容，使用 `register_oot` 模式进行注册
  - **高性能Layer**：为极致性能设计的自定义接口，仅用于 `v1/models` 下的自定义脚本

## 2) 插件架构

### Entry Points

```toml
# pyproject.toml
[project.entry-points."vllm.platform"]
npu = "omni_npu.platform:plugin"

[project.entry-points."vllm.plugins"]
omni_npu_patches = "omni_npu.vllm_patches:apply_patches"
omni_custom_models = "omni_npu.v1.models:register_models"
```

### 平台插件模式

```python
# platform.py
class NPUPlatform(Platform):
    def get_device_name(self, device_type: str) -> str:
        if device_type != "npu":
            raise ValueError(f"不支持的设备类型: {device_type}")
        return "npu"

    def get_attn_backend_cls(
        self,
        selected_backend: str,
        *args,
        **kwargs
    ) -> Type:
        # 选择合适的Attention Backend
```

## 3) Patch Manager

### 补丁发现

补丁自动发现路径：
- `src/omni_npu/vllm_patches/patches/common/` - 通用补丁
- `src/omni_npu/vllm_patches/patches/models/{modeltype}/` - 模型特定补丁

### 执行控制

```python
# OMNI_NPU_VLLM_PATCHES 环境变量
# "ALL" - 执行所有补丁
# "PatchA,PatchB" - 仅执行指定补丁
```

### 补丁模板

```python
from omni_npu.vllm_patches.core import VLLMPatch, register_patch

@register_patch("PatchName", TargetClass)
class MyPatch(VLLMPatch):
    _attr_names_to_apply = ['method_name', 'another_method']

    def method_name(self, *args, **kwargs):
        # 补丁实现逻辑
        pass
```

检查项：
-   `_attr_names_to_apply` 中正确列出所有被 patch 的方法

## 4) CustomOp扩展模式

### 支持的算子类型

- `layernorm` - 层归一化
- `linear` - 线性层
- `fused_moe` - MoE 专家融合算子
- `activation` - 激活函数
- `rotary_embedding` - RoPE 旋转位置编码
- `vocab_embedding` - 词表嵌入

### OOT (Out-Of-Tree) 注册模式

```python
from vllm.model_executor.layers.activation import SiluAndMul

@SiluAndMul.register_oot
class NPUSiluAndMul(SiluAndMul):
    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_swiglu(x)
```
检查项：

-   方法名必须为 `forward_oot`
-   参数签名需与基类一致

## 5) 模型注册模式

```python
# v1/models/__init__.py
def register_models():
    ModelRegistry.register_model(
        "ModelName",
        "omni_npu.v1.models.module:ClassName"
    )
```

## 6) KV Connector注册模式

```python
def register_connectors():
    _safe_register(
        "ConnectorName",
        "module.path",
        "ClassName"
    )
```

## 7) 量化配置模式

```python
@register_quantization_config("config_name")
class ConfigClass(BaseConfig):
    ...
```
通过 `register_quantization_config` 注册自定义量化实现。

**架构要求：**
- 不使用Patch
- 尽量从开源vLLM的类继承

示例：

```python
# Good: register_quantization_config注册自定义的NPU量化方法，从vLLM　CompressedTensorsConfig继承
NPU_COMPRESSED_TENSORS = "npu-compressed-tensors"
@register_quantization_config(NPU_COMPRESSED_TENSORS)
class NPUCompressedTensorsConfig(CompressedTensorsConfig):
```