# NPU适配的特定设计原则

## 1) 设备管理

### 设备选择

```python

# 正确 - 通过current_platform.device_type来获取
torch.tensor(0, dtype=torch.float32, device=current_platform.device_type)

# 错误 - 硬编码设备
torch.tensor(0, dtype=torch.float32, device=torch.device(f"npu:{self.local_rank}")
```

### 设备可用性检查

```python
if not hasattr(torch, "npu"):
    raise RuntimeError("torch_npu is not available")
```

## 2) NPU 算子

### 激活函数

| 通用 (Generic) | NPU 优化 (NPU Optimized) |
|---------|---------------|
| `F.silu(x) * y` | `torch_npu.npu_swiglu(x)` |
| `F.gelu(x)` | `torch_npu.npu_gelu(x)` |

```python
# 正确 - 使用 NPU 算子
@SiluAndMul.register_oot
class NPUSiluAndMul(SiluAndMul):
    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_swiglu(x)
```

### 归一化

```python
# 正确 - NPU RMSNorm
output, _ = torch_npu.npu_rms_norm(x, weight, epsilon)

# 正确 - Add + RMSNorm 融合算子
output = torch_npu.npu_add_rms_norm(x, residual, weight, epsilon)

# 错误 - 通用 PyTorch 实现 (在 NPU 上运行较慢)
output = F.rms_norm(x, (x.shape[-1],), weight, epsilon)
```

### Rotary Embedding

```python
# 正确 - NPU RoPE
output = torch_npu.npu_apply_rotary_pos_emb(
    query, key, cos, sin,
    interleaved=False,
    input_layout='TND'
)

# 布局 (Layout) 选项: 'TND', 'BSH', 'BNSD'
```

## 3) 通信操作

### HCCL 操作

```python
from vllm.distributed.parallel_state import get_tp_group

# All Reduce
get_tp_group().device_group.all_reduce(tensor)
get_tp_group().device_group.all_gather(tensor, dim=0)

# All to All
torch.distributed.all_to_all_single(output, input, group=tp_group)
```

### NCCL -> HCCL 替换

| CUDA | HCCL |
|------|------|
| `torch.distributed.nccl` | `torch.distributed` (HCCL backend) |
| `CudaCommunicator` | `NPUCommunicator` |



## 4) 环境变量

| 变量名 | 用途 | 有效值 |
|----------|---------|--------------|
| `ASCEND_RT_VISIBLE_DEVICES` | 设备可见性 | "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" |
| `TORCH_COMPILE_GE` | GE 图编译 | "True", "False" |
| `OMNI_NPU_VLLM_PATCHES` | Patch 选择 | "ALL", "PatchA,PatchB" |
| `OMNI_NPU_PATCHES_DIR` | 手动指定 Patch 目录 | 路径字符串 |

## 5) 性能优化

### Contiguous Tensors

```python
# 在执行 NPU 算子前确保张量内存连续
x = x.view(...).contiguous()
```

## 6) 特定的架构设计原则

### Layer设计

**Base Layer vs. 高性能Layer**
- Base Layer：layer模块与开源vLLM模块接口兼容，可能替换了底层的算子或实现逻辑，原生vLLM模型脚本可以直接调用，位于`src/omni_npu/layers`下
- 高性能Layer：layer模块与开源vLLM模块接口可能不兼容，为了实现最佳性能做了特定优化，原生vLLM模型脚本不能直接使用，位于`src/omni_npu/v1/layers`下

**设计原则要求：**
- 新增的Base Layer及高性能Layer模块，应该尽量复用已有的模块
- 除sampler模块外，其它新增Layer模块，应该以vLLM已有的插件/注册等方式进行扩展，避免patch

### 自定义模型脚本

为了在NPU平台上实现极致性能，当前对vLLM开源模型脚本做了适配，自定义实现位于`src/omni_npu/v1/models`下，应该尽量避免对vLLM模型脚本直接打补丁。

**设计原则要求：** 
- 如果实现了自定义的脚本，请确保一定引用了`src/omni_npu/layers`下面的模块，否则应该给出违反设计原则的告警。