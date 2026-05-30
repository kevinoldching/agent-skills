# 代码风格检查

## 1) 文件组织

### 文件头注释

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
```

*注* 
- 示例中2025年，根据当前实际的年份可能不同
- 源码文件中添加，其它文件，如ymal/json配置文件不要求

### 导入顺序

```python
# 1. 标准库 (Standard library)
import os
from typing import Optional, Tuple

# 2. 第三方库 (Third party)
import torch
import torch_npu
from vllm.logger import init_logger

# 3. 本地模块 (Local)
from omni_npu.vllm_patches.core import VLLMPatch
from omni_npu.attention import ops
```

## 2) 命名规范

### 文件名
文件名要尽量规范，拼写正确，能够准确地表达文件的功能

### 类

```python
# PascalCase
class NPUSiluAndMul(SiluAndMul):
    ...

class NPUAttentionBackendImpl:
    ...
```

### 函数

```python
# snake_case
def apply_patches():
    ...

def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
    ...
```

### 常量

```python
# UPPER_SNAKE_CASE
MAX_GEAR_NUM = 8
NZ_DIM = 16
BLOCK_NUM_FLOATING_RANGE = 8
```

### 私有成员

```python
# _prefix
_attr_names_to_apply = ['method1', 'method2']
_target = None
```

## 3) 类型提示

```python
from typing import Optional, List, Dict, TYPE_CHECKING

def forward(
    self,
    x: torch.Tensor,
    metadata: Optional[AttentionMetadata] = None
) -> torch.Tensor:
    ...
```

## 4) 文档字符串 (Docstrings)

### Google 风格

```python
def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
    """
    Perform all-reduce operation on the input tensor.

    Args:
        input_: Input tensor to reduce

    Returns:
        The reduced tensor
    """
```

### 类文档字符串

```python
class NPUSiluAndMul(SiluAndMul):
    """
    NPU-optimized SiLU activation followed by element-wise multiplication.

    Uses torch_npu.npu_swiglu for improved performance on Ascend NPUs.
    """
```

## 5) 注释

### 行内注释

```python
# 将 q 转换为适用于 FIA 的 TND 格式
query = query.view(-1, self.num_heads, self.head_size).contiguous()

# 权重使用 NPU 特定格式
torch_npu.npu_format_cast(weight, ACL_FORMAT_FRACTAL_NZ)
```

### 章节注释

```python
# 初始化 NPU 算子
self.fused_op = torch.ops.npu.npu_fused_infer_attention_score

# 配置 KV 缓存
self.kv_cache = ...
```

### 注释语义正确性

注释的语义需要与代码逻辑保持一致

## 6) 错误处理

### 断言

```python
# 用于内部逻辑校验
assert self.num_heads % self.num_kv_heads == 0
assert not (layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0)
```

### 异常处理

```python
try:
    import torch_npu
except Exception as e:
    if hasattr(torch, "npu"):
        return "omni_npu.platform.NPUPlatform"
    raise RuntimeError(f"无法加载 torch_npu: {e}")
```

### 日志

```python
from vllm.logger import init_logger

logger = init_logger(__name__)

logger.info("patch applied: %s => %s.%s", patch_name, module, method)
logger.warning("connector: '%s' already present", connector_name)
logger.error("failed to apply %s: %s", patch_name, e)
```

## 7) 代码布局

### 类结构

```python
class MyClass:
    """Class docstring."""

    # Class attributes first
    CONST_VALUE = 42

    def __init__(self):
        """Initialization docstring."""
        # Instance attributes

    # Public methods
    def public_method(self):
        """Public method docstring."""
        pass

    # Protected methods
    def _protected_method(self):
        """Protected method docstring."""
        pass

    # Private methods
    def __private_method(self):
        """Private method docstring."""
        pass
```

## 8) 行长度

- 最大行宽：100 个字符
- 较长的函数调用应拆分为多行

```python
result = some_function(
    arg1=value1,
    arg2=value2,
    arg3=value3
)
```

## 9) 空格与空行

### 操作符周围

```python
x = a + b  # 二元操作符两端加空格
x = a*b + c*d  # 为了可读性，乘法两端可不加空格
```

### 空行

```python
class ClassA:
    pass


def function_a():
    pass


# 顶级定义（类或函数）之间保留两个空行
```

## 字符串引号 (String Quotes)

- 普通字符串使用双引号 `"`
- 字典键（Keys）使用单引号 `'`

```python
message = "这是一个字符串"
config = {'key': 'value'}
```
