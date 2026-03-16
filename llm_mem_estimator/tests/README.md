# LLM Memory Estimator 测试文档

本文档说明 tests 目录下各测试用例的功能点、测试流程和调用方法。

## 目录结构

```
tests/
├── conftest.py                 # pytest 配置和工具函数
├── test_models.yaml           # 测试模型配置（指定 HuggingFace 模型）
├── test_*.py                  # 测试用例文件
└── outputs/                   # 生成的模型配置（测试数据）
```

---

## 测试流程

### 1. 配置测试模型

编辑 `test_models.yaml` 指定要测试的 HuggingFace 模型：

```yaml
models:
  # HuggingFace 公开模型
  - name: Qwen2.5-0.5B
    type: huggingface
    path: Qwen/Qwen2.5-0.5B
    description: Qwen 0.5B 模型
```

### 2. 生成模型配置

运行 `test_model_detection.py` 为指定模型生成配置文件：

```bash
# 为 HuggingFace 模型生成配置（会下载到 outputs/）
python tests/test_model_detection.py --model Qwen/Qwen2.5-0.5B

# 或者运行 pytest
pytest tests/test_model_detection.py -v
```

生成的文件：
- `outputs/<model>_config.yaml` - 模型配置文件
- `outputs/<model>_weights_metadata.md` - 权重元数据

### 3. 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_model_config.py::TestConfigFromOutputs -v
pytest tests/test_memory_estimator.py::TestMemoryEstimatorFromOutputs -v
pytest tests/test_integration.py::TestEndToEndFromOutputs -v
```

---

## 为什么只测试 HuggingFace 模型

**原因**：test_model_detection.py 需要从模型权重中提取元数据来生成配置。

- **HuggingFace 模型**：可以从 safetensors 元数据中获取权重信息 → 可以生成配置
- **本地模型**：已有 YAML 配置文件在 `configs/models/`，不需要再生成

**测试覆盖**：
| 数据来源 | 用途 |
|----------|------|
| HuggingFace 模型 | test_model_detection.py 生成配置到 outputs/ |
| outputs/*.yaml | test_model_config.py, test_memory_estimator.py, test_integration.py 测试 |
| configs/models/*.yaml | test_configs.py 验证配置文件有效性 |

---

## 测试文件说明

### 1. test_models.yaml

**功能**：指定要测试的 HuggingFace 模型列表

**配置项**：
- `name` - 模型名称
- `type` - 模型类型（当前只支持 `huggingface`）
- `path` - HuggingFace 模型路径
- `description` - 模型描述

---

### 2. conftest.py

**功能**：pytest 配置和工具函数

**主要函数**：
- `get_test_models_config()` - 加载 test_models.yaml
- `get_outputs_configs()` - 获取 outputs/ 目录下的所有配置

---

### 3. test_model_detection.py

**功能**：
- 从 HuggingFace 获取模型配置和权重元数据
- 生成配置文件到 outputs/ 目录

**调用方法**：
```bash
# 生成指定模型配置
python tests/test_model_detection.py --model Qwen/Qwen2.5-0.5B

# 运行测试
pytest tests/test_model_detection.py -v
```

---

### 4. test_model_config.py

**功能**：基于 outputs/ 目录的参数化测试

**测试内容**：
- 加载 YAML 配置
- 验证模块存在
- 验证计算规则存在

**调用方法**：
```bash
pytest tests/test_model_config.py::TestConfigFromOutputs -v
```

---

### 5. test_memory_estimator.py

**功能**：基于 outputs/ 目录的显存估算参数化测试

**测试内容**：
- 计算权重显存
- 计算 KV Cache 显存
- 计算激活值显存
- 完整显存估算

**调用方法**：
```bash
pytest tests/test_memory_estimator.py::TestMemoryEstimatorFromOutputs -v
```

---

### 6. test_integration.py

**功能**：基于 outputs/ 目录的端到端参数化测试

**测试内容**：
- 端到端流程
- 并行策略测试
- 多 batch size 测试
- 多序列长度测试

**调用方法**：
```bash
pytest tests/test_integration.py::TestEndToEndFromOutputs -v
```

---

## 测试优先级

| 优先级 | 测试 | 说明 |
|--------|------|------|
| 1 | test_model_detection.py | 生成 outputs/ 配置 |
| 2 | test_model_config.py | 配置加载测试 |
| 3 | test_memory_estimator.py | 显存计算测试 |
| 4 | test_integration.py | 端到端测试 |
| 5 | test_configs.py | 配置文件验证 |
| 6 | test_model_detector.py | 权重分类测试 |

---

## 注意事项

1. **网络要求** - test_model_detection.py 需要访问 HuggingFace
2. **outputs/ 目录** - 除 test_model_detection.py 外，其他测试依赖 outputs/ 目录
3. **本地模型** - configs/models/ 下的本地模型配置会被 outputs/ 测试自动覆盖
