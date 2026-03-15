# 修复总结

## 修复的问题

### 1. 权重合并逻辑 ✅
**问题**: 每一层的每个权重都被单独存储，导致 YAML 文件过大（29514 tokens）

**修复**: 重写 `_classify_weights` 方法
- 按基础模式分组权重（去掉层号）
- 将相同模式的权重合并为一个条目
- 使用 `layers` 字段表示出现的层数

**结果**: YAML 文件从 29514 tokens 减少到 198 行（3.9KB）

### 2. 权重分类失败 ✅
**问题**: 所有权重都被分类为 `others`

**修复**:
1. 修复 YAML 规则文件结构 - 将嵌套的 `ffn.router` 改为平级的 `ffn_router`
2. 修复 `WeightClassifier._match_pattern` 方法 - YAML 中的模式已经是正则表达式格式，不需要再次转义
3. 添加 `_resolve_inheritance` 方法 - 支持规则继承（如 `gpt_oss: inherit: generic`）

**结果**: 权重被正确分类到 5 个模块：attention, embedding, ffn_expert, ffn_router, norm

### 3. 数据类型支持 ✅
**问题**: 不支持 `U8` (uint8) 数据类型

**修复**: 在 `get_dtype_bytes` 函数中添加 `uint8`, `u8`, `u4` 支持

## 测试结果

### GPT-OSS-120B 模型

**生成的 YAML 文件**:
- 大小: 198 行, 3.9KB
- 模块分类: 5 个模块（attention, embedding, ffn_expert, ffn_router, norm）
- 权重数量: 22 个权重模式（vs 之前的 687 个单独权重）

**显存估算结果**:
```
总显存: 64.45 GB
- 权重: 60.77 GB (94.3%)
  - ffn_expert: 56.80 GB (93.5%)
  - embedding: 2.16 GB (3.6%)
  - attention: 1.78 GB (2.9%)
  - ffn_router: 0.02 GB (0.0%)
  - norm: 0.00 GB (0.0%)
- KV Cache: 0.10 GB (0.2%)
- 激活值: 1.58 GB (2.5%)
- 系统保留: 2.00 GB (3.1%)
```

## 修改的文件

1. `llm_mem_estimator/model_detector.py`
   - 重写 `_classify_weights` 方法（权重合并逻辑）
   - 删除 `_count_weight_layers` 方法（不再需要）

2. `llm_mem_estimator/weight_classifier.py`
   - 添加 `_resolve_inheritance` 方法
   - 修复 `_match_pattern` 方法
   - 更新 `classify_weight` 方法以支持新的规则结构

3. `llm_mem_estimator/model_config.py`
   - 添加 `uint8`, `u8`, `u4` 数据类型支持

4. `configs/weight_mapping_rules.yaml`
   - 修复规则结构（将嵌套改为平级）
   - 统一命名：`ffn_router`, `ffn_expert`, `ffn_shared_expert`, `ffn_dense`

## 验证

所有功能测试通过：
- ✅ 从 HuggingFace 获取元数据（使用 `get_safetensors_metadata`，无需下载）
- ✅ 权重合并（687 个权重合并为 22 个模式）
- ✅ 权重分类（正确分类到 5 个模块）
- ✅ YAML 文件生成（198 行，3.9KB）
- ✅ 显存估算（64.45 GB）
- ✅ 模块化架构正常工作
