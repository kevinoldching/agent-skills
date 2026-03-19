# CLAUDE.md

LLM GPU 显存估算工具，用于计算大语言模型的 GPU 显存占用。

## Development Workflow

### Requirements and Bug Fixes

When a user raises a requirement or problem fix, always provide:

1. **Implementation Plan**: Specific code changes with file paths, line numbers, and code snippets
2. **Verification Plan**: Specific test commands to verify the implementation works correctly

Do NOT implement changes without providing implementation details first.

## Architecture

- **Modular Design**: Attention、FFN、Norm 模块独立实现
- **Configuration-Driven**: 每个模型定义在 YAML 配置文件中
- **Parallelization**: 支持 TP、PP、DP、EP、CP、SP 等并行策略

## Memory Formula

```
Total Memory = Weights + Activations + KV Cache + System Reserved
```

## Data Type Sizes

- FP32: 4 bytes
- FP16/BF16: 2 bytes
- FP8/INT8: 1 byte
- INT4: 0.5 bytes
