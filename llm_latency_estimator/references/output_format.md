# LLM Roofline 性能分析报告 - 输出格式参考

## 报告结构

```
## 模型: [模型名]
## 芯片: [芯片型号]

### 注意力类型: [MHA / GQA / MLA / ...]
### 关键参数: hidden_dim=XXX, n_heads=XXX, n_kv_heads=XXX, ffn_dim=XXX

### 模块级分析

| 模块 | FLOPs (T) | 数据传输 (GB) | 计算强度 | 瓶颈类型 |
|------|-----------|--------------|---------|---------|
| Embedding | 0.01 | 0.5 | 0.02 | Memory-Bound |
| Attention | 2.5 | 1.2 | 2.08 | Compute-Bound |
| FFN | 5.3 | 2.1 | 2.52 | Compute-Bound |
| Output | 0.02 | 0.01 | 2.0 | Balanced |

### Attention Decode 瓶颈转折点
- 临界 seq_len ≈ [AI 计算值] tokens
- seq_len < [临界值]: Compute-Bound
- seq_len > [临界值]: Memory-Bound

### Roofline 图示

使用 ASCII 图表格式，避免 Mermaid 兼容性问题：

```
AI Balance ([芯片]) = [值] FLOPs/Byte

Memory-Bound 区域  |  Compute-Bound 区域
(I < [AI_Balance]) |  (I > [AI_Balance])
                  |
    ↑             |  ← Attention: [值]
    |             |  ← FFN: [值]
    | 斜率        |  ←─────────────── Peak: [峰值] TFLOPS
    | Bandwidth   |
    | [带宽] GB/s |
    |             |
    +-------------+──────────────→
    0        [AI_Balance]    I
```

### 整体评估

- 模型总计算强度: XXX FLOPs/Byte
- 硬件 AI Balance: XXX FLOPs/Byte
- 瓶颈分布: XX% Compute-Bound, XX% Memory-Bound
```

## 时延估算格式

```markdown
## 时延估算

### 硬件参数
- 芯片: [芯片型号]
- 峰值算力: [值] TFLOPS
- 带宽: [值] GB/s

### 输入参数
- prompt_len: [值]
- gen_len: [值]
- 注意力类型: [类型]

### 阶段时延

| 阶段 | 时延 |
|------|------|
| Prefill | [值] ms |
| Decode (单 token) | [值] ms |
| 端到端总时延 | [值] ms |

### 时延分解

Prefill:
  - Embedding: [值] ms
  - Attention: [值] ms
  - FFN: [值] ms
  - Output: [值] ms

Decode (per token):
  - Attention (Memory-Bound): [值] ms
  - FFN (Compute-Bound): [值] ms
  - Output: [值] ms
```

## 优化建议格式

```markdown
## 性能优化建议

### 瓶颈分析总结

| 模块 | 瓶颈类型 | 占比 |
|------|---------|------|
| Attention | Memory-Bound | XX% |
| FFN | Compute-Bound | XX% |
| Embedding | Memory-Bound | XX% |

### 针对性建议

#### 1. [Memory-Bound 模块]

- 🔴 **当前问题**: [AI 分析的具体瓶颈描述，如 KV cache 访存量过大]
- ✅ **建议 1**: [具体优化措施]
- ✅ **建议 2**: [具体优化措施]

#### 2. [Compute-Bound 模块]

- 🔴 **当前问题**: [AI 分析的具体问题，如计算密度不足]
- ✅ **建议 1**: [具体优化措施]
- ✅ **建议 2**: [具体优化措施]

#### 3. 硬件配置优化

- ✅ **建议**: [针对芯片特性的优化]

#### 4. 架构级建议

- ✅ **建议**: [如 PD 分离部署等架构层面的优化]
```
