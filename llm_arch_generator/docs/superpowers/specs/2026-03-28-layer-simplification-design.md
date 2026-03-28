# Layer 占位符简化设计

> **目标：** 简化 Layer 占位符结构，移除 Add 节点和残差虚线，使 Mermaid 图更清晰易读。

## 背景

当前的 Layer 占位符使用 Add 节点和虚线残差连接来表示 pre-norm 架构：

```mermaid
ln1 --> attn --> add1((+))
   ↑          |
   |--- Residual 1 ---|
add1 --> ln2 --> ffn --> add2((+))
   ↑                   |
   |--- Residual 2 -----|
add2 --> Layer_Out
```

问题：
- 虚线残差连接在展开后显得杂乱
- Add 节点增加视觉复杂度
- pre-norm 是 Transformer 的固有特性，读者已知

## 设计决策

### 简化后的 Layer 占位符

```mermaid
ln1["RMSNorm"]:::norm --> attn_module["Attention"]:::attention --> ln2["RMSNorm"]:::norm --> ffn_module["FFN"]:::moe --> Layer_Out["Output"]
```

**原则：** 残差连接是 pre-norm 架构的内置特性，不需要显式画出来。简洁的线性结构更易于阅读。

## 需要修改的位置

### SKILL.md

1. **Level 2 模板（第226-238行）** — 移除 `add1`、`add2` 和残差虚线
2. **第178-179行** — 移除"每个 transformer 层必须有两条残差路径"说明
3. **第112行** — Level 1 描述中移除"残差连接"
4. **第119行** — 移除残差虚线表示说明
5. **第96行** — 移除残差检测说明
6. **第465行** — 颜色约定表中 "Residual | dashed | #999" 行需要移除或改为 "Aggregation | dashed | #999"，因为 `-.->` 仍用于 MoE 聚合但不再用于残差

### Attention Detail 和 FFN Detail

**不受影响**，它们已经是线性结构：
- Attention: `attn_in --> Q --> K --> V --> RoPE --> Softmax --> O --> attn_out`
- FFN: `ffn_in --> gate --> up --> SiLU --> down --> ffn_out`

### MoE Detail

**不受影响**，MoE 本身没有残差连接：
- `router --> expert_1/2/x/n`
- `shared -.-> MoE_out` 和 `expert -.-> MoE_out` 是 MoE 聚合，不是残差

## 不修改的内容

- 设计文档 `2026-03-21-llm_arch_generator-design.md` 中的 `pre-norm` 描述是架构知识，保留
- 所有模型 YAML 模板无需修改（架构描述不变，只是图示简化）

## 实现步骤

1. 修改 SKILL.md Level 2 模板
2. 更新 SKILL.md 中的相关说明文字
3. 验证修改后的 SKILL.md 模板生成正确
