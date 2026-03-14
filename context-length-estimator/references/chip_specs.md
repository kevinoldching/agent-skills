# 常见芯片显存规格参考

## NVIDIA GPU

| 型号 | 显存类型 | 显存容量(GB) | 显存带宽(Gbps) | TDP(W) | 备注 |
|------|---------|-------------|---------------|--------|------|
| H100 | HBM3 | 80 | 3.35 | 700 | SXM版本 |
| H100 | HBM3 | 80 | 3.35 | 350 | PCIe版本 |
| H100 | HBM3e | 141 | 4.8 | 700 | H100e |
| H200 | HBM3e | 141 | 4.8 | 700 | |
| A100 | HBM2 | 40 | 1.6 | 400 | SXM |
| A100 | HBM2e | 80 | 2.0 | 400 | A100 80GB |
| A100 | HBM2e | 80 | 2.0 | 400 | PCIe |
| A100-L40 | GDDR7 | 48 | N/A | 350 | |
| L40S | GDDR7 | 48 | 864 | 350 | |
| RTX 4090 | GDDR6X | 24 | 1008 | 450 | 消费级 |
| RTX 3090 | GDDR6X | 24 | 936 | 350 | 消费级 |
| RTX 3080 | GDDR6X | 10 | 760 | 320 | 消费级 |
| RTX A6000 | GDDR6 | 48 | 768 | 300 | 工作站 |
| RTX A5500 | GDDR6 | 24 | 768 | 230 | 工作站 |
| A10 | GDDR6 | 24 | 600 | 150 | 云实例 |
| T4 | GDDR6 | 16 | 320 | 70 | 云实例 |
| V100 | HBM2 | 32 | 900 | 250 | |
| V100 | HBM2 | 16 | 900 | 250 | |

---

## 华为昇腾 (Ascend)

| 型号 | 显存类型 | 显存容量(GB) | 显存带宽(Gbps) | TDP(W) | 备注 |
|------|---------|-------------|---------------|--------|------|
| Ascend 910B1 | HBM2e | 64 | 1.8 | 400 | |
| Ascend 910B2 | HBM2e | 64 | 1.8 | 400 | |
| Ascend 910B3 | HBM2e | 64 | 1.8 | 400 | |
| Ascend 910B4 | HBM2e | 32 | 1.8 | 400 | |
| Ascend 910C | HBM2e | 128 | 1.8 | 400 | |
| Ascend 910A | HBM2 | 32 | 1.2 | 300 | |
| Ascend 310 | DDR4 | 8 | 120 | 8 | 推理卡 |

---

## Intel Gaudi

| 型号 | 显存类型 | 显存容量(GB) | 显存带宽(Gbps) | TDP(W) | 备注 |
|------|---------|-------------|---------------|--------|------|
| Gaudi 2 | HBM2e | 96 | 2.4 | 600 | |
| Gaudi 3 | HBM2e | 144 | 3.35 | 600 | |
| Gaudi | HBM2 | 32 | 1.0 | 350 | 第一代 |

---

## 其他芯片

| 厂商 | 型号 | 显存类型 | 显存容量(GB) | 显存带宽(Gbps) | 备注 |
|------|------|---------|-------------|---------------|------|
| AMD | MI300X | HBM3 | 192 | 5.3 | |
| AMD | MI250X | HBM2e | 128 | 3.2 | |
| AMD | MI210 | HBM2 | 64 | 1.6 | |
| Graphcore | IPU-POD16 | SRAM | 0.9 | N/A | |
| Cambricon | MLU370 | GDDR6 | 64 | 1.2 | 寒武纪 |

---

## 显存估算注意事项

1. **系统开销**: 实际可用显存 ≈ 标称显存 - 2~4GB (驱动/CUDA运行时)
2. **多卡并行**: 需要考虑模型并行开销，每卡通信缓冲区约0.5~1GB
3. **KV Cache动态**: 推理时KV Cache占用随batch size和seq length线性增长
4. **Activation与Batch**: 大batch size时activation可能成为瓶颈
5. **PagedAttention**: vLLM的PagedAttention可减少约10-20%的KV Cache碎片

---

## 常用计算公式

### 模型权重占用 (FP16)
```
模型权重(GB) = 参数总量(B) × 2 / 1024
```

### KV Cache 占用
```
KV Cache(GB) = 2 × batch_size × seq_len × num_layers × num_heads × head_dim × 4 / 1024 / 1024 / 1024
简化: KV Cache(GB) = 2 × batch_size × seq_len × hidden_size × num_layers × 4B / 1024^3
```

### 推理时序列长度估算
```
max_seq_len = (可用显存 - 模型权重) / (每token KV Cache + 每token激活)
```
