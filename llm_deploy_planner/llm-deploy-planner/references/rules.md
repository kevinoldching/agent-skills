# PD 部署规划参考规则

## 关键公式

| 公式 | 说明 |
|------|------|
| `Decode TPS_混部 = batch_size × (S_in + S_out) / (TTFT + S_out × TPOT)` | 混部 TPS |
| `Decode TPS_分离 = batch_size / TPOT` | 分离 TPS |
| `prefill_cards = TP_prefill × DP_prefill` | Prefill 每实例卡数 |
| `decode_cards = TP_decode × DP_decode` | Decode 每实例卡数 |
| `x = ceil(prefill_cards_total / cards_per_instance)` | Prefill 实例数 |

## 约束条件

始终验证：
1. `TP <= 单机卡数 或 TP = 单机卡数 × 机器数量`
2. `TP × DP = EP`
3. `EP < 实例总卡数`
4. `[若MoE] EP >= 2 且 EP <= expert数量`
5. **xPyD: y = 1（固定，不可更改），x × TP_prefill × DP_prefill = y × TP_decode × DP_decode（Prefill和Decode总卡数必须相等）**
6. 总卡数 <= 可用卡数
7. TP/EP 取值为2的幂；TP 在 [1, 单机卡数] 范围内为 1, 2, 4, 8, ...；超出单机卡数后为 单机卡数×2^k（k=1,2,3,...）；EP=1 仅限非MoE模型；MoE模型 EP >= 单机卡数
8. **总卡数必须是单机卡数的整数倍**（total_cards % 单机卡数 == 0，避免出现56、42等无法整除的值）

## 并行策略约束速查

| 规则 | 说明 |
|------|------|
| TP × DP = EP | EP 由 TP 和 DP 决定 |
| Dense 模型 EP = 1 | 无需 Expert Parallel |
| MoE 模型 EP >= 单机卡数 | EP 不能小于单机卡数 |
| 总卡数 % 单机卡数 = 0 | 卡数必须整除 |
