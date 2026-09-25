# 黑色系三腿相对价值回测

这个策略对应“IM 多、IH 和 IC 空；或者反向”的**三腿方向结构**，在黑色系可先用 `RB` 为主腿、`HC,I` 为两条对冲腿进行研究。RB、HC 都是钢材，I 是铁矿石；三者有产业关联，但价格关系可能长期变化。这是统计相对价值交易，**不是无风险三角套利**，三腿下单也不保证同时成交。品种和权重均可配置。

在 `pro` 目录运行：

```bash
export PYTHONPATH=.
export ROLE_DIR=/home/dev/data/ctp/role
export KLINE_DIR=/home/dev/data/ctp/kline
python demos/triangle_spread/run_backtest.py \
  --anchor RB --hedges HC,I --hedge-weights 0.5,0.5 \
  --start-day 2026-01-05 --end-day 2026-05-29 \
  --lookback 120 --entry-z 2 --exit-z 0.5 \
  --target-notional 300000 --rebalance-interval 5 \
  --log-level WARNING
```

`--anchor` 是主腿；`--hedges` 是两个不同的对冲品种。三者均使用上一交易日角色表确定的当日**主力真实合约**。信号价为真实合约收盘价乘因果复权因子；订单仍在真实合约的未复权 Bar 上模拟成交。换月时旧合约目标清零，新主力按当前方向重新设目标，换月日需要旧、新合约 Bar。

首个完整同步 Bar 的三品种复权价作为归一化基准。每根完整同步 Bar 计算

```text
spread = log(RB / RB基准) - 0.5 × log(HC / HC基准) - 0.5 × log(I / I基准)
z = (当前spread - 前120根spread均值) / 前120根spread标准差
```

计算当前 `z` 时窗口只含**之前**的同步 Bar。`z <= -2` 表示主腿相对偏低，目标为**多 RB、空 HC、空 I**；`z >= +2` 表示主腿相对偏高，目标反向。持仓后 `|z| <= 0.5` 则三腿目标归零。达到入场阈值时可直接反向；窗口标准差为零时 `z=0`。`--lookback`、`--entry-z`、`--exit-z` 控制这些阈值，窗口单位是同步 Bar 根数，不是交易日。

`--target-notional 300000` 表示主腿目标名义金额 30 万元；权重 `0.5,0.5` 表示两条对冲腿各约 15 万元，组合总名义金额约 60 万元。实际手数按 `目标金额 / (真实收盘价 × 合约乘数)` 向下取整；若任一腿不足一手，回测报错，避免悄悄变成两腿。权重必须为正且合计为 1。`--rebalance-interval` 控制持仓期间按多少根同步 Bar 重新配手数；方向变化、平仓或换月会立即更新目标。此配比只按名义金额，**不保证价格敏感度或产业因子中性**。

默认从 `$ROLE_DIR/fut_contract_data.feather`、`$ROLE_DIR/fut_basic.feather`、`$KLINE_DIR` 读取数据。`--contract-struct`、`--fut-basic`、`--bars-dir` 可覆盖路径；交易所、价格最小变动和乘数均从元数据读取。起始资金、每手手续费、保证金、风控上限和日志级别也可传参。回测会检查请求区间覆盖与所需 Bar；结果保存在 `demos/triangle_spread/results/<run-id>/`，包括订单、成交、持仓、账户 CSV、JSON 摘要和可选绩效图。`--no-tearsheet` 跳过图表，`--require-fills` 要求至少一笔成交。
