# CTP 多品种主力截面动量回测

使用 `--products` 指定品种，例如 `RB,HC,I`。每天根据上一交易日的合约角色表选出各品种主力，在所有品种的同一分钟主力 Bar 到齐后计算复权收益率排名：做多最强，做空最弱，其余品种目标为零。换月时旧真实合约的目标归零，新主力按信号建立目标仓位。旧主力在换月日也需要 Bar，用于模拟平仓。

在 `pro` 目录运行：

```bash
export PYTHONPATH=.
export ROLE_DIR=/home/dev/data/ctp/role
export KLINE_DIR=/home/dev/data/ctp/kline
python examples/cross_section/run_backtest.py \
  --products RB,HC,I \
  --start-day 2026-01-05 --end-day 2026-07-26 \
  --lookback 20 --rebalance-interval 5 --target-notional 100000
```

默认读取 `$ROLE_DIR/fut_contract_data.feather`、`$ROLE_DIR/fut_basic.feather` 和 `$KLINE_DIR`；可分别用 `--contract-struct`、`--fut-basic`、`--bars-dir` 覆盖。Bar 文件名为 `真实合约_YYYYMMDD.feather`，需有 `datetime` 或 `timestamp` 及 OHLCV 列。交易所、最小价格变动、合约乘数、挂牌和到期日由 `fut_basic.feather` 读取。策略参数、起始资金、手续费、保证金、风控上限、日志级别和报表目录都可从命令行配置。

回测启动时打印请求区间与实际行情范围。若前后或中间有超过 14 个自然日的行情缺口，或所需主力/换月旧合约缺少 Bar，会在撮合前报错。结果保存在 `examples/cross_section/results/<run_id>/`，包括订单、成交、持仓、账户 CSV、JSON 摘要及可选绩效图。多腿目标同批提交，不保证交易所层面原子成交；模拟端使用基础净持仓、固定每手手续费和简化保证金规则。
