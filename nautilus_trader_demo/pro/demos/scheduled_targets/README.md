# 外部目标计划驱动回测

本策略从 CSV 读取一组**完整目标持仓**，在各计划时点由独立时钟提交，不使用 Bar 回调生成信号。它适合回测已经由研究模型生成的调仓计划：目标正数为多头手数，负数为空头手数，`0` 为平仓。下一时点未列出的旧合约也会因完整组合 `REPLACE` 语义被清零。真实合约、交易所、乘数和最小价格变动均从目标 CSV 与 `fut_basic.feather` 解析；代码没有固定品种或合约，也不引用 `examples/scheduled_targets` 的实现。

目录中已提供可编辑的 `targets.csv` 示例，共 100 条目标记录，覆盖 2026-05-06 至 2026-05-19 的 10 个工作日；每天 5 次开仓、5 次平仓。下方展示前两条记录（同一 `timestamp` 的多行构成一个完整组合）：

```csv
timestamp,target_key,target_qty
2026-05-06 09:40:00,rb2610.SHFE,1
2026-05-06 09:45:00,rb2610.SHFE,0
```

`target_key` 可写为 `真实合约.交易所`，或只写真实合约代码并由 `fut_basic` 确定交易所；旧列名 `instrument_id` 也可用。每个时点的同一目标只能出现一次，手数必须为整数。附带文件演示 RB 合约开仓与平仓；运行前请确认自己的 Bar 文件和 `fut_basic` 覆盖 `rb2610` 及这些日期、时点，并按实际行情覆盖情况修改计划。计划时间默认为 `Asia/Shanghai`，可用 `--timezone` 指定。

在 `pro` 目录运行：

```bash
export PYTHONPATH=.
export ROLE_DIR=/home/dev/data/ctp/role
export KLINE_DIR=/home/dev/data/ctp/kline
python demos/scheduled_targets/run_backtest.py \
  --targets demos/scheduled_targets/targets.csv \
  --start-day 2026-05-06 --end-day 2026-05-29 \
  --log-level WARNING
```

`ROLE_DIR` 提供 `fut_basic.feather`，`KLINE_DIR` 提供 `<合约>_<YYYYMMDD>.feather` 真实 Bar；可分别用 `--fut-basic`、`--bars-dir` 覆盖。`--targets` 必填，计划时点必须落在请求日期范围内。期货目标可跨品种和交易所，前提是每个目标合约都有基础数据和区间内 Bar。指数、期权等没有完整合约与撮合配置的目标会明确拒绝。

行情回放和计划时钟合并时，同一时间戳先处理全部 Bar，再提交目标；新订单要等后续市场事件才能成交。时钟越过的时点记为 `MISSED`，回放结束仍未到达的时点记为 `NOT_REACHED`；默认要求全部时点触发，可用 `--allow-missed-slots` 只查看审计。`--require-fills` 额外要求至少一笔模拟成交。请把首个目标设在有可用行情之后，并给最后一个目标留出后续 Bar，否则可能缺少风控参考价或无法成交。

`--starting-balance`、`--commission-per-contract`、`--margin-init`、`--margin-maint`、`--max-quantity`、`--max-notional` 和 `--max-market-age-seconds` 调整基础模拟账户及风控。结果默认保存在 `demos/scheduled_targets/results/<run-id>/`，含订单、成交、持仓、`schedule_audit.csv`、各交易所账户 CSV、`summary.json` 和可选绩效图；`--report-dir`、`--no-tearsheet` 可覆盖。模拟账户使用净持仓、固定每手手续费与简化保证金规则，不等同于 CTP 实盘平今、平昨和逐日结算。
