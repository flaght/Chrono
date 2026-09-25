# CTP 品种主力 EMA 回测

本目录的策略通过 `--product` 选择品种，例如 `RB`、`I`、`HC`。上一交易日角色表确定当天主力，已到达的真实合约 Bar 收盘价乘以因果复权因子计算 EMA；订单仍在真实合约的未复权 Bar 上模拟成交。换月由动态路由先处理旧合约仓位，再执行新主力目标。

先在运行环境设置数据目录：

```bash
export ROLE_DIR=/home/dev/data/ctp/role
export KLINE_DIR=/home/dev/data/ctp/kline
export PYTHONPATH=.
```

在 `pro` 目录运行：

```bash
python examples/main_ema/run_backtest.py \
  --product RB --start-day 2026-01-05 --end-day 2026-01-26
```

其他品种使用同一入口，把 `--product RB` 换成 `--product I`、`--product HC` 等。每个合约的最小价格变动直接读取 `fut_basic.feather` 中对应行的 `minChgPriceNum`；该字段缺失、为空或非正数时直接报错。交易所也从合约元数据读取，不在策略中写死。还可设置 `--fast`、`--slow`、`--quantity`、`--max-notional` 和 `--require-fills`。

默认读取 `$ROLE_DIR/fut_contract_data.feather`、`$ROLE_DIR/fut_basic.feather` 和 `$KLINE_DIR` 下的真实合约 Feather Bar；`--contract-struct`、`--fut-basic`、`--bars-dir` 可以覆盖这些位置。输入文件名应为 `品种合约_YYYYMMDD.feather`，换月日还必须有旧主力 Bar。主力复权因子根据真实合约的历史收盘价计算；这条单品种路径不读取外部 `fut_adjustment_factors.feather`。

启动时会打印请求区间、实际识别到的首末交易日及天数。若请求区间前后或中间有超过 14 个自然日的行情缺口，回测会在撮合前报错，避免把仅有一个月的数据误认成半年结果。结束日期可以是周末，例如 2026-07-26 是周日，此时最后一根行情可以早于请求日期。

框架日志默认 `WARNING`；用 `--log-level INFO` 查看订单明细，或用 `--log-level ERROR` 只看错误。回测通过后，输出位于 `examples/main_ema/results/<run-id>/`：`summary.json`、`orders.csv`、`fills.csv`、`positions.csv`、`account.csv`，安装 Plotly 后另有 `tearsheet.html`。可用 `--report-dir` 改输出根目录，`--no-tearsheet` 只导出数据。

回测使用 `CtpFuturesBasicProfile` 的基础净持仓撮合、固定每手手续费和简化保证金规则；不等于交易所平今平昨、逐日结算或真实柜台绩效。请在有 Bomber/Nautilus 依赖与历史数据的环境验收结果。
