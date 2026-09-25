# CTP 单品种三角色复权价穿越回测

本目录包含独立的三角色穿越策略、信号计算、本地数据适配和回测入口，不从 `examples/role_cross` 导入代码。使用 `--product` 选择品种，例如 RB、I 或 HC。角色表的 `main/second/far` 分别映射为主力、次主力、远期；每天使用此前交易日发布的角色记录。`recent` 不参与加载、信号或行情完整性校验。

在 `pro` 目录运行：

```bash
export PYTHONPATH=.
export ROLE_DIR=/home/dev/data/ctp/role
export KLINE_DIR=/home/dev/data/ctp/kline
python demos/role_cross/run_backtest.py \
  --product RB --start-day 2026-01-05 --end-day 2026-05-29 \
  --quantity 1 --log-level WARNING
```

每个角色的真实合约 Bar 用该角色自己的因果累计因子得到研究价。只有三角色**同一分钟**真实 Bar 全部到齐时才计算：

```text
spread = 主力复权价 − (主力复权价 + 次主力复权价 + 远期复权价) / 3
```

`spread` 从零下穿到零上时，目标为当日主力**多 `--quantity` 手**；从零上穿到零下时，目标为当日主力**空 `--quantity` 手**。第一帧只预热；价差未穿越时保留原目标，不人为制造订单。信号使用复权研究价，模拟订单使用真实主力合约未复权 Bar。主力换月时，动态路由先处理旧合约仓位，再在新合约有行情后执行目标。

默认从 `$ROLE_DIR/fut_contract_data.feather`、`$ROLE_DIR/fut_basic.feather`、`$KLINE_DIR` 读取数据；`--contract-struct`、`--fut-basic`、`--bars-dir` 可覆盖路径。合约交易所、乘数、最小价格变动、挂牌日与到期日均从 `fut_basic.feather` 读取，无需指定 `--price-increment`。外部累计因子表不参与信号：三个角色各自根据真实合约同日收盘价计算换月因子。

`--start-day`、`--end-day` 限定回测区间；若实际行情覆盖不足或缺少某日主力、次主力、远期或换月旧主力 Bar，会在撮合前报错。`--starting-balance`、`--commission-per-contract`、`--margin-init`、`--margin-maint`、`--max-notional`、`--max-market-age-seconds` 控制模拟账户与风控。`--log-level ERROR` 减少框架日志；`INFO` 或 `DEBUG` 查看更多细节。`--require-fills` 要求至少一笔模拟成交，初次检查数据时可省略，因为样本可能没有穿越信号。

运行时打印研究数据加载、回放装配和历史回放耗时。结果保存在 `demos/role_cross/results/<run-id>/`，含 `summary.json`、订单、成交、持仓、账户 CSV 和可选 `tearsheet.html`。`--report-dir` 修改输出目录，`--no-tearsheet` 跳过图表。模拟端采用基础净持仓、固定每手手续费和简化保证金规则，不模拟交易所平今/平昨或逐日结算。
