# CTP 多品种次主力信号、主力合约执行

本目录包含完整的策略、信号核、本地数据适配和回测入口，不从 `examples/black_sector` 导入代码。默认使用 `JM,I` 两个领头品种与 `RB` 比较：三个品种的次主力真实 Bar 经因果复权后，在同一时间戳形成完整信号帧；领头品种收益率的平滑均值与比较品种收益率决定多空。订单只在指定执行品种的当日主力真实合约上模拟撮合。换月由框架动态路由处理旧仓与新主力目标。

在 `pro` 目录运行，先使用与 `demos/main_ema` 相同的环境变量：

```bash
export PYTHONPATH=.
export ROLE_DIR=/home/dev/data/ctp/role
export KLINE_DIR=/home/dev/data/ctp/kline
python demos/black_sector/run_backtest.py \
  --start-day 2026-01-05 --end-day 2026-05-29 \
  --leader-products JM,I --comparison-product RB --execution-product RB \
  --quantity 1 --return-period 30 --sector-period 15 \
  --log-level WARNING --require-fills
```

## 命令参数与策略逻辑

| 参数 | 作用 |
| --- | --- |
| `--start-day`、`--end-day` | 请求的回测日期范围。 |
| `--leader-products JM,I` | 两个领头品种，用它们的收益率构造行业参考值；必须是两个不同品种。 |
| `--comparison-product RB` | 与行业参考值比较的第三个品种。 |
| `--execution-product RB` | 实际交易的品种，须为上述三个品种之一；示例交易 RB 当日主力。 |
| `--quantity 1` | 多空目标仓位的绝对手数；反手时可能生成平仓、开仓两张订单。 |
| `--return-period 30` | 分别计算 JM、I 最近 30 个同步 Bar 收益率的平均值。 |
| `--sector-period 15` | 将两个领头品种的平均收益率取均值，再对所得序列取最近 15 个值的平均。 |
| `--log-level WARNING` | 仅显示 WARNING 和 ERROR 级别的框架日志。 |
| `--require-fills` | 要求回测至少产生一笔模拟成交，否则报错；首次检查数据接入时可省略。 |

注意参数名是 **`--require-fills`**（复数），写成 `--require-fill` 会报 `unrecognized arguments`。

每个时间戳先等待 JM、I、RB 三个品种的**次主力 Bar 全部到齐**，用复权收盘价计算各品种相对上一同步 Bar 的收益率。完成窗口预热后，若 RB 当前收益率高于行业参考值，提交 RB 主力**多 1 手目标仓位**；低于则提交**空 1 手目标仓位**；相等时保持原方向。`--quantity` 改变目标仓位的绝对手数。信号只用次主力，订单在执行品种的主力真实合约上模拟撮合；换月时动态切换执行合约。

`30` 和 `15` 都是**同步 Bar 数，不是交易日数**。默认信号角色为 `secondary`（角色表 `second` 列），执行角色为 `main`；可用 `--signal-role`、`--execution-role` 更改。`--submission-delay-bars` 指定信号后再等多少根执行合约 Bar 才提交目标，默认 0。目标提交不等于成交，实际成交取决于后续行情事件。

默认从 `$ROLE_DIR/fut_contract_data.feather`、`$ROLE_DIR/fut_basic.feather`、`$KLINE_DIR` 读取数据；可用 `--contract-struct`、`--fut-basic`、`--bars-dir` 覆盖。交易所、最小价格变动和乘数来自 `fut_basic.feather`。还可配置 `--starting-balance`、`--commission-per-contract`、`--margin-init`、`--margin-maint`、`--max-notional`。框架日志默认 `WARNING`；用 `--log-level ERROR` 减少输出，或用 `INFO`、`DEBUG` 查看更多细节。

回测前会打印请求与实际行情范围；两端或中间缺口超过 14 个自然日会报错。所需信号合约、执行合约及换月日旧主力 Bar 缺失也会报错。成功运行后，`demos/black_sector/results/<run-id>/` 保存 `summary.json`、订单、成交、持仓和各交易所账户 CSV；可选生成 `tearsheet.html`。`--report-dir` 修改报表根目录，`--no-tearsheet` 跳过图表，`--require-fills` 要求至少一笔模拟成交。

运行时会分别打印研究数据加载、回放装配、历史回放的耗时，便于定位慢在读取 Feather、注册合约，还是撮合阶段。若出现 `行情时间差...超出允许范围...`，表示订单所需真实合约的参考行情超过 `--max-market-age-seconds`（默认 120 秒）；跨夜或换月时不能靠放宽时效来假装行情仍新鲜。自动换月会等待旧合约的当日 Bar 后平旧仓，再等新合约的当日 Bar 后开新仓。

模拟端采用基础净持仓、固定每手手续费和简化保证金规则；不模拟 CTP 平今/平昨或交易所逐日结算。信号 Bar 提交目标后，成交仍取决于后续行情事件。
