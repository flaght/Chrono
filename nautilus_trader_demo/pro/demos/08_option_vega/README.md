# 认购期权卖方与同月期货对冲策略

## 1. 使用方法

### 启动命令

从 pro 根目录、在已安装运行依赖的环境中执行，先设置 `export PYTHONPATH=.`。入口自动读取已有 `.env` 配置；没有配置数据路径时，可通过可选的 `--data-root` 或各文件路径参数指定。示例日期需与实际数据匹配。

**完整执行（显式设置常用参数）：**

```bash
python demos/08_option_vega/run_backtest.py \
  --option-product MO --future-product IM --index-code 000852 \
  --start-day 2026-09-01 --end-day 2026-09-11 \
  --bar-timestamp end --log-level WARNING --no-tearsheet
```

**默认执行（仅必填参数）：**

```bash
python demos/08_option_vega/run_backtest.py \
  --start-day 2026-09-01 --end-day 2026-09-11
```

默认执行采用参数表中的默认值及已有数据路径配置；例如默认启用绩效图，不要求必须产生成交。

### 参数说明

“必选”表示启动时必须传入；“可选”表示有默认值或可由已有配置解析。数据路径虽然是可选参数，但运行前必须能解析到有效文件。

| 参数 | 必选 / 可选 | 含义与默认值 |
| --- | --- | --- |
| `--start-day / --end-day` | 必选 | 必填，请求日期闭区间 |
| `--data-root` | 可选 | 数据根目录，通常包含 role、kline/fut、kline/opt、kline/index |
| `--fut-dir / --opt-dir / --index-dir` | 可选 | 显式覆盖期货、期权、指数真实分钟行情目录 |
| `--fut-basic / --opt-basic` | 可选 | 显式覆盖基础条款文件，默认 role 目录下 fut_basic.feather / opt_basic.feather |
| `--option-product / --future-product / --index-code` | 可选 | 默认 MO / IM / 000852；期权必须有可用同月期货对冲，指数只用于信号 |
| `--bar-timestamp` | 可选 | 可省略，默认 end；源分钟标签为 start 时须显式指定，公共层只移动一次到分钟结束 |
| `--calendar` | 可选 | 可选显式交易日历；未提供时由三类行情全部已知文件日期的并集推导，信任上游交易日完整性，不自动补工作日 |
| `--starting-balance / --nav` | 可选 | 初始资金默认 40000000；固定策略规模基准默认同初始资金，不随账户权益自动更新 |
| `--vega-budget` | 可选 | 默认 1.25；每一个百分点波动率变化的现金风险预算为 nav × 参数 / 10000 |
| `--delta-targets` | 可选 | 默认 0.2,0.25,0.3,0.35，按最接近各档位的 Delta 选择认购；重复合约合并 |
| `--option-slot / --hedge-slots` | 可选 | 每日选约默认 13:58；对冲复核默认 10:00,11:00,13:30,14:00,14:55；实际期权仓位变化也触发对冲 |
| `--min-remaining-days / --close-remaining-days` | 可选 | 开仓至少剩余 5 个交易日，到期前剩余不超过 2 日清仓；计数不含当天、含最后交易日 |
| `--flatten-slot` | 可选 | 最后回测日开始全组合清仓，默认 14:50；指数必须在该时点之后至少有两分钟行情 |
| `--rate / --min-time-value` | 可选 | 默认 0.02 / 5；固定年化利率及允许反解隐含波动率的最少期权时间价值 |
| `--max-market-age-seconds / --max-iv-age-seconds` | 可选 | 默认 300 / 300 秒，真实价格及成功波动率缓存的最长允许年龄 |
| `--max-option-lots / --max-total-option-lots / --max-future-lots` | 可选 | 默认 100 / 200 / 100；单认购、认购总手数、单对冲期货手数上限 |
| `--commission / --max-notional` | 可选 | 每手每次固定手续费默认 1；单订单及单合约持仓名义金额上限默认 100000000 |
| `--margin-init / --margin-maint` | 可选 | 基础期货保证金比例，默认 0.15 / 0.12 |
| `--option-margin-init / --option-margin-maint` | 可选 | 基础期权保证金比例，默认 1 / 1，以权利金名义金额近似 |
| `--require-fills / --no-tearsheet` | 可选 | 可选要求至少一笔成交 / 跳过绩效图 |
| `--index-precision` | 可选 | 指数价格精度默认 4 |
| `--report-dir` | 可选 | 输出根目录，默认本策略目录的 results/ |
| `--log-level` | 可选 | 框架日志等级，默认 WARNING |

### 数据要求

需要 opt_basic.feather、fut_basic.feather、认购链、同月期货与指数行情。期权最小变动价位直接读取 tickNum，缺失或为空时兼容 minChgPriceNum / price_increment，必须为有限正数；tickUnit 仅为单位说明。未指定 calendar 时用全部已知行情文件日期并集推导交易日；需覆盖期限判断所需的未来日期。末日指数须在清仓时点后至少还有两分钟行情，实际交易腿也须有退出报价和后续撮合机会。

| 数据项 | 要求 |
| --- | --- |
| opt_basic.tickNum | 从每条期权基础条款读取最小变动价位，必须为有限正数；无需命令行或环境变量指定 |

### 输出内容

默认保存到本策略目录的 `results/<run-id>/`，可通过 `--report-dir` 修改根目录。控制台输出数据准备、运行回测、报表与清理的阶段耗时。

| 输出文件 | 内容 |
| --- | --- |
| orders.csv | 模拟委托及订单状态 |
| fills.csv | 实际模拟成交；提交目标不等于成交 |
| positions.csv | 真实合约持仓记录 |
| account.csv | 模拟账户及资金记录 |
| summary.json | 回测摘要、策略参数、信号或调仓统计 |
| input_manifest.json / input_coverage.json / input_issues.json | 输入版本、覆盖与质量诊断 |
| tearsheet.html | 可选绩效图；省略 --no-tearsheet 且绘图依赖可用时生成 |
| signals.csv | 选约、调仓、对冲或退出行为记录 |

## 2. 策略详细说明

### 策略思路与候选选择

这是按固定 Vega 预算卖出虚值认购、再用同月期货管理 Delta 的组合策略。它不比较隐含波动率与预测波动率，也不以 IV 高低设置择时门槛；IV 和风险指标主要用于选约、配仓与对冲。

每日默认 13:58，要求认购、同月期货和指数同一分钟价格都已到达。候选须在生命周期内、行权价高于指数、至少剩余 5 个交易日，并能得到有效 Greeks。先保留合格候选中最近的最后交易日，再为各 Delta 档选择最接近的认购；距离相同时按合约代码排序。

同月期货作为 Black76 定价输入，认购真实权利金用于反解 IV。期限按距离最后交易日 15:00 的自然年化时间计算；开平仓天数阈值则使用交易日计数，两者用途不同。

### Vega 预算与卖方手数

设规模基准为 NAV，预算参数为 b，合约乘数为 m。函数中的 Vega 按波动率变化 1 计算：

```text
现金 Vega 预算 B = NAV × b / 100
每一个百分点波动率变化的现金预算 = B / 100
每个 Delta 档的权重 a_i = 1 / (Gamma_i × m_i)
该档未取整手数 = B × a_i / Σ(a_j × Vega_j × m_j)
```

同一认购可匹配多个 Delta 档，先合并数量，再向下取整并应用单合约、总手数上限，最终目标取负数。默认 NAV 40000000、b=1.25，对应每一个百分点的现金预算 5000。NAV 是固定配仓基准，不随回测账户权益更新；每日重新计算替换总目标，不向原仓位累加。取整或手数上限可能使实际 Vega 低于预算。

### 实际持仓对冲与退出

| 环节 | 当前规则 |
| --- | --- |
| 对冲基础 | 使用实际已成交认购仓位，不用尚未成交的目标仓位代替敞口 |
| 分月汇总 | 每个同月期货分别汇总其对应认购的 `实际手数 × 乘数 × Delta` |
| 期货目标 | `round(-汇总敞口 / 期货乘数)`；卖方认购通常对应多头期货保护 |
| 复核触发 | 目标变化、实际认购持仓变化或指定对冲时点 |
| 在途门控 | 相应订单在途时等待；对冲需求超过上限则停止 |
| 认购退出 | 剩余交易日不超过 2，或以指数计算的时间价值不大于零 |
| 风险指标失效 | 仍需持有的认购无法可靠计算 Delta 时停止；待平认购暂保留已有对冲直到实际退出 |
| 最后回测日 | 默认 14:50 起全部期权和期货目标归零，结束核对实际仓位与在途订单 |

成功 IV 可在有效期内缓存，但重算 Greeks 仍使用当前价格与期限；缓存不代替新鲜行情。场次恢复后等待持仓相关腿的新报价，超时会停止，不能将昨日价格当作今天的对冲依据。默认真实价格与 IV 缓存年龄均为 300 秒。

新分钟到达时先处理上一分钟，避免用尚未送达的同帧行情选约。批量组合目标不保证认购与期货同步成交。本例没有额外止损或价格预测，验收重点是卖方方向、Vega 分配、实际成交后对冲、过期报价诊断和末日实际全平。

## 3. 使用核心组件描述

### dataprep：输入准备

| 关键接口 | 本例用途 |
| --- | --- |
| input_session / resolve_option_args | 共享来源与读取缓存，解析期权、期货、指数行情及基础条款路径 |
| load_calendar / infer_market_calendar | 读取显式日历或推导全部已知行情日期，为期限判断提供日期集合 |
| load_options_basic / load_cffex_futures | 校验股指认购与真实期货条款，返回规范化静态资料 |
| inventory / BarFileKey / plan_option_chain | 索引三类行情，声明选约及持仓管理所需文件，生成 InputPlan |
| BarReadSpec / prepare_option_chain | 按日盘一分钟统一时间、身份和严格 OHLCV 校验，返回 HistoricalInputBundle |
| add_prepared_bar_source | 注册公共层已准备行情，不重新读取原始文件或重复移动时间 |
| write_input_reports | 输出输入版本、覆盖与问题诊断 |

### datahub：参考资料查询

| 关键接口 | 本例用途 |
| --- | --- |
| 本例不调用参考资料库 | 固定认购条款、同月期货映射及日历由公共层准备，策略直接使用这些静态资料，不查询连续主力角色或复权价格 |

### market：真实行情回放

| 关键接口 | 本例用途 |
| --- | --- |
| FileReplayFeed / InstrumentMeta | 注册认购、同月期货与指数真实行情；指数仅提供信号行情元信息 |
| DataType.BAR / MarketStreamBinding | 声明交易腿一分钟回放流；指数不创建执行合约或交易路由 |

### trader：卖方及对冲执行

| 关键接口 | 本例用途 |
| --- | --- |
| instrument_factory.make_option / make_future / instrument_meta | 构造原生期权、期货合约及对应行情元信息，无例子内转发文件 |
| StrategyTemplate / set_targets | 提交认购和对冲期货的完整目标，包括旧腿零目标 |
| CtpFuturesBasicProfile / NautilusSimExecutionBackend | 模拟账户、手续费、基础保证金及原生合约回测 |
| PositionManager / NetTargetOrderPlanner | 管理实际与在途持仓，把目标转换为净持仓订单 |
| MarketReferencePriceStore / PreTradeRiskManager / RiskLimits | 校验真实报价时效、手数和名义金额，指数没有交易风控路由 |
| DataBinding / ExecutionRoute | 按符号绑定三类信号行情，只给期权与期货创建执行路由 |
| UnifiedStrategyRunner / NautilusMarketFeedAdapter / UnifiedHistoricalRuntime | 串联真实行情、策略、模拟执行与历史回放生命周期 |
