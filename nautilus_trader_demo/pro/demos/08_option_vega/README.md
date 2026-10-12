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
| 参考资料来源 | 回测使用文件条款；Live使用DolphinDB条款快照。策略只使用准备后的认购条款、同月期货映射和日历，不查询数据库、连续主力角色或复权价格 |

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

## 回测与 Live 共用装配（2026-10-11）

`assembly.py` 统一创建策略、数据绑定和执行路由；`run_backtest.py` 和
`run_live.py` 分别提供行情与执行积木。08 不引用其他 demo 文件。
策略使用 `position()` 和 `strategy_working_quantity()` 查询自己的归属成交及
委托分配；物理账户总仓和其他策略的委托不参与 Vega／Delta 对冲计算。

```text
文件历史 Bar → 有序历史分钟 ─────────────────┐
CTP LAST成交／MID报价 → 分钟Bar ─┐           │
DolphinDB期权／期货分钟流 ──────┼─ 公共分钟帧 ├─ 同一个 OptionVegaStrategy
DolphinDB指数分钟流（独立来源）─┘           │             │
                                          ┘         完整组合目标
                                                        │
                                 回测模拟／Recording／Model／原生CTP
```

Live 入口支持：

- `--market-source ctp|dolphindb`：期权及同月期货；来源明确绑定，禁止隐式回退。
- `--index-source dolphindb`：独立指数通道，只有行情绑定，没有交易路由。
- `--mode recording`：仅记录目标，不连接 TD、不产生假成交。
- `--mode model`：公共参考价模型成交，维护独立策略账本，不查询物理仓位。
- `--mode simnow`：公共原生 CTP 执行，需要显式报单授权；支持
  `--account-reconciliation strict|observe|off`，默认 strict。
- 一对多第三方跟随端通过 `build_runtime(args, account_routes=...)` 与公共
  `AccountExecutionRoute` 装配，策略主账本选择 Model；没有新增第三方连接器。

指数流表和期权／期货流表必须是**已收盘分钟流**，字段遵循公共
`DolphinDbStreamSpec.bar()` 契约。`--index-bar-timestamp`、
`--market-bar-timestamp` 显式声明 start/end 标签。Live 不读取静态指数文件
假装当前指数行情，不用期货价格替代指数；历史文件仍走回测入口。

### 时间、同步与 Replay

08 的 Replay 与 01 的接收时间工程聚合不同：**保留原始事件时间估值**。
`--simnow-environment replay --replay-md-trading-day YYYYMMDD` 只豁免历史事件
相对墙钟的年龄；原始 MD 交易日、接收健康、时间倒退和来源日期仍受校验。

- 行情帧 `valuation_ns` 决定选约日期、13:58 时点和 Black76 剩余期限。
- `request_ns` 使用本次接收墙钟，供执行请求和下单时效检查。
- 同一分钟等待 `--frame-wait-seconds`（默认2秒），封闭后迟到 Bar 不追溯修改决策。
- 无行情的候选不入选；持仓腿、对应期货、指数缺失或过期会等待恢复或失败闭闸。
- LAST 模式没有成交就不造 Bar；显式 `--ctp-price-type mid` 使用 MID 报价 Bar，
  volume=0。MID 不表示已成交，也不证明盘口可以按中间价成交。
- Replay 的期权、期货、指数必须来自相同估值日；不能混用旧 CTP 价格和今天的指数。
  日内时点仍匹配原始行情时刻，接收时刻到13:58并不会自动触发历史13:58信号。
- 资料按估值日检查上市、最后交易日；原生 CTP 还检查合约在实际 TD 日仍可交易。
  资料内容版本变化或日期跨日要求停止并重新装配，不会删除旧持仓腿继续运行。

### 首次验收命令

在项目 `pro` 目录、具备原生 Bomber 的远程 Python 环境先执行无网络回归：

```bash
python -u -m tests.run_cross_section_acceptance \
  --only option-vega --only option-vega-live \
  --only live-common --only attribution --only assembly
```

`option-vega-live` 使用真实框架与 SDK 行情实体，但网络和成交仍是夹具，
不连接柜台。本地没有原生 SDK 时可分别执行：

```bash
python -m unittest tests.test_demo_option_vega tests.test_option_vega_live
python -m tests.run_option_vega_offline
```

第二条只替换 SDK 行情实体和未使用的 SDK 桥，实际 Runner、归属、风控、
执行状态机与 JSON 持久化保持原代码。它不能替代目标环境的原生实体检查。

在线首轮选 Recording。下面四个变量必须使用已核验的值，不能从示例猜测：

```bash
read -r -p "实际TD/参考会话交易日 YYYYMMDD: " TD_DAY
read -r -p "Replay原始行情估值日 YYYYMMDD: " MD_DAY
read -r -p "真实交易日历 CSV/Feather 路径: " CALENDAR_FILE
read -r -p "同一Replay估值日的指数已收盘分钟流表: " INDEX_TABLE

export CTP_MD_ADDRESS=tcp://182.254.243.31:40011
export CTP_PRODUCTION_MODE=true

python -u -m demos.08_option_vega.run_live \
  --connect --mode recording \
  --market-source ctp --index-source dolphindb \
  --simnow-environment replay --replay-md-trading-day "$MD_DAY" \
  --expected-trading-day "$TD_DAY" \
  --reference-source dolphindb --reference-database dfs://bomber_daily \
  --calendar "$CALENDAR_FILE" --index-bar-table "$INDEX_TABLE" \
  --index-bar-timestamp end --ctp-price-type last \
  --option-product MO --future-product IM --index-code 000852 \
  --nav 40000000 --vega-budget 1.25 \
  --max-option-lots 1 --max-total-option-lots 2 --max-future-lots 1 \
  --max-notional 2000000 --seconds 900
```

连接凭据由项目 `.env` 读取，DolphinDB 使用 `DDB_HOST`、`DDB_PORT`、
`DDB_USERNAME`、`DDB_PASSWORD`，流订阅可另配 `DDB_STREAMING_PORT`。
期权基础表默认 `opt_basic`，可用 `DDB_OPT_BASIC_TABLE` 指定。
指数表也可配置 `DDB_INDEX_BAR_TABLE`；整条链走 DolphinDB 时再配置
`--market-source dolphindb --market-bar-table <已收盘期权/期货流表>`。

Recording通过后，改为 `--mode model` 并加：

```bash
--state-file "demos/08_option_vega/results/model-vega-$(date +%Y%m%d-%H%M%S).json" \
--require-fills
```

Model 中的 `--require-fills` 要求本次有策略归属成交；运行时段必须覆盖原始行情
的选约时点。省略该参数的通过结果只验证行情帧，不等同于信号或成交通过。
原生 SimNow 则改为 `--mode simnow --enable-orders --confirm-simnow`，提供独立
`--state-file`、实际 `CTP_TD_ADDRESS`，再按相同步骤核对真实订单、成交及柜台仓位。

### 恢复和停机

- `checkpoint.py` 保存期权／期货目标、当日选约记录和策略计数。
- 公共状态管理器保存目标版本、归属、订单状态、成交去重键和执行政策。
- `--resume --state-file <已有原文件>` 保留已成交仓位；不重新生成状态文件名。
- 恢复不直接续发保存的未完成开仓目标。首次新鲜行情帧以归属成交仓位重建
  目标并重算对冲；当天已选约不会重复开仓，旧价格和 IV 缓存不恢复为新鲜数据。
- 合约集合、条款指纹、日历、时间模式、行情来源、参数和执行政策必须一致。
  当前只支持固定资料会话、同一执行交易日恢复；失败恢复不覆盖原文件。
- 停机撤销自己的活动订单并保存已成交仓位，**默认不自动平仓**。
  仅显式 `--flatten-day YYYYMMDD --flatten-slot HH:MM` 才执行策略主动清仓。
- 状态文件使用独立文件锁；原生 CTP 另使用公共账户锁。

回测示例现在也支持模块方式：

```bash
python -u -m demos.08_option_vega.run_backtest \
  --option-product MO --future-product IM --index-code 000852 \
  --start-day 2026-09-01 --end-day 2026-09-11 \
  --bar-timestamp end --log-level WARNING --no-tearsheet
```

当前验收范围：已有策略/资料测试、时间与恢复回归、SDK替身下的实际执行装配。
真实 DolphinDB 指数流、CTP期权接收、CFFEX原生报单及完整原生回测仍需远程验收。
回测的权利金比例保证金近似、Model的同步参考价成交、缺少到期结算等限制保持显式；
本次没有增加期权卖方真实保证金模型或行权／到期结算模型。
