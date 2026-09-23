# 第一类策略：单标的EMA

具体策略位于`examples/single_ema/strategies`；框架目录`strategy`只存放通用Runtime、
Portfolio、Risk和Execution能力。CTP、Binance、离线与在线装配均复用同一个
`EmaCrossTargetStrategy`。

## 积木式架构：以`ema_bar_backtest.py`为装配清单

`ema_bar_backtest.py`是组合入口，不是另一份EMA策略。它把同一策略接到不同
Bar文件和模拟交易规则上；策略文件仍只在`examples/single_ema/strategies`。

```text
CSV / Feather
  → Parser（文件列和时区→标准Bar）
  → FileReplayFeed（按事件时间回放）
      ├─ NautilusMarketFeedAdapter → NautilusSimExecutionBackend（模拟时钟、撮合旧订单）
      └─ UnifiedStrategyRunner → EmaCrossTargetStrategy（EMA信号、目标仓位）
                                → TargetStore → PortfolioCoordinator
                                → ExecutionRoute → SimulationExecutionClient
                                → NetTargetOrderPlanner → PreTradeRiskManager
                                → NautilusSimExecutionBackend（接收新订单）
                                → ExecutionReport → PositionManager（仓位、在途量）
      └─ MarketReferencePriceStore（最新行情价/时间）→ PreTradeRiskManager
```

### 各块职责与替换边界

| 积木 | 本示例中的职责 | 替换时要保持的契约 |
|---|---|---|
| `Parser` | BN标准K线CSV使用`BinanceKlineParser`；RB Feather使用`MappedBarParser`指定列、时区及`XSGE→SHFE`别名 | 产出相同标准`Bar`，策略不解析源文件 |
| `FileReplayFeed` | 注册合约行情元数据，加载并按时间发布Bar | 实时可换Stream Feed；需要继续发布对应标准事件 |
| `InstrumentMeta` | 为Feed提供行情价格/数量精度、乘数和交易所 | 与可交易`instrument`一致，但不负责撮合 |
| `Profile`＋`instrument` | 定义模拟Venue账户、保证金、手续费及具体合约的价格步长/乘数/有效期 | 换交易场所规则时在此配置，不写进EMA策略 |
| `DataBinding` | `primary_bar`→指定feed ID、合约、`BAR/1-MINUTE` | feed ID必须与`runner.add_data_feed()`一致；策略只识别逻辑`data_key` |
| `MarketStreamBinding`＋Adapter | 同一份Bar另送模拟Backend，推进时钟并撮合先前订单 | 只对Backend绑定可撮合的标准事件；不是第二份数据源 |
| `EmaCrossTargetStrategy` | 用原生EMA计算方向，输出逻辑`position`目标 | 不接触文件、账户客户端或原生订单API |
| `ExecutionRoute` | 把逻辑`position`映射为具体客户端和交易合约 | 改目标落地账户/标的时改装配，不改策略 |
| Runner中的目标/组合层 | `TargetStore`保存带版本的策略目标，`PortfolioCoordinator`汇总同账户的多策略净目标 | 单策略也走相同通道，为多策略组合留接口 |
| `PositionManager`＋Planner | 保存账户仓位/在途量，比较目标与有效仓位，生成开平订单意图 | 当前`NetTargetOrderPlanner`只适合净持仓；CTP今昨仓须换专用Planner |
| `MarketReferencePriceStore`＋Risk | 行情Observer维护参考价和时间；`RiskLimits`限制数量、预计仓位、名义金额及行情时效 | 价格存储本身不设置阈值；阈值按标的放在`RiskLimits` |
| `SimulationExecutionClient`＋Backend | 客户端串接Planner/Risk/Backend；Backend持有Nautilus模拟引擎并回报成交 | 正式实盘换执行客户端/Backend，不能把本示例当成真实下单 |
| `UnifiedHistoricalRuntime` | 统一启动、回放、结束和释放；保证模拟行情回调先于策略回调挂载 | `run_case()`仅装配，`runtime.run()`才真正执行 |

`Profile`、`InstrumentMeta`和`RiskLimits`容易混淆：前两者分别服务于**模拟交易规则**
和**行情解释**；`RiskLimits`是订单进入Backend之前额外的安全边界。例如RB报价3,000元、
一手乘数10时，风控名义金额约30,000元，而不是3,000元。

### 一根Bar触发时实际发生什么

1. Feed分发标准Bar。由于Historical Runtime先注册Adapter，同一事件先推进模拟引擎、
   撮合此前已有的订单，然后Runner才将它按`DataBinding`交给策略；不是读取两次文件。
2. Runner旁路更新风控参考价；EMA只消费`primary_bar`。完成预热后，快EMA不低于慢EMA
   则输出正目标，否则输出负目标；目标没有改变就不重复提交。
3. Runner保存策略目标、汇总账户净目标，按`ExecutionRoute`构造执行请求。
4. Planner比较目标和`实际仓位＋在途量`。例如当前`+1`、目标`-1`，先生成平多1手，
   再生成开空1手；Planner只产生订单意图，并不负责撮合。
5. Risk以最新参考价、RB乘数及阈值检查整批订单；获准后Backend接收订单。后续行情
   触发模拟撮合，`ExecutionReport`回到Client，更新实际仓位及在途量。

这个回调顺序旨在避免“当前Bar产生的信号立刻按当前Bar成交”；但该文件目前只断言有
目标、订单、成交且没有回报状态错误，**并未逐笔验证N→N+1或固定结果基准**。如需严格
验收，请运行下面已有的CTP/BN正式回测测试并检查其逐笔断言。

### 当前文件的运行选择与适用范围

当前`main()`先给`rb2704`赋值，随后又给`BTCUSDT`赋值；后一次覆盖前一次，所以直接
执行`python examples/single_ema/ema_bar_backtest.py`实际运行的是**Binance离线Bar**。
仅当显式选择RB的合约、路径和目标数量时，才会验证RB分支。该脚本的EMA配置固定为
`skip_single_price=True`，因此BN合法的单价Bar也会被跳过；若要验证BN完整连续消费，
请使用`ema_binance_backtest.py`中的相应配置与断言。`CtpFuturesBasicProfile`只提供基础
净持仓模拟，不表示已覆盖CTP双向今昨仓、平今平昨及真实柜台行为。

此示例只接`SimulationExecutionClient`，**不会向真实交易所下单**。切换到实盘时必须另外
装配实时Feed、实盘执行客户端/Backend及其对账和授权边界；EMA策略类可以保持不变。

所有命令在`pro`目录执行，并先设置：

```bash
export PYTHONPATH=.
```

## 1. 策略逻辑和内存正式回测

```bash
python tests/strategies/single_ema/run_test.py
python tests/run_unified_historical_runtime.py --stage 3
```

前者验证EMA参数、预热、过滤、多空目标和防重；后者验证统一Runner、Planner、Risk、
Nautilus模拟Backend、成交回报和N→N+1撮合顺序。

## 2. CTP离线Bar和Tick

```bash
python examples/single_ema/ema_ctp_backtest.py --source bar
python examples/single_ema/ema_ctp_backtest.py --source tick
python examples/single_ema/ema_ctp_backtest.py --source all
```

`all`会为Bar和Tick分别启动子进程，避免Nautilus进程级日志器重复初始化。

- Bar：`rb2704.SHFE` Feather，固定基准为351条、7根有效EMA Bar、1张订单和1笔成交；
- Tick：`rb2609.SHFE` CSV，由Nautilus内部聚合1分钟Bar，固定基准为42,530条事件、
  112根有效EMA Bar、45张订单和45笔成交；
- Tick内部聚合Bar与下一份Tick可能共享CTP毫秒时间戳；回测会拒绝早于信号的成交，
  并报告同时间戳成交数量。仅凭时间戳不能证明是否同一行情事件，不能把该项解读为
  Tick链已完成严格N→N+1事件顺序验收；
- Tick文件没有ActionDay，示例将夜盘自然日显式设为`20260727`；换文件时必须同步修改；
- 两份数据不是同一合约，不能合并为一条持仓时间线。

CTP Bar/Tick现在共用RB风控限制：单笔最多2手、账户绝对仓位最多1手、单笔及
预计持仓名义金额各不超过50,000元、参考行情最多120秒；名义金额按RB乘数10计算。
价格Observer会从标准Bar或原始Quote/Trade更新参考价。Tick内部聚合Bar可能晚于
信号时间才发出，因此模拟执行以当时已处理到的行情时间评估参考价时效，信号
时间仍单独保留用于前视审计。配置位于`ema_ctp_backtest.py`的`_RB_RISK_LIMITS`。
这组阈值用于验证机制，不是建议的实盘风控参数。

可先运行无行情文件依赖的风控测试：

```bash
python tests/run_risk_manager.py --stage 4
```

## 3. Binance离线Bar

```bash
python examples/single_ema/ema_binance_backtest.py
```

默认读取USDT永续`BTCUSDT`的一天1分钟CSV，共1,440根Bar，使用
`BinanceUsdtFuturesProfile`和统一模拟执行链。测试会检查：

- EMA连续消费全部Bar；
- 目标、订单、原生成交和统一执行回报完整；
- 账户最终仓位等于策略最终目标；
- 在途数量归零；
- 所有成交时间严格晚于对应信号时间，防止同Bar前视。

## 4. 在线转换与安全Recording测试

先运行完全离线的三项测试：

```bash
python tests/run_single_ema_online.py --stage offline
```

它依次验证：

1. Binance只把`x=true`的已收盘Kline转换为标准Bar；
2. Binance Kline驱动EMA并生成Recording请求；
3. CTP TradeTick通过通用`TradeTickBarFeed`形成1分钟Bar，再驱动同一个EMA。

### Binance真实在线行情，不下单

Binance EMA从零开始预热，默认2/3周期至少需要等待3根已收盘分钟Bar：

```bash
BN_EMA_TIMEOUT=300 python tests/run_single_ema_online.py --stage 4
```

常用环境变量：`BN_SYMBOL`、`BN_MARKET_TYPE`、`BN_WS_BASE_URL`、`EMA_FAST`、
`EMA_SLOW`、`EMA_BN_QUANTITY`和`BN_EMA_TIMEOUT`。

### CTP真实在线行情，不下单

沿用`.env`中的`CTP_MD_ADDRESS`、`CTP_BROKER_ID`、`CTP_ACCOUNT_ID`、
`CTP_PASSWORD`、`CTP_SYMBOL`等配置：

```bash
CTP_EMA_TIMEOUT=300 python tests/run_single_ema_online.py --stage 5
```

CTP原生行情只提供Tick，因此先由`TradeTickBarFeed`聚合已收盘1分钟Bar。没有成交的
分钟不会补零；最后尚未结束的分钟在停机时不会发出。

同一链路现另有正式示例入口，仍使用相同`EmaCrossTargetStrategy`，但交易端固定为
Recording，不会连接CTP TraderApi或发送订单：

```bash
python examples/single_ema/ema_ctp_live.py --symbol rb2610 --timeout 300
```

该示例仅在行情时段能观察到新Bar和目标；合约、最小变动价位及乘数应按实际合约设置。

## 5. Binance统一在线示例

默认只接收`market.stream.bn`的Kline并记录目标，不下单：

```bash
python examples/single_ema/ema_binance_live.py --environment live
```

限制运行时间可使用：

```bash
python examples/single_ema/ema_binance_live.py --environment live --timeout 360
```

`--enable-orders`目前在DEMO与LIVE均会被拒绝：示例尚未接入受控客户端的
资金、全市场活动订单恢复与人工授权链。请先用默认Recording模式验证行情和信号；
`binance_demo_readonly.py`可单独用于显式连接DEMO账户的只读对账，不会下单。

### 四模式验收状态

| 行情/执行 | CTP | Binance |
|---|---|---|
| 离线回测 | Bar/Tick正式模拟入口已具备；真实样本回归需按各脚本断言复测 | 期货Bar正式模拟入口已具备；真实样本回归需复测 |
| 在线Recording | `ema_ctp_live.py`已装配，待交易时段验证 | `ema_binance_live.py`默认模式已装配，待实流验证 |
| 在线受控交易 | 原生TdApi无网络测试至P4-CTP9，真实SimNow只读/订单联调未完成 | DEMO只读组件已具备，EMA示例受控执行装配未完成；旧下单标志已禁用 |

四种模式始终复用`EmaCrossTargetStrategy`；“装配入口存在”不等于真实账户
已经验收，更不意味着当前允许真实下单。

新示例的结构是：

```text
BNWSStreamDataFeed
  → UnifiedStrategyRunner
  → EmaCrossTargetStrategy
  → TargetStore / PortfolioCoordinator
  → RecordingExecutionClient

计划中的受控下单链路（**尚未接入本示例**）：

BNWSStreamDataFeed
  → UnifiedStrategyRunner
  → EmaCrossTargetStrategy
  → ControlledLiveExecutionClient
  → Planner → Risk
  → NautilusLiveExecutionBackend
  → Binance Exec Client
```

不再使用旧`NautilusStrategyBridge`作为策略执行路径。

## 安全边界

- 在线测试阶段4/5以及Binance示例默认模式都不会下单；
- `--enable-orders`会在DEMO环境发送模拟订单；
- LIVE下单还要求`--confirm-live`；
- 示例默认设置订单数量、绝对仓位、订单名义金额、持仓名义金额和行情时效限制；
- EMA示例只用于验证架构闭环，不代表具有可投入资金的收益能力。
