# 策略示例与统一框架适配分析

> 分析日期：2026-09-20  
> 分析范围：`temp/example01`～`temp/example10`、`temp/example101`、`temp/strategies/cta_by_pos`、`temp/strategies/cta_rn`  
> 对照框架：`/Users/kerry/work/bomber1/pro/strategy`  
> 本文只做架构分析和实施规划，不包含代码实现。

## 1. 结论摘要

这些目录并不是 12 个互不相关的策略，而是几条逐步演进的能力链：

1. `example01`、`example02`：单标的、单周期、指标信号；
2. `example03`、`example04`：多标的同步、截面排序、组合目标；
3. `example05`：期货和期权联动、连续合约信号、安全换月；
4. `example06`、`example07`：真实合约行情、因果数据查询、角色合约映射、执行管理器；
5. `example08`～`example10`：跨品种派生信号、统一 DataHub、下一根 Bar 执行、审计；
6. `example101`、`strategies/cta_by_pos` 和 `strategies/cta_rn`：外部目标持仓文件驱动的定时组合调度。

当前统一框架已经正确解决了第一层问题：

- 策略通过 `data_key` 消费行情，不感知文件、CTP、DolphinDB 或 Binance；
- 策略通过 `target_key` 输出目标，不感知 NT、Bomber 或 vn.py；
- `TargetPortfolio` 表达“最终想持有什么”，而不是直接表达买卖订单；
- 行情标的和交易标的可以不同。

但是当前实现仍是最小骨架。它足以迁移 `example01`、`example02`，也可以承载
`example03`、`example04` 的信号部分；若要可靠承载 `example05`～`example10`，还缺少：

- 多路 Bar 同步和数据完整性判定；
- 带 `event_ns / available_ns` 的因果 DataHub；
- 动态角色合约解析，不能只靠静态 `ExecutionRoute`；
- 目标状态保存、版本去重、组合净额和目标范围语义；
- 真实持仓、活动订单、成交回报的闭环；
- 换月、期权腿、下一根 Bar 等执行策略；
- 交易日历、会话时钟、定时目标和重启恢复；
- 回测撮合、手续费、滑点和审计。

因此不建议把现有示例直接改写成新的 `StrategyTemplate` 后立刻接实盘。正确顺序是先补齐通用端口和执行闭环，再逐类迁移策略。

## 2. 应坚持的职责边界

```text
标准行情源
  └─ QuoteTick / TradeTick / Bar / CustomBar
          │
          ▼
行情同步器 / 因果 DataHub / 合约主数据
          │
          ▼
StrategyTemplate
  └─ 只计算逻辑目标 TargetPortfolio
          │
          ▼
目标仓位管理 / 多策略组合管理
          │
          ▼
动态目标解析 / 换月与组合执行策略
          │
          ▼
ExecutionClient
  └─ NT / Bomber / vn.py / 模拟撮合
          │
          ▼
委托、成交、持仓和资金回报
```

关键原则如下：

- 指标、截面排名、价差和方向判断属于策略层；
- CSV、Feather、CTP 结构体和 DolphinDB 行解析属于行情层；
- 主力、次主力、近月、远月等角色映射属于参考数据层；
- 复权价格只能用于信号，不能作为可成交价格；
- 目标仓位到差额订单的换算属于执行层，策略不应自己计算 `delta`；
- 撤旧单、平旧仓、等待成交、切换新合约属于执行状态机；
- 真实持仓必须来自撮合器或交易柜台，不能由策略自己推测；
- 多策略共同交易同一账户时，策略目标和账户净仓必须分开保存。

## 3. 各示例分析

### 3.1 example01：自定义单标的 EMA 策略

主要行为：

- 单个 Bar 输入；
- 快慢 EMA 产生多空方向；
- 策略直接读取 NT Portfolio；
- 策略自行平反向仓并提交市价单。

迁移方式：

- 一个 `DataBinding`：如 `primary_bar`；
- 一个逻辑 `target_key`：如 `position`；
- EMA 仍保留在策略内部；
- 信号为多时 `set_target("position", +1)`，为空时提交 `-1`；
- 删除策略中的订单创建、平仓和 Portfolio 判断。

当前框架适配度：高。它应作为第一个真正的离线 Bar 策略验收用例。

需要补充：指标预热状态和停止时是否清仓应成为明确配置，不能默认把 `on_stop`
等同于清仓。

### 3.2 example02：NT 内置 EMACross

主要行为与 `example01` 相同，区别只是使用 NT 自带的 `EMACross`。

迁移时不应继续直接继承或包装 NT 的订单型 `EMACross`，因为它的策略逻辑与 NT
Portfolio、OrderFactory 和订单接口耦合。应复用 EMA 算法和参数含义，重新实现为输出目标的轻量策略。

当前框架适配度：高。

### 3.3 example03：Tick 聚合后的五品种截面动量

主要行为：

- CTP Tick 转成五个品种的一分钟 Bar；
- 要求同一时刻五路 Bar 全部到齐；
- 使用最近窗口收益率排名；
- 做多最强、做空最弱，其余目标为零；
- 按目标名义金额换算合约张数。

迁移方式：

- Tick 到 Bar 的聚合属于行情层，不属于策略；
- 五个 Bar 分别绑定为五个 `data_key`；
- 策略在同步帧上计算一次排名；
- 一次 `set_targets()` 提交完整的五标的逻辑组合；
- 目标名义金额换算需要 `InstrumentMetaProvider` 提供乘数、精度和最小数量。

当前框架适配度：中。`TargetPortfolio` 已能表达组合目标，但缺少通用 Bar 同步器和合约元数据查询接口。

### 3.4 example04：Feather Bar 驱动的五品种截面动量

策略逻辑与 `example03` 相同，只是输入已是外部一分钟 Bar。

这正是数据源无感知设计的验证案例：`example03` 和 `example04` 最终应使用同一个策略类，差异只存在于 Runner 装配的 Feed 和 Bar 生成方式中。

当前框架适配度：中。

### 3.5 example05：期货/期权联动、连续合约和安全换月

主要行为：

- 连续合约或派生序列只产生信号；
- 订单落到真实期货或真实期权；
- IM 信号可只交易 MO 期权；
- RB 信号可同时交易期货和方向期权；
- 主力切换时先撤旧单、平旧仓、确认归零，再开新仓；
- 换月过程中保留最新信号。

该示例实际上包含三层逻辑：

1. 信号策略；
2. 动态合约选择，包括期货主力和期权合约；
3. 多腿执行与换月状态机。

只应把第 1 层迁入 `StrategyTemplate`。第 2 层应由合约解析器完成，第 3 层应由执行策略完成。

当前框架适配度：低。静态 `ExecutionRoute(target_key → InstrumentId)` 无法表达“今天路由到
IM2609，明天路由到 IM2610”，也无法表达根据到期日、行权价和方向动态选择期权。

### 3.6 example06：RB 四角色复权价策略

主要行为：

- 主力、次主力、近月、远月都是动态角色；
- 角色表使用前一交易日信息，防止未来数据；
- 复权角色价格用于信号，真实主力合约用于交易；
- 当前主力切换时执行安全换月。

值得保留的设计：

- 策略按决策时点执行 as-of 查询；
- 角色表和复权值不伪装成可交易 Bar；
- 数据缺失或过旧时跳过决策；
- 信号标的和执行标的严格分离。

需要调整的地方：策略仍直接读取 Portfolio、活动订单并创建订单。这些应移到执行层。

当前框架适配度：低到中。策略输出目标的能力已具备，但 DataHub 和动态路由尚未实现。

### 3.7 example07：通用 DataHub 与 ContractExecutionManager

这是所有示例中最值得吸收的架构版本。

值得吸收：

- `TimedValue` 同时保存 `event_ns` 和 `available_ns`；
- `SnapshotQuery` 固定 `as_of_ns`，阻止未来数据；
- `ContractUniverse` 描述角色到真实合约的动态映射；
- `TargetIntent` 表达产品级目标，而不是订单；
- `ContractExecutionManager` 把换月从信号策略中分离；
- 每个品种拥有独立的换月状态。

需要调整：

- DataHub 快照当前同时包含参考数据、持仓和活动订单，边界仍然偏宽；
- 参考/外部数据应由 DataHub 提供，持仓和订单应由 ExecutionStateProvider 提供；
- ExecutionManager 不应持有 NT Strategy 实例，而应面向统一执行端口；
- `TargetIntent` 的能力应并入或转换为现有 `TargetPortfolio`，避免两套目标协议。

当前框架适配度：它是补齐当前框架缺口的主要参考，而不是可直接复制的最终实现。

### 3.8 example08：黑色系跨品种复权收益率策略

主要行为：

- JM、I、RB 三个品种的次主力复权价格共同形成信号；
- 三路数据必须同步；
- 只交易 RB 当前真实主力；
- RB 主力变化时安全换月；
- 有两层滚动窗口和预热要求。

迁移后策略应只负责：

- 消费同步后的 JM/I/RB 逻辑数据；
- 维护收益率窗口；
- 输出逻辑目标，例如 `rb_main = -1/0/+1`。

DataHub 负责角色与复权，目标解析器负责把 `rb_main` 动态解析到真实合约，执行协调器负责换月。

当前进展（第五类 V1～V4）：已新增 `datahub/sector_roles.py` 的通用
品种/角色因果快照、`examples/black_sector/sector_signal.py` 的纯
30/15 收益率窗口、`sector_strategy.py` 的真实 Bar 同步与可选额外目标提交延迟，
并通过既有动态路由、换月协调和模拟 Backend 装配正式回测入口。默认在信号
Bar 提交目标，Historical Runtime 保证原生撮合最早发生在后续行情事件；
额外等待一根后再提交并不等同于在那根 Bar 成交。
本地 V1/V2 已通过；V3 原生 Bar 和 V4 真实样本撮合须在 `uv-nautilus`
环境验收。外部 `pcr_cumfactor` 与当前按同日旧/新收盘价计算的三条次主力
因子仍需核对生产口径，不能直接宣布完全迁移结束。

边界修正：`datahub/sector_roles.py` 现已改为通用品种/角色映射；
JM/I/RB、`secondary`、RB `main` 和 `rb_main` 只由第五类配置选择。

### 3.9 example09：策略自描述数据需求和 DataHub

主要进步：

- 策略配置集中声明真实行情和外部数据需求；
- DataHub 统一加载合约结构、复权因子和基础资料；
- 强调真实 Bar、派生信号和可交易合约的边界；
- 强调查询时点、夜盘交易日和防重复触发。

不建议直接沿用的设计：

- Strategy 自己准备文件数据并把自己安装进 BacktestEngine；
- 策略模板直接依赖回测引擎和本地路径；
- 回测环境装配与信号算法放在同一个对象中。

在当前框架中，数据需求应由独立的策略规格或装配配置声明，由 Runner/HistoryService
完成加载。策略仍只接收逻辑数据和 DataHub 视图。

当前框架适配度：概念可复用，但实现边界需要重新切分。

### 3.10 example10：下一根 Bar 执行与成交审计

相对 `example09` 新增：

- 信号产生后延迟若干 Bar 执行；
- 同一标的尚未执行的目标可被新目标覆盖；
- 成交、平仓、手续费和 PnL 审计；
- 对订单拒绝、撤单和仓位事件作处理。

这里要区分两个概念：

- “延迟一根 Bar 才提交目标”是信号调度策略；
- “使用下一根 Bar 的什么价格成交”是回测撮合模型。

前者可以由 `execution_policy="NEXT_BAR"` 或目标调度器表达；后者必须由模拟执行客户端实现，不能由策略伪造成交。

当前框架适配度：低。现有 `execution_policy` 只是字符串透传，还没有真实策略实现；也没有执行回报和审计事件。

### 3.11 example101/cta_by_pos：目标文件驱动

该目录包含两个重要场景：

- 每日固定时点提交一组目标；
- 日内多个精确时点提交期货和期权目标，且不同标的周期可能不同。

它与当前框架天然匹配：CSV 中的每一组记录本身就是 `TargetPortfolio`，策略不需要计算买卖差额。

迁移后需要：

- `TargetScheduleProvider` 读取并校验目标文件；
- 交易日历和统一时钟触发指定时点；
- 一个目标时点一次性提交完整组合；
- 明确目标缺失是“保持原目标”还是“目标归零”；
- 到期信息来自合约主数据，不能通过代码格式猜第三个周五；
- 日终检查未触发目标并产生告警。

`sendTargetVol(..., isLast)` 是 Bomber 传输层的批次协议，不应该进入通用策略。统一框架中的一次
`set_targets()` 就应天然代表一个原子的目标版本，由 Bomber ExecutionClient 负责转换为自己的批次调用。

当前框架适配度：中到高。最大的缺口是交易日历、时钟和目标全量/增量语义。

### 3.12 temp/strategies/cta_by_pos：生产化尝试

与 `example101/cta_by_pos` 相比，它改用 `onBatchBar(isLast=True)` 一次提交一组目标，并尝试跳过到期合约。

值得保留的是“同一时点按完整组合提交”，需要删除的是基于代码字符串推断到期日。不同交易所、期货和期权的最后交易日规则不同，必须查询权威 Instrument 主数据。

此外，Bar 到达不应是定时目标唯一的时钟来源。某个标的缺 Bar 时，整个目标批次是否等待、降级或继续，需要由同步/调度策略显式配置。

### 3.13 temp/strategies/cta_rn：日内多时点混合资产目标调度

`cta_rn` 不是一个新的行情指标策略，而是 `cta_by_pos` 的日内多时点、混合资产版本：

- `positions.csv` 可以在同一交易日包含多个精确到分钟的目标时点；
- 同一目标组合可以包含期货、期权和指数；
- 期货订阅 M1，期权和指数订阅 M5；
- 使用 Bomber 合约信息接口判断资产类型、品种和乘数；
- 每个时点通过多次 `sendTargetVol` 加最后一条 `isLast=True` 发送完整批次；
- 使用 `(date, HH:MM)` 防止同一目标时点重复提交；
- 计算按品种汇总的目标名义金额，用于监控或风控。

它在统一框架中的正确定位应是：

```text
TargetScheduleProvider
  → ScheduledTargetStrategy
  → TargetPortfolio(REPLACE)
  → PortfolioCoordinator / Risk
  → ExecutionClient
```

其中 CSV 加载和时点索引属于 `TargetScheduleProvider`；定时触发属于 Scheduler；
`CtaRnStrategy` 只需要在目标时点提交一份完整的 `TargetPortfolio`。Bomber 的 `isLast`
仍然只应存在于 Bomber ExecutionClient 内部。

当前实现还存在以下风险，迁移时不能原样保留：

1. **异周期批次边界不明确**：M1 期货与 M5 期权/指数混合订阅时，`onBatchBar(isLast=True)`
   是否代表目标组合所需的所有数据已经到齐并不明确。目标调度不能依赖这一隐式假设。
2. **名义金额使用了错误价格**：循环中所有目标都使用本次回调的 `barData.closePrice`，该价格
   只属于触发回调的一个标的，不能用于给其他期货、期权和指数估值。应从同一时点的价格快照
   按标的分别取价，并记录价格新鲜度。
3. **到期判断不通用**：`代码前两位 + YYMM + 第三个周五` 只覆盖极少数形式，对期权、商品
   期货和不同交易所规则均不可靠；代码还使用 UTC 的 15:00 构造到期时刻。必须改用权威
   Instrument 主数据中的最后交易日和时区。
4. **防重状态提交过早**：代码在实际发送目标前就写入 `_sent_slots`。如果过滤、发送或客户端
   调用中途失败，该时点会被认为已经成功。防重记录应和目标持久化/客户端接受确认绑定。
5. **14:55 清理可能导致重复**：发送完 14:55 目标后立即删除当天防重记录；若随后收到重复
   14:55 批次，可能再次提交。应在交易日切换或日终完成事件中清理。
6. **CSV 校验不足**：当前未严格检查必需列、重复的 `(timestamp, instrument_id)`、空目标、
   非法数量和时区。应复用 `example101/option_test` 中更严格的校验思路。
7. **运行器与策略接口不一致**：`run.py` 读取 `STRATEGY_NAME`、`MULTIPLIER_MAP`、
   `EXTRA_INDEX_CONTRACTS` 和 `EXTRA_FUTURES_CONTRACTS` 类属性，但当前策略文件没有定义这些
   属性。除非基类另有可靠默认值，否则目录本身不能视为完整自洽的可运行包。
8. **无失败和遗漏审计**：`onDailyClose` 为空，无法发现计划目标未触发、只发送了一部分、被到期
   过滤或被客户端拒绝的情况。

与 `cta_by_pos` 相比，`cta_rn` 进一步证明框架需要的是通用“目标计划 + 调度 + 原子组合提交”，
而不是继续扩充基于 Bar 回调的专用策略模板。

## 4. 当前框架能力对照

| 能力 | 当前状态 | 能承载的示例 | 说明 |
|---|---|---|---|
| 标准行情事件 | 已有 | 全部 | `QuoteTick/TradeTick/Bar/CustomBar` |
| 行情源无感知 | 已有 | 全部 | `DataBinding` 已建立正确边界 |
| 交易客户端无感知 | 协议已有 | 全部 | 目前主要是记录型客户端，真实适配器待实现 |
| 逻辑目标组合 | 已有基础 | 全部 | `TargetPortfolio` 可表达多目标 |
| 静态执行路由 | 已有 | 01～04 | `target_key → client + instrument` |
| 动态角色路由 | 缺失 | 05～10 | 主力、次主力、期权选择无法静态配置 |
| 真实持仓查询 | 只有协议 | 01～10 | 默认实现永远返回 0，不能用于生产 |
| 目标版本去重/保存 | 缺失 | 全部 | revision 生成了，但下游未拒绝旧版本 |
| 多策略组合净额 | 缺失 | 03～10 | 同账户同合约目标尚无归属和净额规则 |
| Bar 同步器 | 缺失 | 03、04、08～10 | 目前只能由每个策略自行拼接 |
| 因果 DataHub | 缺失 | 06～10 | 需要 event/available/as-of/staleness |
| 合约主数据 | 不完整 | 03、05～10、CTA | 乘数、到期、角色、期权属性需要统一来源 |
| 执行状态机 | 缺失 | 05～10 | 换月、多腿、撤单和等待成交 |
| 执行回报回调 | 缺失 | 05、06、08～10 | 策略/协调器收不到订单与成交状态 |
| 定时器/交易日历 | 缺失 | 05、CTA、cta_rn | 夜盘、交易日、多个定点目标无法可靠表达 |
| 混合周期目标批次 | 缺失 | cta_rn | M1/M5 标的不能依赖隐式 `isLast` 组成原子批次 |
| 组合估值快照 | 缺失 | 03、05、08～10、cta_rn | 每个目标腿需要自己的价格、乘数和新鲜度 |
| 历史预热 | 缺失 | 01～10 | 指标窗口和外部数据启动拼接未统一 |
| 回测撮合 | 缺失 | 全部 | FileReplayFeed 只是事件播放器 |
| 手续费/滑点/PnL | 缺失 | 全部 | 不能用记录型执行端评价策略绩效 |
| 状态持久化/恢复 | 缺失 | 05～10 | 重启后目标、换月阶段和订单归属会丢失 |

## 5. 必须先明确的协议语义

### 5.1 TargetPortfolio 是全量快照还是局部更新

这是当前最重要的未决问题。

例如：

```text
revision 1: {rb: +1, jm: -1}
revision 2: {rb:  0}
```

revision 2 到底表示：

- 只把 RB 改成 0，JM 保持 -1；还是
- 这是完整组合，未出现的 JM 也应归零？

建议显式增加目标模式：

- `REPLACE`：该目标作用域内的完整快照，缺失目标归零；
- `PATCH`：只更新出现的目标，其他保持不变。

截面策略、`cta_by_pos` 和 `cta_rn` 默认应使用 `REPLACE`；单腿事件型策略可选择 `PATCH`。

### 5.2 策略持仓与账户持仓

`position(strategy_id, target_key)` 不能简单等于账户净仓。多个策略可能共同交易同一个真实合约：

```text
策略A目标 rb +2
策略B目标 rb -1
账户净目标 rb +1
```

需要同时保存：

- 策略目标；
- 策略归属持仓；
- 账户真实持仓；
- 账户净目标；
- 未完成订单产生的在途数量。

否则策略会互相平仓，或者在重启后重复下单。

### 5.3 静态路由与动态目标解析

现有 `ExecutionRoute` 适合 `trade_leg → rb2610.SHFE`，但不适合 `rb_main → 当日主力`。

建议保留静态路由，同时新增动态解析端口：

```text
logical target
  → TargetResolver(as_of_ns, target_key)
  → one or more concrete legs
  → ExecutionPlan
```

解析结果需要记录依据和版本，例如角色表日期、主力合约、期权到期日和行权价，以便审计和重放。

### 5.4 execution_policy 不能只是字符串

当前字段可以保存 `DIRECT`、`NEXT_BAR`、`ROLL_SAFE`、`SPREAD` 等名字，但下游尚未实现对应行为。

至少需要一个执行策略注册表，把策略名映射到真正的协调器，并规定：

- 是否允许拆腿；
- 是否先平后开；
- 是否等待撤单确认；
- 超时、拒单和部分成交如何处理；
- 新 revision 如何覆盖旧计划；
- deadline 到期后如何处理。

## 6. 建议新增的通用模块

### 6.1 BarSynchronizer / StrategyFrameBuilder

职责：

- 按逻辑 `data_key` 收集同一决策时点的数据；
- 支持 `EXACT`、`ASOF`、`WINDOW` 三种对齐；
- 检查缺失、重复、过旧和乱序；
- 形成一次不可变 `StrategyFrame`；
- 明确超时后是跳过、使用旧值还是进入 DEGRADED。

它用于 `example03`、`example04`、`example08`～`example10`，不应由每个策略重复实现。

### 6.2 StrategyDataHub

吸收 `example07` 的优点，提供：

- `event_ns`：数据代表的业务时间；
- `available_ns`：数据最早可被策略看到的时间；
- `as_of_ns`：本次决策时点；
- 最大允许年龄和质量标记；
- 合约角色、复权因子、现货、库存等非 Bar 数据。

DataHub 不应包含持仓和活动订单；这两类状态属于执行域。

### 6.3 InstrumentMetaProvider / ContractUniverseProvider

职责：

- 合约乘数、价格/数量精度、最小变动；
- 上市和到期时间；
- 期货/期权属性；
- 品种、交易所、币种；
- 动态角色到真实合约的映射；
- 信息生效时间和可见时间。

它替代字符串猜测到期日和策略硬编码合约。

### 6.4 TargetStore / PortfolioCoordinator

职责：

- 保存每个策略最新目标和 revision；
- 拒绝重复、倒序和过期目标；
- 处理 `REPLACE/PATCH`；
- 多策略目标汇总和净额；
- 资金、限仓和风险裁剪；
- 生成账户级目标并保留策略归属。

### 6.5 DynamicTargetResolver

职责：

- 把 `rb_main`、`mo_call` 等逻辑腿解析到真实合约；
- 一个逻辑目标可以解析成一个或多个真实腿；
- 解析严格使用 `as_of_ns` 当时可见的合约信息；
- 保留解析版本，支持审计和重放。

### 6.6 ExecutionCoordinator

职责：

- 读取账户真实持仓、活动订单和在途成交；
- 将账户目标转换成可执行差额；
- 实现 DIRECT、NEXT_BAR、ROLL_SAFE、SPREAD 等策略；
- 管理撤单、部分成交、拒单、超时和重试；
- 国内期货开平今昨最终交给 Bomber/CTP 执行适配层处理；
- 状态可持久化并能在重连后恢复。

`example07/ContractExecutionManager` 可作为换月算法原型，但需要去除对 NT Strategy 的直接依赖。

### 6.7 ExecutionStateProvider 与统一回报

至少需要：

- 真实持仓快照；
- 活动订单；
- 委托接受、拒绝、撤销；
- 部分成交和全部成交；
- 资金与保证金；
- 启动时对账完成状态。

Runner 只有在行情和交易状态都 READY 后才允许增加风险。

### 6.8 Clock / TradingCalendar / Scheduler

用于：

- `cta_by_pos` 的 09:40 或多个日内时间点；
- 夜盘交易日映射；
- 收盘前清仓；
- Bar 缺失时仍能触发检查和告警；
- 回测时钟与实时系统时钟使用同一接口。

## 7. 策略模板应该保持多薄

不建议为每一种示例建立层层继承的大模板。建议只保留一个小而稳定的
`StrategyTemplate`，通过组合获得其他能力：

```text
StrategyTemplate
  + Indicator/Window（策略内部纯计算）
  + StrategyFrame（同步后的输入）
  + StrategyDataView（因果外部数据）
  + StrategyContext（目标提交、策略持仓、时钟）
```

可提供少量辅助基类，但不能把执行端重新带回策略：

- `SingleInstrumentSignalStrategy`：单标的指标策略辅助；
- `SynchronizedPortfolioStrategy`：同步帧消费辅助；
- `ScheduledTargetStrategy`：目标文件或定时目标辅助。

这些辅助类只能减少样板代码，不能创建订单、读取交易客户端或决定开平今昨。

## 8. 推荐实施和验证顺序

保持与 Market 包相同的“小步迁移、小步验证”方式。

### 阶段 0：先冻结目标协议

验证：

- `REPLACE/PATCH` 语义；
- revision 去重和倒序拒绝；
- 一次多目标提交的原子性；
- 同一目标跨两个交易客户端的拆分规则；
- 策略目标、账户目标和真实持仓的区别。

### 阶段 1：迁移 example01/example02

使用 CTP 或 Binance 离线 Bar，RecordingExecutionClient：

- 同一策略切换 Feed 后目标序列一致；
- 策略不创建订单；
- EMA 预热和方向切换正确。

### 阶段 2：迁移 cta_by_pos 和 cta_rn

先只验证目标调度，不撮合：

- CSV 校验；
- 定点产生完整 `TargetPortfolio`；
- 缺 Bar/缺时钟时产生明确告警；
- 同一目标不会重复提交；
- 目标中的 0 和缺失字段语义正确。
- 同一天多个目标时点按交易日历准确触发；
- M1/M5 混合标的不依赖任意一根 Bar 的 `isLast`；
- 名义金额按每个标的各自的价格快照和乘数计算；
- 客户端未接受的目标不能被记录为已成功发送。

### 阶段 3：迁移 example03/example04

实现最小 BarSynchronizer：

- Tick 聚合 Bar 与 Feather Bar 驱动同一策略；
- 五路未到齐不决策；
- 同一时间只决策一次；
- 一次提交完整组合；
- 名义金额换算使用统一 Instrument 元数据。

### 阶段 4：实现因果 DataHub

先只做离线单元测试：

- `available_ns > as_of_ns` 的数据绝不能返回；
- 向前填充仍保留 `source_ns`；
- 过旧数据进入 unavailable/degraded；
- 夜盘交易日不能按自然日推断。

### 阶段 5：动态路由和安全换月

以 `example06/example07` 为验收案例：

- 信号目标为 `rb_main`；
- 角色变化后撤旧单、平旧仓、确认归零、开新仓；
- 换月期间新信号覆盖目标，但不提前开新仓；
- 状态重启后可恢复；
- 全流程只连接模拟执行端。

### 阶段 6：多腿和期权

以 `example05` 为验收案例：

- 动态期权选择；
- 多腿原子目标；
- 先后顺序、裸腿上限和失败补偿；
- 到期日来自主数据；
- 再接 Bomber 模拟或 SimNow。

### 阶段 7：跨品种策略

迁移 `example08`～`example10`：

- 同步帧；
- 复权因果性；
- 动态主力执行；
- NEXT_BAR 策略；
- 成交、费用和 PnL 审计。

### 阶段 8：实盘源和真实执行端

最后才组合：

- Binance 在线行情 + RecordingExecutionClient；
- CTP/DolphinDB 在线行情 + RecordingExecutionClient；
- 离线行情 + 模拟撮合；
- 在线行情 + Bomber/SimNow；
- 重连、对账、重复消息和故障恢复测试。

## 9. 不建议直接迁移的代码

- 示例中策略直接调用 `order_factory`、`submit_order`、`cancel_all_orders` 的部分；
- 策略直接读取账户 Portfolio 并自行计算下单差额的部分；
- `example09/10` 中策略自行准备文件并安装 BacktestEngine 的部分；
- 根据合约代码字符串猜到期日的逻辑；
- 用复权或连续合约价格参与撮合的逻辑；
- 每个策略各写一套多路 Bar 同步和换月状态机；
- 把 Bomber 的 `isLast` 批次传输细节暴露给策略；
- 把 Bar 到达当作唯一系统时钟。

## 10. 建议保留并抽象的代码思想

- `example03/04` 的完整组合目标和截面调仓；
- `example05` 的安全换月原则和“换月期间保留最新目标”；
- `example06` 的角色价格只用于信号、真实合约只用于交易；
- `example07` 的 `event_ns / available_ns / as_of_ns` 和执行管理器分层；
- `example08` 的跨品种同步与复权收益率计算；
- `example09` 的声明式数据需求，但声明应由装配层执行；
- `example10` 的延迟执行与审计需求，但成交由执行端决定；
- `cta_by_pos` 的完整目标快照和严格缺失检查；
- `cta_rn` 的多时点目标计划、混合资产分类和组合监控需求。

## 11. 开始编码前需要确认的决策

以下问题不确认，会导致后面反复修改协议：

1. `TargetPortfolio` 默认采用 `REPLACE` 还是 `PATCH`？
2. 多策略交易同一真实合约时，采用账户净额执行还是策略独立虚拟仓位执行？
3. 动态目标键采用 `rb_main` 这类逻辑名称，还是把品种、角色定义成结构化对象？
4. 换月由统一 ExecutionCoordinator 承担，还是最终全部交给 Bomber？
5. 回测第一阶段使用 NT 撮合器，还是先实现最小 Bomber 模拟执行端？
6. `NEXT_BAR` 指下一根 Bar 到达时发单，还是强制按下一根 Bar 的开/收盘价模拟成交？
7. 目标文件缺少某个标的，是保持上一目标还是归零？
8. 目标调度以交易日历定时器为准，还是必须等待指定 Bar 到齐？
9. 合约角色和期权选择的权威数据来自本地文件、Bomber，还是独立 Instrument Provider？
10. 策略、目标管理器和执行协调器哪些状态必须落盘并支持重启恢复？
11. `cta_rn` 的目标时点是无条件定时触发，还是必须等待每个目标腿的新鲜价格快照？若等待，
    最大等待时间和缺失腿处理规则是什么？

## 12. 最终判断

当前框架的方向无需推倒重来，但需要从“Runner 直接把策略目标转发到 Client”演进为：

```text
Runner
  → StrategyTemplate
  → TargetStore / PortfolioCoordinator
  → DynamicTargetResolver
  → ExecutionCoordinator
  → ExecutionClient
```

同时在策略输入侧补充：

```text
MarketDataFeed
  → BarSynchronizer / StrategyFrameBuilder
  → StrategyDataHub
  → StrategyTemplate
```

完成这两条链以后，现有所有示例都可以落入同一框架，而不需要为 CTP、Binance、
DolphinDB、NT、Bomber 或 vn.py 分别维护策略版本。
