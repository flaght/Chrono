# 单品种跨期价差策略

Live入口已实现，见[第4节](#4-live版本)。当前固定交易日和固定双腿；原生离线及在线验收待Linux运行。

## 1. 使用方法

### 启动命令

从 pro 根目录、在已安装运行依赖的环境中执行，先设置 `export PYTHONPATH=.`。入口自动读取已有 `.env` 配置；没有配置数据路径时，可通过可选的 `--data-root` 或各文件路径参数指定。示例日期需与实际数据匹配。

**完整执行（显式设置常用参数）：**

```bash
python demos/03_calendar_spread/run_backtest.py \
  --product RB \
  --start-day 2026-01-05 --end-day 2026-05-29 \
  --leg1-role secondary --leg2-role far \
  --quantity 1 --lookback 120 --entry-z 2 --exit-z 0.5 \
  --bar-timestamp end --log-level WARNING --no-tearsheet
```

**默认执行（仅必填参数）：**

```bash
python demos/03_calendar_spread/run_backtest.py \
  --product RB \
  --start-day 2026-01-05 --end-day 2026-05-29
```

默认执行采用参数表中的默认值及已有数据路径配置；例如默认启用绩效图，不要求必须产生成交。

### 参数说明

“必选”表示启动时必须传入；“可选”表示有默认值或可由已有配置解析。数据路径虽然是可选参数，但运行前必须能解析到有效文件。

| 参数 | 必选 / 可选 | 含义与默认值 |
| --- | --- | --- |
| `--product` | 必选 | 选择一个品种，两腿均属于该品种；例如 RB、I、JM、TA |
| `--start-day / --end-day` | 必选 | 请求日期闭区间 |
| `--leg1-role / --leg2-role` | 可选 | 默认 secondary / far，对应角色表 second / far；也支持 main / secondary、main / far，不使用 recent |
| `--quantity` | 可选 | 每腿目标手数默认 1，正整数；两腿方向相反 |
| `--lookback` | 可选 | 此前同步价差窗口根数，默认 120 根一分钟双腿 Bar；第 121 根开始可生成信号 |
| `--entry-z / --exit-z` | 可选 | 入场 / 出场阈值，默认 2 / 0.5 |
| `--rebalance-interval` | 可选 | 持仓时每隔若干根完整同步 Bar 重提目标；方向变化即时提交，默认间隔 5 |
| `--bar-timestamp` | 可选 | 源分钟标签 start / end，默认 end；start 在公共层统一移动到分钟结束 |
| `--missing-role-policy` | 可选 | 默认 raise，任一指定腿缺文件时报错；next-available 在请求角色范围内按当日文件可用性替换并记录。严格无前视研究保持 raise |
| `--contract-struct / --fut-basic / --bars-dir` | 可选 | 覆盖角色表、基础条款、分钟行情路径 |
| `--starting-balance` | 可选 | 初始资金默认 1000000 |
| `--commission-per-contract` | 可选 | 固定每手手续费默认 1 |
| `--margin-init / --margin-maint` | 可选 | 初始 / 维持保证金比例默认 0.10 / 0.08 |
| `--max-notional / --max-market-age-seconds` | 可选 | 名义金额上限默认 1000000，真实报价最大年龄默认 120 秒 |
| `--data-root` | 可选 | 可显式指定数据根目录，按 role/ 与 kline/ 布局推导路径；已配置路径时可省略 |
| `--report-dir` | 可选 | 输出根目录，默认本策略目录的 results/ |
| `--log-level` | 可选 | 框架日志等级，默认 WARNING |
| `--no-tearsheet` | 可选 | 指定时仅保存 CSV / JSON，不生成绩效图 |
| `--require-fills` | 可选 | 指定时要求至少一笔模拟成交，否则报错 |

### 数据要求

需要角色表 second / far 列及两腿真实行情；默认只选择次主力与 far。每天使用严格早于当日的最新角色记录，不读取复权因子。换约日需提供有敞口的旧双腿行情；带非空 available_ns 的延迟发布角色记录当前不支持。

| 数据项 | 要求 |
| --- | --- |
| 角色表 | 优先 fut_contract.feather，旧名 fut_contract_data.feather 仅在新名不存在时回退；两个同时存在须显式指定。需要 trade_date、code 及所选角色列 |
| 基础条款 | fut_basic.feather：symbol、code、exchangeCD、contMultNum、minChgPriceNum、listDate、lastTradeDate |
| 分钟行情 | 真实合约的 `<symbol>_YYYYMMDD.feather`，由公共层校验身份、时间、OHLC 和成交量 |

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

## 2. 策略详细说明

### 策略思路与统计口径

这是同品种、两个期限角色的价差均值回归策略。默认只使用**次主力（角色表 second）和 far（角色表 far）**，没有第三个 near 合约，也不使用 recent。内部统一命名为 `leg1 / leg2`：第一腿次主力，第二腿 far，实际合约由日级日程确定。策略没有根据合约到期天数重新排序，也没有拟合价差对冲系数，两腿按相等手数交易。

`CalendarSelection.leg1_symbol / leg2_symbol` 保存每日选出的两个真实合约；`CalendarSpreadConfig.leg1_role / leg2_role` 保存角色配置。默认价差明确为“次主力 close − far close”，不是额外选择一个近月合约。两腿参数都可省略，默认就是 secondary / far。

```text
S(t) = 第一腿真实 close - 第二腿真实 close
μ(t) = 此前 lookback 个同步价差的均值
σ(t) = 此前 lookback 个同步价差的总体标准差
z(t) = [S(t) - μ(t)] / σ(t)
```

当前价差计算完信号后才加入窗口；默认第 121 个完整双腿帧开始评价。方差为零时 z 为零。只有两腿同分钟到齐才推进窗口，缺失腿不补价；不使用复权因子。

| 标准分数条件 | 第一腿目标 | 第二腿目标 |
| --- | --- | --- |
| `z ≤ -entry_z` | `+quantity` | `-quantity` |
| `z ≥ entry_z` | `-quantity` | `+quantity` |
| `abs(z) ≤ exit_z` | 0 | 0 |
| 其余区间 | 保持方向 | 保持方向 |

方向变化立即更新目标。方向未变且持仓方向非零时，按累计完整帧数满足 `rebalance_interval` 重提目标，供执行层调整实际持仓；这不会逐次累加手数。

### 换约与完整目标表

任一角色合约变化即启动换约：清空统计窗口、方向归零；若旧双腿有实际仓位或在途订单，等待旧双腿同分钟行情提交零目标。只有确认旧敞口消失后，才等待新双腿同步行情切换，并重新预热。旧合约缺文件且仍有敞口时明确报错。

目标表包含回测区间注册的全部真实合约，但有效组合只有当前两腿。其他合约的零值用于清理历史仓位，不能把“字典里出现多个合约”理解为同时套利多个组合。批量目标不保证两腿同时成交，相等手数也不保证逐时金额完全相等。

本例没有固定止损、持有期限或末日强制平仓。退出依赖价差进入出场区间或换约清仓。验收应检查入场 z-score、两腿成交方向、旧组合实际归零和换约后重新预热，而不只检查信号目标。

## 3. 使用核心组件描述

### dataprep：输入准备

| 关键接口 | 本例用途 |
| --- | --- |
| input_session / read_feather | main 与 run_case 共享读取缓存、来源版本和诊断；读取基础条款 |
| resolve_futures_args | 返回 DataPaths，解析行情目录、角色表与基础条款路径 |
| product_inventory / load_role_assignments | 列出所选品种每天真实行情合约，读取并校验所需角色列与重复记录 |
| BarFileKey / plan_fixed_contracts | 声明每日必需的新双腿文件和换约日可选旧合约文件，生成 InputPlan |
| BarReadSpec / prepare_fixed_contracts | 使用固定合约场景统一读取、时间归一及严格行情校验，返回已准备的 HistoricalInputBundle |
| venue / futures.instrument | 校验真实合约条款；调用 trader 合约工厂，返回合约、行情元信息和乘数 |
| add_prepared_bar_source | 将 Bundle 中已校验的行注册到 Feed，不重新读取原文件 |
| write_input_reports | 输出输入来源、文件覆盖、假设与问题，便于对照交易报表 |

### datahub：参考资料服务

| 关键接口 | 本例用途 |
| --- | --- |
| 本例不调用 DataHub Store | 使用 CalendarSelection 的固定双腿日程；无需复权因子或连续角色价格查询。角色文件读取仍由 dataprep 完成 |

### market：行情回放

| 关键接口 | 本例用途 |
| --- | --- |
| FileReplayFeed | 保存真实合约元信息与已准备行情源，按时间回放 Bar |
| DataType.BAR / MarketStreamBinding | 声明每个真实合约的一分钟 Bar 流 |

### trader：策略与执行

| 关键接口 | 本例用途 |
| --- | --- |
| StrategyTemplate / set_targets | 策略提交两腿等手数反向目标，保留信号与换约原因 |
| CtpFuturesBasicProfile / NautilusSimExecutionBackend | 模拟账户、手续费、保证金及真实合约回测 |
| PositionManager / NetTargetOrderPlanner | 维护实际与在途敞口，将目标转换为净持仓订单 |
| MarketReferencePriceStore / PreTradeRiskManager / RiskLimits | 用真实合约参考价检查数量、名义金额和行情时效 |
| DataBinding / ExecutionRoute | 绑定真实行情合约和对应执行路由 |
| UnifiedStrategyRunner / NautilusMarketFeedAdapter / UnifiedHistoricalRuntime | 串联行情、策略、执行客户端及历史回放生命周期 |

## 4. Live版本

从 `pro` 根目录用模块入口启动：`python -u -m demos.03_calendar_spread.run_live`。
默认 Recording 只连接CTP行情并记录目标；Model维护独立归属成交账本；SimNow
连接交易前置并执行受控模拟盘委托。三种模式复用上面的价差公式和同一个策略。
生产代码不从其他demo导入组件。

### 运行前资料与环境

`.env`继续使用项目已有配置。Recording / Model只需要`CTP_MD_ADDRESS`、
`CTP_BROKER_ID`（默认9999）、`CTP_ACCOUNT_ID`、`CTP_PASSWORD`和
`CTP_PRODUCTION_MODE`；不要求TD地址、AppID和AuthCode。SimNow另需
`CTP_TD_ADDRESS`、`CTP_APP_ID`、`CTP_AUTH_CODE`，沿用公共账户锁及模拟盘授权。
DolphinDB地址、账户及表配置复用`DolphinDbReferenceConfig.from_env`。

参考资料使用DolphinDB的角色和期货基础表，不使用分钟文件或复权因子表。
必须提供所选品种的两个角色及其真实条款；角色来源日严格早于本次参考交易日，
`available_ns`仍核验发布时间。默认secondary / far，不因缺腿改用main或recent。
支持SHFE、INE、DCE、CZCE人民币期货，郑商所柜台代码由公共适配器转换。
可选手工核验必须同时提供两腿，例如
`--expected-instrument secondary=rb2701.SHFE --expected-instrument far=rb2705.SHFE`，
其中示例合约须根据当前资料替换。

下面日期变量须先填入实际核验值；Replay的原始MD日也须从当前前置核对。
不要将示例合约或文档日期当作实际登录结果。

```bash
export PYTHONPATH=.
export REFERENCE_DAY=YYYYMMDD
export ROLE_SOURCE_DAY=YYYYMMDD
```

### Recording

```bash
python -u -m demos.03_calendar_spread.run_live \
  --connect --mode recording --product RB \
  --expected-trading-day "$REFERENCE_DAY" --expected-source-day "$ROLE_SOURCE_DAY" \
  --reference-database dfs://bomber_daily \
  --leg1-role secondary --leg2-role far \
  --quantity 1 --lookback 120 --entry-z 2 --exit-z 0.5 \
  --rebalance-interval 5 --max-notional 1000000 --seconds 9000
```

默认实时环境使用事件时间聚合；只有收到下一分钟成交才发送上一分钟Bar，
无成交分钟不补价，两腿同分钟完整才推进窗口。跳过启动时的不完整分钟。
默认需要至少121个完整帧，运行时间应覆盖有效交易时段。

第二套SimNow Replay可在命令中增加：

```bash
--simnow-environment replay --replay-md-trading-day "$MD_DAY" --instrument-max-age-seconds 30
```

须先设置实际`MD_DAY`和对应MD前置。Replay按接收时间聚合，策略不判断模式；
原始MD日、深度、接收心跳与源时间倒退仍检查。参考交易日、角色来源日及可见性
不因回放而豁免；SimNow的TD日必须等于`REFERENCE_DAY`。

### Model与恢复

在Recording命令上替换模式并增加状态文件：

```bash
--mode model --strategy-position-source model --account-reconciliation off \
--state-file demos/03_calendar_spread/results/calendar-model.json
```

新会话须使用不存在的文件。恢复使用完全相同的模式、参数、参考资料与原文件，
追加`--resume`，不换文件名。模型参考价成交有model来源标记，不代表盘口或柜台
成交，不模拟费用、滑点和限价队列。

恢复保留归属仓位和目标revision，清空行情窗口；重新预热后的首个有效判断才
发布新目标，即使目标归零也会提交，以处理保留仓位。恢复预热期间旧目标不续发。
本地或柜台仍有活动订单、参数/合约/资料指纹/行情环境/交易日变化时拒绝恢复。

### 受控SimNow

在Recording命令上替换模式，并增加：

```bash
--mode simnow --enable-orders --confirm-simnow \
--state-file demos/03_calendar_spread/results/calendar-simnow.json
```

默认strict核验管理双腿空仓；范围外仓位不归属本策略。恢复原文件追加`--resume`，
并核验同日成交账本、归属、今昨仓及柜台活动订单。也可显式配置
`--account-reconciliation observe|off --strategy-position-source fills`，物理账户锁、
资金与活动订单核验仍生效。本入口只支持Broker9999模拟盘，不授权真实资金交易。

每腿持仓上限为quantity，单笔数量允许2×quantity以支持Model反手；
每腿订单和持仓名义金额均受max-notional限制，超限拒绝。
SimNow还支持`--limit-offset-ticks`（默认1）、`--max-session-orders`（默认20）
和`--query-timeout`（默认15秒）。原生反手先平后开，由新鲜Tick推进保留目标；
SHFE/INE平今平昨和DCE/CZCE通用平仓复用公共Planner。
多腿提交不保证原子成交；首腿拒单或回报冲突后停止后续腿并要求核对。

### 会话范围和输出

这是固定交易日、固定双腿会话。资料换版、角色/条款变化、MD/TD日期变化时闭闸
停止，停机保留已成交仓位；SimNow撤销活动订单并核验保存账本。
目前没有跨日自动換约或跨日带仓恢复；有旧仓时不能换用新合约和新文件绕过核验。
历史回测原来的清旧仓换约逻辑继续保留，本次没有声明在线动态换约已完成。

默认输出`results/calendar-spread-<mode>-<ns>/summary.json`和`events.txt`，
可用`--report-dir`改输出根目录。报告包含双腿、帧数、z-score、目标、归属仓位、
模型/执行仓位、可选物理账户观测、回调数、参考指纹及就绪渠道。
Recording的execution_origin为recording、strategy_position_source为none。

默认`passed`要求至少lookback+1个同步帧；没有信号或成交也可能通过分钟帧验收。
Model/SimNow需要本次归属成交证据时增加`--require-fills`；Recording不接受此参数。
即使有成交也须分别检查双腿实际仓位、拒单和停机结果，不以受理回报当成交。

### 离线验收（Linux uv-nautilus）

```bash
python -u -m tests.run_calendar_spread_acceptance
```

入口先核验`tests/calendar_spread_live.sha256`，再运行22个Live测试方法及原有
选约/信号断言，保存分组日志与JSON。它使用假MD/TD，既不连接外部系统也不
报实际订单。可用`--only live`或`--only selection`重跑失败组。
公共代码未改，可选`--only public`复核既有公共组。

当前本机仅做静态审阅和文件SHA核验，未运行Python、测试或编译；远端原生导入、
离线组、实际DolphinDB/CTP行情、Model恢复及SimNow成交验收均待运行。
详见[交接记录](../../doc/HANDOFF/live/HANDOFF_CALENDAR_SPREAD_2026-10-12.md)。
