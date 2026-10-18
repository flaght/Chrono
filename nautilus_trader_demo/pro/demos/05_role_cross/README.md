# 单品种三角色复权价穿越策略

## 1. 使用方法

### 启动命令

从 pro 根目录、在已安装运行依赖的环境中执行，先设置 `export PYTHONPATH=.`。入口自动读取已有 `.env` 配置；没有配置数据路径时，可通过可选的 `--data-root` 或各文件路径参数指定。示例日期需与实际数据匹配。

**完整执行（显式设置常用参数）：**

```bash
python demos/05_role_cross/run_backtest.py \
  --product RB \
  --start-day 2026-01-05 --end-day 2026-05-29 \
  --quantity 1 \
  --bar-timestamp end --log-level WARNING --no-tearsheet
```

**默认执行（仅必填参数）：**

```bash
python demos/05_role_cross/run_backtest.py \
  --product RB \
  --start-day 2026-01-05 --end-day 2026-05-29
```

默认执行采用参数表中的默认值及已有数据路径配置；例如默认启用绩效图，不要求必须产生成交。

### 参数说明

“必选”表示启动时必须传入；“可选”表示有默认值或可由已有配置解析。数据路径虽然是可选参数，但运行前必须能解析到有效文件。

| 参数 | 必选 / 可选 | 含义与默认值 |
| --- | --- | --- |
| `--product` | 必选 | 一个品种，例如 RB、I、HC；三个角色都属于该品种 |
| `--start-day / --end-day` | 必选 | 请求日期闭区间 |
| `--quantity` | 可选 | 主力逻辑目标手数，默认 1，必须为正整数 |
| `--bar-timestamp` | 可选 | 源分钟标签 start / end，默认 end；公共层把 start 移到分钟结束 |
| `--contract-struct / --fut-basic / --bars-dir` | 可选 | 覆盖角色表、基础条款和真实行情目录 |
| `--starting-balance / --commission-per-contract` | 可选 | 初始资金默认 1000000，固定每手手续费默认 1 |
| `--margin-init / --margin-maint` | 可选 | 初始 / 维持保证金比例，默认 0.10 / 0.08 |
| `--max-notional / --max-market-age-seconds` | 可选 | 每合约订单及持仓名义金额上限、真实报价最大年龄，默认 1000000 / 120 秒 |
| `--require-fills` | 可选 | 可选，要求至少一笔成交；样本可能没有穿越信号，初次检查可省略 |
| `--data-root` | 可选 | 可显式指定数据根目录，按 role/ 与 kline/ 布局推导路径；已配置路径时可省略 |
| `--report-dir` | 可选 | 输出根目录，默认本策略目录的 results/ |
| `--log-level` | 可选 | 框架日志等级，默认 WARNING |
| `--no-tearsheet` | 可选 | 指定时仅保存 CSV / JSON，不生成绩效图 |

### 数据要求

需要角色表 main、second、far，以及三个角色每日真实行情和换约日旧主力行情。三个角色各自的复权研究价格由分钟角色场景准备，不读取外部主力累计因子表。

| 数据项 | 要求 |
| --- | --- |
| 角色表 | trade_date、code、main、second、far；分别映射主力、次主力、远期，recent 不参与。使用严格早于当日的最新角色记录 |
| 角色表文件名 | 优先 fut_contract.feather，旧名 fut_contract_data.feather 仅在新名不存在时回退；两个同时存在须显式指定 |
| 基础条款 | fut_basic.feather：symbol、code、exchangeCD、contMultNum、minChgPriceNum、listDate、lastTradeDate |
| 分钟行情 | `<symbol>_YYYYMMDD.feather`；三个当日角色和换约日旧主力都是必需输入 |

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

### 策略思路与价格口径

这是单品种期限结构穿越策略：观察主力相对主力、次主力、远期三角色复权均价的位置变化，只对主力建立方向仓位。三个角色用于研究，不组成三腿交易组合；`recent` 不参与。

```text
P_main、P_secondary、P_far = 三角色各自的复权研究收盘价
M = (P_main + P_secondary + P_far) / 3
D = P_main - M
```

策略须等三个角色对应的真实行情事件同分钟到齐，并确认快照中每个研究价格的来源时间均等于当前时间，才计算 D。文件预加载使研究价可查询，并不意味着可以提前发信号。

| 穿越条件 | 主力目标 |
| --- | --- |
| 上一 D ≤ 0，当前 D > 0 | `+quantity` |
| 上一 D ≥ 0，当前 D < 0 | `-quantity` |
| 第一完整帧 | 保存 D，不下单 |
| 未发生上述穿越 | 不提交新目标，保持原目标 |

当前 D 恰好为零不会主动平仓；之后从零进入正或负区间可以产生对应穿越。若回测第一帧 D 已为正，策略仍等待后续穿越，不立即做多。这与按当前均线关系直接建仓的 EMA 示例不同。

### 因子、换约与执行

| 环节 | 当前实现 |
| --- | --- |
| 因子来源 | 分钟角色场景为三个角色分别维护因果换约因子，不读取外部主力累计因子表 |
| 初始与累计 | 初始值为 1；换约使用来源日旧、新合约共同收盘价比值累计 |
| 锚点缺失 | 最多回看此前一个有行情交易日；仍缺失时记录缺口，使相关快照不可用 |
| 信号连续 | 角色切换不重置上一 D，继续比较复权研究价 |
| 执行目标 | 一个主力逻辑键；动态路由负责真实主力切换和旧仓处理 |
| 成交价格 | 主力真实未复权行情，次主力和远期不建立目标仓位 |

三条复权序列的相对水平受各自初始基准及换约口径影响，因此 D 表示该研究构造下的相对关系，不等于原始近远月价差。没有穿越的区间可以完全没有交易；`require-fills` 只用于验收成交条件，不会强制产生信号。

本例没有中性平仓、独立止盈止损或末日强制平仓。验收时检查完整三角色帧、因子锚点及缺口、D 的前后符号，并核对只有主力产生实际订单。

## 3. 使用核心组件描述

### dataprep：输入准备

| 关键接口 | 本例用途 |
| --- | --- |
| input_session / read_feather | main 与 run_case 共享文件读取缓存、来源版本及输入诊断；加载基础条款 |
| resolve_futures_args | 解析角色资料、基础条款、真实行情路径，返回 DataPaths |
| prepare_role_research(minute_roles=SIGNAL_ROLES) | 调用单品种分钟角色场景，读取研究收盘价，组装角色价格库并返回日末时间、文件数和行数 |
| BarFileKey / plan_fixed_contracts | 声明每日三个角色及换约旧主力的必需文件，生成 InputPlan |
| BarReadSpec / prepare_fixed_contracts | 统一时间标签、合约身份及严格 OHLCV 校验，返回已准备的 HistoricalInputBundle |
| venue / futures.instrument | 校验交易所、基础条款、真实合约身份，调用 trader 合约工厂返回合约、行情元信息和乘数 |
| add_prepared_bar_source | 将已校验行情注册到回放源，不重新读取原文件 |
| write_input_reports | 输出来源、覆盖、准备假设与问题诊断 |

### datahub：角色研究价查询

| 关键接口 | 本例用途 |
| --- | --- |
| RolePriceStore / assignment_at | 保存分钟真实收盘价、三角色日程与因果换约因子；按日末时间查找当日日程 |
| MinimalDataHub / snapshot(timestamp) | 查询事件时刻可见的三角色研究价，缺因子或资料不可用时明确拒绝查询 |
| RoleSnapshot / RolePrice | 提供真实合约映射、来源时间、原始收盘价与各角色复权研究价 |
| factor_gaps / factor_anchors | 记录因子缺失及来源日以外的共同收盘价回看锚点，写入摘要 |

### market：真实行情回放

| 关键接口 | 本例用途 |
| --- | --- |
| FileReplayFeed | 注册主力、次主力、远期及换约旧主力的真实行情与元信息 |
| DataType.BAR / MarketStreamBinding | 声明每个真实合约的一分钟行情流 |

### trader：主力逻辑目标执行

| 关键接口 | 本例用途 |
| --- | --- |
| StrategyTemplate / set_target | 将穿越方向转换为主力逻辑目标的正负手数 |
| ScheduledContractResolver / ContractAssignment | 保存主力逻辑键到真实主力的生效时间、可用时间和版本日程 |
| DynamicExecutionRoute | 按日程解析真实执行合约，并协调主力换约处理 |
| CtpFuturesBasicProfile / NautilusSimExecutionBackend | 模拟账户、手续费、保证金及真实期货合约回测 |
| PositionManager / NetTargetOrderPlanner | 维护实际与在途敞口，将目标转为订单 |
| MarketReferencePriceStore / PreTradeRiskManager / RiskLimits | 使用真实价格、数量、乘数和报价时效执行交易前风控 |
| DataBinding / UnifiedStrategyRunner / NautilusMarketFeedAdapter / UnifiedHistoricalRuntime | 绑定真实行情、策略及模拟执行，管理历史回放生命周期 |
