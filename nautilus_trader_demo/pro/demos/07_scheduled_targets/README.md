# 外部完整目标计划策略

## 1. 使用方法

### 启动命令

从 pro 根目录、在已安装运行依赖的环境中执行，先设置 `export PYTHONPATH=.`。入口自动读取已有 `.env` 配置；没有配置数据路径时，可通过可选的 `--data-root` 或各文件路径参数指定。示例日期需与实际数据匹配。

**完整执行（显式设置常用参数）：**

```bash
python demos/07_scheduled_targets/run_backtest.py \
  --targets demos/07_scheduled_targets/targets.csv \
  --start-day 2026-05-06 --end-day 2026-05-29 \
  --bar-timestamp end --log-level WARNING --no-tearsheet
```

**默认执行（仅必填参数）：**

```bash
python demos/07_scheduled_targets/run_backtest.py \
  --targets demos/07_scheduled_targets/targets.csv \
  --start-day 2026-05-06 --end-day 2026-05-29
```

默认执行采用参数表中的默认值及已有数据路径配置；例如默认启用绩效图，不要求必须产生成交。

### 参数说明

“必选”表示启动时必须传入；“可选”表示有默认值或可由已有配置解析。数据路径虽然是可选参数，但运行前必须能解析到有效文件。

| 参数 | 必选 / 可选 | 含义与默认值 |
| --- | --- | --- |
| `--targets` | 必选 | 必填，外部 CSV；支持绝对路径、当前目录相对路径和项目根目录相对路径 |
| `--start-day / --end-day` | 必选 | 回测日期范围，全部计划时点必须落在该范围内 |
| `--timezone` | 可选 | 计划时间和行情时间使用的时区，默认 Asia/Shanghai |
| `--bar-timestamp` | 可选 | 源分钟标签 start / end，默认 end；公共层把 start 移到分钟结束 |
| `--fut-basic / --bars-dir` | 可选 | 覆盖基础条款与真实行情目录，无需 --contract-struct |
| `--starting-balance` | 可选 | 每个交易所账户初始资金，默认 1000000；多个交易所分别初始化 |
| `--max-quantity / --max-notional` | 可选 | 每合约最大绝对持仓手数和订单 / 持仓名义金额上限，默认 100 / 1000000 |
| `--max-market-age-seconds` | 可选 | 风控参考价格最大年龄，默认 120 秒 |
| `--allow-missed-slots` | 可选 | 允许存在未提交时点，仍记录审计；不表示允许追单，也不跳过交易执行异常检查 |
| `--require-fills` | 可选 | 可选，要求至少一笔成交；计划提交成功不等同于已经成交 |
| `--commission-per-contract` | 可选 | 固定每手手续费默认 1 |
| `--margin-init / --margin-maint` | 可选 | 初始 / 维持保证金比例默认 0.10 / 0.08 |
| `--data-root` | 可选 | 可显式指定数据根目录，按 role/ 与 kline/ 布局推导路径；已配置路径时可省略 |
| `--report-dir` | 可选 | 输出根目录，默认本策略目录的 results/ |
| `--log-level` | 可选 | 框架日志等级，默认 WARNING |
| `--no-tearsheet` | 可选 | 指定时仅保存 CSV / JSON，不生成绩效图 |

### 数据要求

需要目标 CSV、对应真实期货基础条款及区间行情，不需要角色表或复权因子。CSV 必须包含 timestamp、target_key（兼容 instrument_id）、target_qty；同一时点构成完整目标组合，数量为有限整数，重复合约键报错。无时区时间按 timezone 解释。所有计划须落在请求日期内。

| 数据项 | 要求 |
| --- | --- |
| timestamp | ISO 日期时间；无时区时按 --timezone 解释，同一时点的记录组成完整目标组合 |
| target_key / instrument_id | target_key 优先，兼容旧 instrument_id 列；真实合约可带交易所后缀，也可由基础条款确定交易所 |
| target_qty | 有限整数，正数做多、负数做空、零平仓；同一时点同一键不能重复 |
| instrument_type | 可选；如提供则只接受 FUTURE、FUTURES、CTP_FUTURES |
| 基础条款 | fut_basic.feather：symbol、code、exchangeCD、contMultNum、minChgPriceNum、listDate、lastTradeDate；每个目标必须唯一对应真实合约 |
| 分钟行情 | `<symbol>_YYYYMMDD.feather`；读取计划涉及合约在请求范围内已有的文件，统一严格校验 |

### 输出内容

默认保存到本策略目录的 `results/<run-id>/`，可通过 `--report-dir` 修改根目录。控制台输出数据准备、运行回测、报表与清理的阶段耗时。

| 输出文件 | 内容 |
| --- | --- |
| orders.csv | 模拟委托及订单状态 |
| fills.csv | 实际模拟成交；提交目标不等于成交 |
| positions.csv | 真实合约持仓记录 |
| account_<交易所>.csv | 模拟账户及资金记录 |
| summary.json | 回测摘要、策略参数、信号或调仓统计 |
| input_manifest.json / input_coverage.json / input_issues.json | 输入版本、覆盖与质量诊断 |
| tearsheet.html | 可选绩效图；省略 --no-tearsheet 且绘图依赖可用时生成 |
| schedule_audit.csv | 计划时点提交、错过与未到达审计 |

## 2. 策略详细说明

### 策略思路与完整目标语义

这是外部交易计划执行策略。CSV 提供时间和真实合约目标，策略不根据价格计算方向；适合回放已知的调仓计划，检查执行、风控和仓位变化。计划既可以是单合约，也可以是跨品种多空组合。

| 输入或事件 | 处理规则 |
| --- | --- |
| 同一时间戳的多行 | 合并为一个完整组合计划 |
| 正目标 / 负目标 / 零目标 | 分别表示做多、做空、平仓的最终手数 |
| 计划未列出的旧目标 | 使用 REPLACE 语义归零 |
| 与已有目标相同 | 仍可提交计划，但执行层按实际及在途持仓判断是否需要订单 |
| 同一时点重复合约 | 输入校验报错，不把数量相加 |

例如已有 RB +1、HC -1，新计划只写 RB +2，则最终意图是 RB +2、HC 0；不是追加 RB 两手并保留 HC。目标数值与新增订单数量不同，订单由执行层根据仓位差额计算。

### 时钟、行情与审计

独立计划时钟与行情合并回放，同时间戳先处理行情再触发计划。策略只在精确计划时刻输出目标，不通过“收到下一根 Bar”顺延计划，也不对错过的时点追单。已有参考行情仍须通过风控时效检查，独立时钟本身不能提供成交价格。

| 审计状态 | 含义 |
| --- | --- |
| SUBMITTED | 完整目标已成功提交，不代表已经成交 |
| MISSED | 时钟越过计划时点，记录遗漏且不补发 |
| NOT_REACHED | 结束前未到该时点，或未完成提交 |

重复时钟不会重复输出，倒退时钟报错。当前 CSV 被视为启动前已知的静态计划，不处理盘中发布或修订信息。`allow-missed-slots` 只放宽时点验收，不改变精确时点规则或执行异常检查。

### 合约与退出规则

计划使用明确真实合约，不自动切换主力。换约需在下一完整计划中把旧合约归零并加入新合约；若未指定新合约，系统不会替使用者续仓。平仓、止盈止损、最终清仓也都由计划决定，回测结束没有隐含的强制平仓。

验收应对照原始 CSV、`schedule_audit.csv`、订单、成交和最终持仓。最后计划提交零目标后仍需要后续行情撮合；审计显示 SUBMITTED 不能证明旧仓已经退出。

## 3. 使用核心组件描述

### dataprep：输入准备

| 关键接口 | 本例用途 |
| --- | --- |
| input_session / read_feather | 共享基础条款读取缓存、来源版本和输入诊断 |
| resolve_futures_args(require_roles=False) / resolve_targets_path | 解析基础条款、行情目录及外部计划路径，无需角色表 |
| load_target_csv | 校验列名、时区、整数目标和重复键，将同一时点记录组装为目标计划库 |
| contract_rows / symbol_of / venue_of | 校验目标真实期货合约、唯一元数据和交易所身份 |
| bar_files / key_from_path / plan_fixed_contracts | 收集指定真实合约区间文件，生成固定合约 InputPlan |
| BarReadSpec / prepare_fixed_contracts | 统一分钟时间口径、合约身份及严格 OHLCV 校验，返回 HistoricalInputBundle |
| futures.make_instrument / add_prepared_bar_source | 按真实条款构建合约，将已准备行情注册到回放源，不再次读取文件 |
| write_input_reports | 输出输入来源、文件覆盖及准备假设和问题 |

### datahub：完整目标计划查询

| 关键接口 | 本例用途 |
| --- | --- |
| TargetPlan | 保存一个时点的完整目标、可用时间、来源版本和附加信息 |
| TargetScheduleStore.slots / slots_between / at | 提供计划时点日程、查询经过的时点和触发时刻可用的完整目标 |

### market：真实行情输入

| 关键接口 | 本例用途 |
| --- | --- |
| DataType.BAR / MarketStreamBinding | 声明计划涉及合约的一分钟真实行情流 |
| 已准备行情源 | 为模拟撮合和参考价观察者提供真实行情，策略通过独立时钟触发 |

### trader：计划时钟与目标执行

| 关键接口 | 本例用途 |
| --- | --- |
| TimedFileReplayFeed | 合并真实行情和计划时钟，同时间戳先行情后时钟 |
| StrategyTemplate / set_targets / TargetUpdateMode.REPLACE | 提交完整组合，未列出的旧目标按替换语义归零 |
| ExecutionRoute | 将外部目标键映射到明确的真实执行合约 |
| CtpFuturesBasicProfile / NautilusSimExecutionBackend | 按交易所建立账户并执行真实期货模拟回测 |
| PositionManager / NetTargetOrderPlanner | 维护实际与在途持仓，将目标转为净持仓订单 |
| MarketReferencePriceStore / PreTradeRiskManager / RiskLimits | 使用真实报价、数量、乘数和价格年龄进行风控 |
| UnifiedStrategyRunner / NautilusMarketFeedAdapter / UnifiedHistoricalRuntime | 绑定独立时钟、行情观察者及模拟执行，管理历史回放生命周期 |
