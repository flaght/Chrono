# 单品种跨期价差策略

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
