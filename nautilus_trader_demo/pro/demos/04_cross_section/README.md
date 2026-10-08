# 多品种主力横截面动量策略

## 1. 使用方法

### 启动命令

从 pro 根目录、在已安装运行依赖的环境中执行，先设置 `export PYTHONPATH=.`。入口自动读取已有 `.env` 配置；没有配置数据路径时，可通过可选的 `--data-root` 或各文件路径参数指定。示例日期需与实际数据匹配。

**完整执行（显式设置常用参数）：**

```bash
python demos/04_cross_section/run_backtest.py \
  --products RB,HC,I,J,JD,JM,NI,NR \
  --start-day 2026-01-05 --end-day 2026-05-29 \
  --lookback 20 --rebalance-interval 5 \
  --group-fraction 0.30 --target-notional 300000 \
  --bar-timestamp end --log-level WARNING --no-tearsheet
```

**默认执行（仅必填参数）：**

```bash
python demos/04_cross_section/run_backtest.py \
  --products RB,HC,I,J,JD,JM,NI,NR \
  --start-day 2026-01-05 --end-day 2026-05-29
```

默认执行采用参数表中的默认值及已有数据路径配置；例如默认启用绩效图，不要求必须产生成交。

### 参数说明

“必选”表示启动时必须传入；“可选”表示有默认值或可由已有配置解析。数据路径虽然是可选参数，但运行前必须能解析到有效文件。

| 参数 | 必选 / 可选 | 含义与默认值 |
| --- | --- | --- |
| `--products` | 必选 | 至少两个不重复品种，逗号分隔；全部使用主力合约生成信号和下单 |
| `--start-day / --end-day` | 必选 | 请求日期闭区间 |
| `--lookback` | 可选 | 收益率回看同步帧数，默认 20；需要 21 个价格点。只统计所有品种主力同分钟到齐的帧，不是交易日数 |
| `--rebalance-interval` | 可选 | 默认 5；预热后按累计同步帧数整除间隔调仓；检测到主力变化后首个已预热同步帧也会调仓 |
| `--group-fraction` | 可选 | 排名前后每侧选仓比例，默认 0.30，范围为大于 0 且不超过 0.5；每侧数量为 max(1, floor(当前品种数 × 比例))，多空组不重叠 |
| `--target-notional` | 可选 | 每侧总名义金额预算，默认 100000；多空侧各使用该预算，组内等额分配。整数手数向下取整，不足一手时目标为零，余款保留现金 |
| `--bar-timestamp` | 可选 | 源分钟标签 start / end，默认 end；start 由公共层统一移动到分钟结束 |
| `--factors` | 可选 | 默认角色表同目录 fut_adjustment_factors.feather；需要 trade_date、code、symbol、pcr_cumfactor，pcr_factor 不用于信号 |
| `--factor-availability` | 可选 | 默认 aligned：因子已在上游对齐当前交易日；explicit 要求 available_ns。使用当天已对齐因子 |
| `--contract-struct / --fut-basic / --bars-dir` | 可选 | 覆盖角色表、基础条款、分钟行情路径 |
| `--starting-balance` | 可选 | 每个交易所模拟账户的初始资金，默认 1000000；多交易所账户分别初始化 |
| `--max-notional` | 可选 | 每合约订单与持仓名义金额风控上限，默认 1000000 |
| `--commission-per-contract` | 可选 | 固定每手手续费默认 1 |
| `--margin-init / --margin-maint` | 可选 | 初始 / 维持保证金比例默认 0.10 / 0.08 |
| `--data-root` | 可选 | 可显式指定数据根目录，按 role/ 与 kline/ 布局推导路径；已配置路径时可省略 |
| `--report-dir` | 可选 | 输出根目录，默认本策略目录的 results/ |
| `--log-level` | 可选 | 框架日志等级，默认 WARNING |
| `--no-tearsheet` | 可选 | 指定时仅保存 CSV / JSON，不生成绩效图 |
| `--require-fills` | 可选 | 指定时要求至少一笔模拟成交，否则报错 |

### 数据要求

需要全部品种每日主力及换约日旧主力行情，角色表 main 和外部累计因子表。因子使用 trade_date、code、symbol、pcr_cumfactor，当日对齐；main 合约冲突以因子 symbol 为准。缺失因子明确报错。

| 数据项 | 要求 |
| --- | --- |
| 角色表 | 需要 trade_date、code、main；优先 fut_contract.feather，旧名 fut_contract_data.feather 仅在新名不存在时回退，两个同时存在须显式指定 |
| 基础条款 | fut_basic.feather：symbol、code、exchangeCD、contMultNum、minChgPriceNum、listDate、lastTradeDate |
| 分钟行情 | `<symbol>_YYYYMMDD.feather`；每日主力和换约日旧主力都是必需文件，缺文件在回测前报错 |

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

## 2. 策略详细说明

### 策略思路与排名

这是多品种横截面动量策略，比较同一时刻各品种此前一段时间的相对涨幅，做多强组、做空弱组。只使用每个品种当前主力，研究价格为真实 close × 当天累计因子。

| 环节 | 计算规则 |
| --- | --- |
| 同步 | 全部配置品种的当前主力 Bar 必须具有相同时间戳；少一个品种便不推进窗口 |
| 预热 | 每个品种保存 `lookback + 1` 个价格点，默认 21 个完整帧 |
| 分数 | `当前复权价 / lookback 帧前复权价 - 1` |
| 排序 | 分数由高到低，相同分数按品种代码排序 |
| 每侧组数 | `max(1, floor(品种数 × group_fraction))`；比例上限 0.5，避免组重叠 |
| 选仓 | 前一组做多、后一组做空，中间品种目标归零 |
| 调仓 | 预热后累计同步帧数能整除调仓间隔时提交；换约后的已预热完整帧优先更新 |

分组依据品种数量，不依据历史合约数量。8 个品种、比例 0.30 时每侧 2 个；比例 0.45 时每侧 3 个。即使所有动量为负，仍做多相对最高组；即使分数相同，仍按照排序规则分组，不额外设定绝对涨幅门槛。

### 名义金额与实际手数

每侧各获得 `target_notional` 预算，组内等额分配：

```text
每个入选品种预算 = 每侧预算 / 该侧品种数
目标手数绝对值 = floor(品种预算 / (真实价格 × 合约乘数))
多头取正，空头取负
```

例如每侧 100000、两品种时各分 50000；单手金额大于 50000 的入选品种仍是多头或空头候选，但实际目标为零。不会强制买一手，也不把剩余预算转给其他品种。因此选出多空组不保证两侧都有非零成交，整数取整后也不保证实际金额严格中性。

### 换约、退出与核对

换约保留复权动量窗口，在下一有效调仓帧把旧真实合约目标归零，按新主力真实价格重新配仓。目标表里的历史合约和中间品种零值属于完整仓位管理，不是同时建仓。实际旧仓退出受报价、订单及成交约束。

退出发生在品种落入中间组、由多组转空组或由空组转多组，以及主力换约时。本例没有独立止盈止损或末日强制平仓。验收应同时核对排名、多空组、各品种分配预算、单手金额、整数目标和成交；只有分组而手数全零时，首先检查预算换算。

## 3. 使用核心组件描述

### dataprep：输入准备

| 关键接口 | 本例用途 |
| --- | --- |
| input_session / read_feather | main 与 run_case 复用读取缓存、来源版本及输入诊断，加载基础条款 |
| resolve_futures_args | 定位角色表、基础条款与行情目录，返回 DataPaths |
| prepare_role_research | 准备各品种 main 角色、外部当日累计因子和研究行情；组装 DataHub Store，返回日末时间用于选出回测日期 |
| BarFileKey / plan_fixed_contracts | 声明回测每日真实主力及换约日旧主力必需文件，生成 InputPlan |
| BarReadSpec / prepare_fixed_contracts | 统一时间标签、身份及严格 OHLCV 校验，返回 HistoricalInputBundle |
| selected_products / futures.instrument | 查询品种交易所，校验条款并调用 trader 合约工厂，返回原生合约、行情元信息与乘数 |
| add_prepared_bar_source | 注册已准备行情，避免校验后重新读取原文件 |
| write_input_reports | 输出来源版本、覆盖、假设及因子覆盖角色的诊断 |

### datahub：参考资料查询

| 关键接口 | 本例用途 |
| --- | --- |
| SectorRoleStore / snapshot(timestamp) | 按策略事件时刻查询已生效且可见的主力映射和因子 |
| SectorRoleAssignment.instrument(product, main) | 返回该品种当天最终主力真实合约 |
| SectorRoleAssignment.factor(product, main) | 返回上游提供的当前累计因子，供信号价格复权 |

### market：真实行情回放

| 关键接口 | 本例用途 |
| --- | --- |
| FileReplayFeed | 注册各真实主力及换约旧合约元信息与行情源，回放一分钟 Bar |
| DataType.BAR / MarketStreamBinding | 声明各真实合约的行情流 |

### trader：策略与执行

| 关键接口 | 本例用途 |
| --- | --- |
| StrategyTemplate / set_targets | 提交完整真实合约目标快照，附带排名分数、主力映射及同步帧编号 |
| CtpFuturesBasicProfile / NautilusSimExecutionBackend | 按交易所建立模拟账户并运行真实期货合约回测 |
| PositionManager / NetTargetOrderPlanner | 维护实际与在途持仓，把目标换算成订单 |
| MarketReferencePriceStore / PreTradeRiskManager / RiskLimits | 使用真实价格、乘数与报价时效检查订单和持仓名义金额 |
| DataBinding / ExecutionRoute | 绑定行情及各真实合约的执行路由 |
| UnifiedStrategyRunner / NautilusMarketFeedAdapter / UnifiedHistoricalRuntime | 串联行情观察者、策略、模拟执行及历史运行生命周期 |
