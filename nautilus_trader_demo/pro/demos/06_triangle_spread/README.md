# 三品种主力相对价值策略

## 1. 使用方法

### 启动命令

从 pro 根目录、在已安装运行依赖的环境中执行，先设置 `export PYTHONPATH=.`。入口自动读取已有 `.env` 配置；没有配置数据路径时，可通过可选的 `--data-root` 或各文件路径参数指定。示例日期需与实际数据匹配。

**完整执行（显式设置常用参数）：**

```bash
python demos/06_triangle_spread/run_backtest.py \
  --anchor RB --hedges HC,I --hedge-weights 0.5,0.5 \
  --start-day 2026-01-05 --end-day 2026-05-29 \
  --lookback 120 --entry-z 2 --exit-z 0.5 \
  --target-notional 300000 --rebalance-interval 5 \
  --bar-timestamp end --log-level WARNING --no-tearsheet
```

**默认执行（仅必填参数）：**

```bash
python demos/06_triangle_spread/run_backtest.py \
  --anchor RB --hedges HC,I \
  --start-day 2026-01-05 --end-day 2026-05-29
```

默认执行采用参数表中的默认值及已有数据路径配置；例如默认启用绩效图，不要求必须产生成交。

### 参数说明

“必选”表示启动时必须传入；“可选”表示有默认值或可由已有配置解析。数据路径虽然是可选参数，但运行前必须能解析到有效文件。

| 参数 | 必选 / 可选 | 含义与默认值 |
| --- | --- | --- |
| `--anchor / --hedges` | 必选 | 一个主腿品种、两个不同对冲品种；三个品种均使用主力信号及主力执行 |
| `--start-day / --end-day` | 必选 | 请求日期闭区间 |
| `--hedge-weights` | 可选 | 两条对冲腿正权重，合计为 1；默认 0.5,0.5 |
| `--lookback` | 可选 | 此前同步价差窗口根数，默认 120；第 121 个完整同步帧开始计算标准分数，不是交易日数 |
| `--entry-z / --exit-z` | 可选 | 入场 / 出场阈值，默认 2 / 0.5，满足 0 ≤ 出场阈值 < 入场阈值 |
| `--target-notional` | 可选 | 主腿目标名义金额，默认 300000；对冲侧合计相同金额，按权重分配。不是三腿总金额 |
| `--rebalance-interval` | 可选 | 持仓每隔若干同步帧重算手数，默认 5；方向变化或主力换约在下一已预热完整帧更新 |
| `--factors` | 可选 | 默认角色表同目录 fut_adjustment_factors.feather；使用 trade_date、code、symbol、pcr_cumfactor，忽略单期 pcr_factor |
| `--factor-availability` | 可选 | 默认 aligned，使用上游已对齐当日因子；explicit 要求 available_ns |
| `--bar-timestamp` | 可选 | 源分钟标签 start / end，默认 end；公共层把 start 移到分钟结束 |
| `--contract-struct / --fut-basic / --bars-dir` | 可选 | 显式覆盖角色表、基础条款、分钟行情目录 |
| `--starting-balance` | 可选 | 每个交易所账户的初始资金，默认 1000000，各交易所分别初始化 |
| `--max-notional / --max-market-age-seconds` | 可选 | 每合约订单和持仓名义金额上限、报价最大年龄，默认 1000000 / 120 秒 |
| `--require-fills` | 可选 | 可选，要求至少一笔成交；没有穿越信号的样本可不加 |
| `--commission-per-contract` | 可选 | 固定每手手续费默认 1 |
| `--margin-init / --margin-maint` | 可选 | 初始 / 维持保证金比例默认 0.10 / 0.08 |
| `--data-root` | 可选 | 可显式指定数据根目录，按 role/ 与 kline/ 布局推导路径；已配置路径时可省略 |
| `--report-dir` | 可选 | 输出根目录，默认本策略目录的 results/ |
| `--log-level` | 可选 | 框架日志等级，默认 WARNING |
| `--no-tearsheet` | 可选 | 指定时仅保存 CSV / JSON，不生成绩效图 |

### 数据要求

需要三个品种的主力、换约日旧主力行情及当日外部累计因子表。累计因子字段为 trade_date、code、symbol、pcr_cumfactor；main 合约冲突以因子 symbol 为准。

| 数据项 | 要求 |
| --- | --- |
| 角色表 | trade_date、code、main；优先 fut_contract.feather，旧名 fut_contract_data.feather 仅在新名缺失时回退；两个同时存在须显式指定 |
| 基础条款 | fut_basic.feather：symbol、code、exchangeCD、contMultNum、minChgPriceNum、listDate、lastTradeDate |
| 分钟行情 | `<symbol>_YYYYMMDD.feather`；三个当日主力和换约日旧主力均为必需文件 |

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

### 策略思路与三腿价差

这是三个品种的统计相对价值策略：一个主腿与两个加权对冲腿构成归一化对数价差，交易价差偏离历史均值后的回归。品种及权重由参数给定，代码不估计回归系数，也不自动识别产业关系。

三个当前主力同分钟到齐后，以各自真实 close × 累计因子得到 P。第一完整帧固定为各腿基准 P₀：

```text
S(t) = ln(P_anchor(t) / P_anchor₀)
     - w1 × ln(P_hedge1(t) / P_hedge1₀)
     - w2 × ln(P_hedge2(t) / P_hedge2₀)
w1 > 0，w2 > 0，w1 + w2 = 1
z(t) = (当前 S - 此前 lookback 个 S 的均值) / 总体标准差
```

当前帧不进入自己的统计基准。默认第 121 个同步帧开始出信号，零方差时 z 为零；缺少任一主力当分钟行情便不推进窗口。

| 条件 | 主腿 | 对冲腿 1、2 |
| --- | --- | --- |
| `z ≤ -entry_z` | 多 | 空 |
| `z ≥ entry_z` | 空 | 多 |
| `abs(z) ≤ exit_z` | 目标归零 | 目标归零 |
| 其余区间 | 保持方向 | 保持方向 |

### 金额配置、调仓与换约

`target_notional` 是主腿预算 N，两对冲腿预算分别为 `N × w1` 和 `N × w2`。三腿毛名义预算合计为 2N，每腿均按真实价格和乘数向下取整成整数手数。权重用于名义金额分配，不直接表示合约手数比例；任一腿不足一手时整个配仓报错，避免悄悄缺腿。

方向改变立即调仓；持仓方向不变时按累计同步帧满足间隔重新计算手数；任一主力变化后，下一已预热同步帧也会更新。周期调仓可能因价格变化而改变数量，即使 z 的方向未变也可能产生差额订单。

换约保留首帧基准与统计窗口，将历史真实合约目标归零，并把当下方向配置到新主力。该路径提交完整组合目标，未采用跨期示例中的“旧组合实际全平后再启用新组合”专门状态机，成交顺序由执行层和行情决定。

本例没有固定止损或末日强制平仓。等额多空预算与对数价差权重不保证实际金额、产业因子或价格敏感度严格中性。验收应检查三腿同步、z-score、各腿预算与整数手数、出场零目标，以及旧合约是否实际退出。

## 3. 使用核心组件描述

### dataprep：输入准备

| 关键接口 | 本例用途 |
| --- | --- |
| input_session / read_feather | main 与 run_case 共享读取缓存、来源版本和诊断，加载基础条款 |
| resolve_futures_args | 定位角色表、基础条款和行情目录，返回 DataPaths |
| prepare_role_research | 准备三个品种 main 映射、当日外部累计因子及研究收盘价，组装参考资料库并返回日末时间 |
| BarFileKey / plan_fixed_contracts | 声明每日三条真实主力及换约日旧主力必需文件，生成 InputPlan |
| BarReadSpec / prepare_fixed_contracts | 统一时间口径、合约身份和严格 OHLCV 校验，返回 HistoricalInputBundle |
| selected_products / futures.instrument | 读取交易所及合约条款，调用 trader 合约工厂，返回合约、行情元信息与乘数 |
| add_prepared_bar_source | 注册已准备行情，不重新读取原始文件 |
| write_input_reports | 输出输入来源、覆盖、假设及因子覆盖角色映射的诊断 |

### datahub：参考资料查询

| 关键接口 | 本例用途 |
| --- | --- |
| SectorRoleStore / snapshot(timestamp) | 查询事件时间已生效且可见的三个品种角色快照 |
| SectorRoleAssignment.instrument(product, main) | 提供最终真实主力合约映射 |
| SectorRoleAssignment.factor(product, main) | 提供当日外部累计因子，供信号价格复权 |

### market：真实行情回放

| 关键接口 | 本例用途 |
| --- | --- |
| FileReplayFeed | 注册三个真实主力及换约旧主力的元信息和行情源，按时间回放 |
| DataType.BAR / MarketStreamBinding | 声明各真实合约的一分钟行情流 |

### trader：三腿目标执行

| 关键接口 | 本例用途 |
| --- | --- |
| StrategyTemplate / set_targets | 一次提交三腿及历史合约的完整目标快照，记录方向和标准分数 |
| CtpFuturesBasicProfile / NautilusSimExecutionBackend | 按交易所初始化模拟账户，处理手续费、保证金和真实合约回测 |
| PositionManager / NetTargetOrderPlanner | 维护实际和在途敞口，将完整目标转换为订单 |
| MarketReferencePriceStore / PreTradeRiskManager / RiskLimits | 使用真实价格、乘数、名义金额和报价时效执行交易前检查 |
| DataBinding / ExecutionRoute | 绑定真实合约行情及对应执行路由 |
| UnifiedStrategyRunner / NautilusMarketFeedAdapter / UnifiedHistoricalRuntime | 串联行情、策略和模拟执行，管理历史回放生命周期 |
