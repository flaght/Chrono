# 单品种主力 EMA 回测

## 1. 使用方法

### 启动命令

从 pro 根目录、在已安装运行依赖的环境中执行，先设置 `export PYTHONPATH=.`。入口自动读取已有 `.env` 配置；没有配置数据路径时，可通过可选的 `--data-root` 或各文件路径参数指定。示例日期需与实际数据匹配。

**完整执行（显式设置常用参数）：**

```bash
python demos/01_main_ema/run_backtest.py \
  --product RB \
  --start-day 2026-01-05 --end-day 2026-05-26 \
  --fast 3 --slow 5 --quantity 1 \
  --bar-timestamp end --log-level WARNING --no-tearsheet
```

**默认执行（仅必填参数）：**

```bash
python demos/01_main_ema/run_backtest.py \
  --product RB \
  --start-day 2026-01-05 --end-day 2026-05-26
```

默认执行采用参数表中的默认值及已有数据路径配置；例如默认启用绩效图，不要求必须产生成交。

### 参数说明

“必选”表示启动时必须传入；“可选”表示有默认值或可由已有配置解析。数据路径虽然是可选参数，但运行前必须能解析到有效文件。

| 参数 | 必选 / 可选 | 含义与默认值 |
| --- | --- | --- |
| `--product` | 必选 | 品种代码，例如 RB、I、HC；一次运行一个品种 |
| `--start-day / --end-day` | 必选 | 回测日期范围 |
| `--fast / --slow` | 可选 | 快 / 慢 EMA 周期默认 3 / 5，按有效主力 Bar 计数，要求 0 < fast < slow |
| `--quantity` | 可选 | 多空目标手数绝对值，默认 1 |
| `--bar-timestamp` | 可选 | 默认 end，可选 start；start 由公共层转换为完成分钟 |
| `--bars-dir / --contract-struct / --fut-basic / --factors` | 可选 | 显式覆盖行情、角色表、条款及累计因子路径 |
| `--data-root` | 可选 | 按 role/ 和 kline/fut/ 布局推导路径 |
| `--factor-availability` | 可选 | 默认 aligned，使用已对齐资料；explicit 强制要求 available_ns。已有发布时间仍会校验 |
| `--max-notional` | 可选 | 单笔与持仓名义金额上限，默认 1,000,000 |
| `--require-fills` | 可选 | 要求至少一笔模拟成交，否则报错 |
| `--no-tearsheet / --report-dir / --log-level` | 可选 | 跳过绩效图、指定输出根目录（默认本目录 results/）、调整框架日志（默认 WARNING） |

### 数据要求

需要角色表、fut_basic.feather、fut_adjustment_factors.feather 和每日主力行情；换约日还需旧主力行情。因子字段为 trade_date、code、symbol、pcr_cumfactor，按上游已对齐的当天口径使用；合约冲突时以因子 symbol 为准，不再次累计 pcr_factor。

期货角色表默认查找 fut_contract.feather，缺失时兼容 fut_contract_data.feather；两者同时存在需通过 --contract-struct 选择。基础条款默认 fut_basic.feather，行情为 `<symbol>_YYYYMMDD.feather`。

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

### SimNow 接入增量验证

在运行环境中验证当前 `MainEmaStrategy` 与已有 CTP 执行组件的衔接：

```bash
python -u tests/run_main_ema_simnow.py --stage all
```

集中入口按阶段启动独立进程，失败即停止，打印实际 Python、Bomber、策略及公共 Runner 加载路径。单独复测某一模块时，将 `all` 替换为下表中的阶段名；`--timeout` 可选，默认每个阶段或回归脚本 180 秒。

| 阶段 | 验证内容 |
| --- | --- |
| signal | 当前策略的预热、方向、目标去重、合约过滤、角色发布时间及累计因子，共 6 项 |
| routing | LIVE 固定主力路由：默认行为、目标推进和版本、在途及恢复约束、过期目标、行情时间、资料缺失、主力变更闭闸、行情恢复确认及 LIVE 动态路由禁用，共 10 项 |
| ctp | 内存传输下复用原生 CTP Driver、旧限价规划器与受控客户端：LIVE 固定路由验证开仓／平今／反手、最新信号、部分成交／撤单、断线及主力变更关闭授权；HISTORICAL 单独验证跨合约换约，共 7 项 |
| regression | 复用 7 个相关旧回归入口，检查公共修改对既有链路的影响 |

上述验证不连接 SimNow 或发送柜台订单，结果只证明无网络场景。`live_runner.py` 通过 `add_main_strategy` 装配单策略、固定本次主力的 LIVE 路由，保留当前策略仅在方向变化时提交的规则，在新 Bar 决策后显式调用 `continue_execution_target` 推进尚未完成的目标，不增加策略或组合贡献版本。角色变为另一合约时关闭下单授权、请求撤单并标记需恢复；资料不可用时暂停保留目标推进。LIVE 动态换约限制继续保留，不能将 HISTORICAL 换约测试通过解释为 LIVE 换约可用。

该模块不承担柜台连接、交易授权、参考资料加载、预热或跨进程恢复。当前完整 `run_live` 入口及本次柜台交易仍需逐阶段接入和验收。

2026-10-07 远程验证状态：Linux `uv-nautilus` / Bomber 1.230.0 环境下，signal 6/6、修正版 routing 10/10（0.007 秒）、修正版 CTP 7/7（0.021 秒）及 regression 的 7 个相关旧入口全部通过。CTP 首轮 HISTORICAL 换约用例因漏掉旧合约行情推进的撤单阶段而失败，修正事件顺序并补充旧仓未平前禁止开新仓的断言后复验通过。合计 23 项增量用例及 7 个相关回归入口已有远程无网络通过证据，可进入下一模块，已通过的阶段无需重复运行。SimNow 行情与实际柜台交易尚未验收。

下一模块复用既有交易端只读探针。在项目根目录、已激活的 `uv-nautilus` 环境中，使用项目 `.env` 或当前环境变量配置的 SimNow TD：

```bash
python -u -m scripts.integration.ctp.td_readonly --connect --timeout 30
```

该命令执行登录、结算确认和本次仓位／资金／全账户活动订单查询，不发单；读取 `CTP_TD_ADDRESS`、`CTP_BROKER_ID`、`CTP_ACCOUNT_ID`、`CTP_PASSWORD` 及既有认证配置，不在命令中填入凭据。以三项查询完整完成为此模块的验收依据；失败时先反馈完整脱敏输出，不沿用历史柜台快照。通过后继续当前主力 EMA 的实时行情到 Recording 装配，再接受控模拟订单。

通道只读入口已归入 [scripts/integration/ctp](../../scripts/integration/ctp/README.md)，迁移后的配置定位与旧入口转发先通过 `python -u tests/run_ctp_td_readonly.py` 做无网络烟测，再连接柜台。目录调整见[协调记录 PF2026100702](../../doc/HANDOFF/HANDOFF_PARALLEL_BACKTEST_LIVE_2026-10-07.md#pf2026100702-通道联调入口归入-scripts-integration)。

本轮同步至少包含 `trader/runner.py`、`demos/01_main_ema/live_runner.py` 和 `tests/run_main_ema_simnow.py`；当前策略及旧 CTP 测试依赖也须与本次源码版本一致。公共变更及验证状态见[并行协调记录 PF2026100701](../../doc/HANDOFF/HANDOFF_PARALLEL_BACKTEST_LIVE_2026-10-07.md#pf2026100701-保留目标的分阶段执行推进)。

## 2. 策略详细说明

### 策略思路与交易对象

这是单品种趋势方向策略：用快、慢 EMA 的相对位置决定持仓方向，信号与交易均对应当前主力。它判断的是当前均线关系，预热结束后即使没有观察到一次交叉，也可以建立首笔仓位。

| 项目 | 具体规则 |
| --- | --- |
| 研究价格 | 当根主力真实收盘价 × 当天 `pcr_cumfactor` |
| 有效事件 | 事件时刻快照可用，合约与交易所匹配当前主力，时间晚于上一有效主力 Bar |
| 信号周期 | `fast`、`slow` 均按有效主力分钟 Bar 计数；不是自然分钟或交易日 |
| 预热 | 慢 EMA 的 `initialized` 为真后才允许提交目标 |
| 多头 | 快 EMA ≥ 慢 EMA，目标 `+quantity` |
| 空头 | 快 EMA < 慢 EMA，目标 `-quantity` |
| 目标更新 | 仅当目标方向变化时重新提交；同方向不会逐根加仓 |

每根有效价格同时更新两条 EMA，指标计算由 Bomber 的 `ExponentialMovingAverage` 实现。缺少可用角色快照时跳过并计数；旧主力 Bar 和重复时间事件不推进指标。

### 仓位、换约与退出

策略提交的是 `rb_main` 等逻辑目标，动态执行路由将它解析到真实月份合约。若实际持仓为 +1 手、目标变为 -1 手，执行层根据持仓差额规划反向交易；策略并非只发一手卖单。

主力切换时保留两条 EMA 和此前方向，通过累计因子保持研究序列的连续口径；执行路由协调旧主力退出和新主力仓位。换约日的旧合约行情用于执行，不参与新主力的 EMA。因子表与角色表冲突时，数据准备层采用因子表合约，避免研究与执行身份不一致。

本例没有中性区、独立止盈止损或末日强制平仓。方向改变是主要退出及反向条件，回测末尾可能仍持仓。检查结果时应同时查看有效 Bar 数、最终主力、目标方向及 `fills.csv`、`positions.csv`，不能仅凭最终 target 判断成交。

## 3. 使用核心组件描述

### dataprep：输入准备

负责加载和校验文件，把本例需要的资料及真实行情准备好，不计算 EMA、不启动回测。

| 关键接口 | 本例用途 |
|---|---|
| input_session / read_feather | 共享本次运行的读取缓存、文件版本和输入诊断 |
| resolve_futures_args | 返回 DataPaths，定位行情目录、角色表和基础条款 |
| prepare_role_research | 只准备当前品种的 main 信号与执行角色；加载当天累计因子，合约冲突时以同角色因子 symbol 为准，组装 DataHub Store |
| futures.instrument | 校验真实条款，调用 trader 合约工厂，返回原生合约、行情元信息和乘数 |
| BarReadSpec / bar_path / add_bar_source | 定位必需文件，统一一分钟时间标签、严格校验 OHLCV，将准备好的行注册到 market Feed |
| write_input_reports | 写出输入版本、日期覆盖和问题，便于对照交易报表 |

### datahub：参考资料查询

负责保存 dataprep 准备的参考资料，按策略事件时间查询；不回放行情、不生成订单。

| 关键接口 | 本例用途 |
|---|---|
| SectorRoleAssignment | 保存交易日、角色来源日、生效及可用时间、最终真实合约映射和累计因子 |
| SectorRoleStore | 管理日级快照，供策略按时间查询；文件加载及冲突覆盖由 dataprep 完成 |
| snapshot(timestamp) | 查询事件时刻已生效且可用的角色快照 |
| instrument(product, "main") | 从快照取得当前品种的最终真实主力合约 |
| factor(product, "main") | 从快照取得当前主力的累计复权因子，供 EMA 研究价格使用 |
| SectorDataUnavailable | 参考资料尚不可用时通知策略跳过事件并计数，不补因子或生成目标 |

### market：行情定义和回放

负责真实行情的数据结构、解析和文件回放；保留真实价格，复权只在策略计算信号时使用。

| 关键接口 | 本例用途 |
|---|---|
| InstrumentMeta | 描述真实合约的价格精度、最小变动、乘数、币种和交易所 |
| DataType.BAR / Bar | 声明分钟行情类型，向策略提供真实 OHLCV、合约身份和事件时间 |
| FileReplayFeed | 管理并回放 dataprep 注册的数据源；换月当日包含旧、新主力行情 |
| FixedInstrumentBarParser / MappedBarParser | 将准备好的行解析为固定真实合约 Bar，映射事件时间和可用时间 |

当前主力 Bar 用于 EMA，旧主力 Bar 用于换月执行；连接行情与模拟后端的 NautilusMarketFeedAdapter 属于 trader。

### trader：策略运行和模拟执行

负责把行情、策略目标、真实合约路由、仓位和模拟后端连接起来。

| 关键接口 | 本例用途 |
|---|---|
| StrategyTemplate | 策略基类；set_target 提交逻辑目标仓位，例如 rb_main |
| ScheduledContractResolver / DynamicExecutionRoute | 按生效时间映射真实主力，协调换月旧仓处理与新合约目标 |
| PositionManager / NetTargetOrderPlanner | 维护仓位，根据目标与当前有效仓位的差额生成订单 |
| MarketReferencePriceStore / PreTradeRiskManager | 维护真实参考价，检查手数、名义金额和行情年龄 |
| NautilusMarketFeedAdapter | 将 market 行情连接策略回放与模拟后端；该适配器属于 trader |
| UnifiedStrategyRunner / UnifiedHistoricalRuntime | 注册数据和执行绑定，推进历史运行，管理停止清理 |
| instrument_factory / NautilusSimExecutionBackend / CtpFuturesBasicProfile | 构造原生合约并模拟执行；本例起始资金 1,000,000、每手手续费 1，采用基础保证金模型 |
