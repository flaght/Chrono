# 产业链策略 主力信号与次主力执行

## 1. 使用方法

### 启动命令

从 pro 根目录、在已安装运行依赖的环境中执行，先设置 `export PYTHONPATH=.`。入口自动读取已有 `.env` 配置；没有配置数据路径时，可通过可选的 `--data-root` 或各文件路径参数指定。示例日期需与实际数据匹配。

**完整执行（显式设置常用参数）：**

```bash
python demos/02_sector_chain/run_backtest.py \
  --start-day 2026-01-05 --end-day 2026-05-29 \
  --leader-products JM,I --comparison-product RB \
  --signal-role main --execution-role secondary \
  --quantity 1 --return-period 30 --sector-period 15 \
  --bar-timestamp end --log-level WARNING --no-tearsheet
```

**默认执行（仅必填参数）：**

```bash
python demos/02_sector_chain/run_backtest.py \
  --start-day 2026-01-05 --end-day 2026-05-29
```

默认执行采用参数表中的默认值及已有数据路径配置；例如默认启用绩效图，不要求必须产生成交。

### 参数说明

“必选”表示启动时必须传入；“可选”表示有默认值或可由已有配置解析。数据路径虽然是可选参数，但运行前必须能解析到有效文件。

| 参数 | 必选 / 可选 | 含义与默认值 |
| --- | --- | --- |
| `--start-day / --end-day` | 必选 | 请求日期闭区间 |
| `--leader-products` | 可选 | 两个不同领头品种，逗号分隔，默认 JM,I |
| `--comparison-product` | 可选 | 与行业参考值比较的第三个品种，须与领头品种不同，默认 RB |
| `--execution-product` | 可选 | 默认跟随比较品种，可显式指定三个信号品种中的任意一个 |
| `--signal-role / --execution-role` | 可选 | 默认 main / secondary；显式改信号角色时必须有对应角色因子 |
| `--return-period` | 可选 | 两领头品种单根收益率的滚动平均窗口，默认 30 个完整同步 Bar |
| `--sector-period` | 可选 | 两领头均值的平均数再做滚动平均，默认 15 个值 |
| `--quantity / --submission-delay-bars` | 可选 | 目标绝对手数，及提交前等待的执行合约 Bar 数，默认 1 / 0 |
| `--bar-timestamp` | 可选 | 源分钟标签 start / end，默认 end |
| `--bars-dir / --contract-struct / --fut-basic / --factors` | 可选 | 覆盖行情、角色表、基础条款及累计因子路径 |
| `--data-root / --factor-availability` | 可选 | 根目录推导；因子默认 aligned，explicit 强制要求发布时间 |
| `--starting-balance / --commission-per-contract / --margin-init / --margin-maint` | 可选 | 模拟资金、手续费和保证金配置 |
| `--max-notional / --max-market-age-seconds` | 可选 | 名义金额和参考报价年龄限制，默认 1,000,000 / 120 秒 |
| `--report-dir / --no-tearsheet / --log-level / --require-fills` | 可选 | 输出目录、跳过图表、日志等级及至少一笔成交检查 |

### 数据要求

需要三个信号品种的主力角色、主力累计因子与同步行情，以及执行品种的次主力和换约日旧执行合约行情。主力累计因子使用 trade_date、code、symbol、pcr_cumfactor，冲突以同角色因子 symbol 为准；次主力成交不需要复权因子。更换产业链可直接修改 leader-products 与 comparison-product，例如 TA,MA 与 PP；执行品种默认跟随比较品种。

期货角色表默认查找 fut_contract.feather，缺失时兼容 fut_contract_data.feather；两者同时存在需通过 --contract-struct 选择。基础条款默认 fut_basic.feather，行情为 `<symbol>_YYYYMMDD.feather`。

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

### 策略思路与信号公式

这是可配置三品种的产业链相对强弱策略。两个领头品种构成平滑参考，比较品种当前收益率高于参考时做多、低于参考时做空。默认只交易比较品种的次主力，领头品种只提供信号；执行品种也可以改为三个信号品种中的另一个。

设两个领头为 A、B，比较品种为 C，复权主力价格为 P。只在三个主力同一时间戳到齐时计算：

```text
r_A(t) = P_A(t) / P_A(上一完整同步帧) - 1
r_B(t) = P_B(t) / P_B(上一完整同步帧) - 1
r_C(t) = P_C(t) / P_C(上一完整同步帧) - 1
领头参考(t) = [MA_return_period(r_A) + MA_return_period(r_B)] / 2
产业链参考(t) = MA_sector_period(领头参考)
```

| 条件 | 目标规则 |
| --- | --- |
| `r_C > 产业链参考` | 执行品种目标 `+quantity` |
| `r_C < 产业链参考` | 执行品种目标 `-quantity` |
| 两者相等 | 保留上一次方向；初始方向为零时继续等待 |
| 窗口未满 | 不提交方向目标 |

第一帧仅保存价格；之后积累收益率。首次可计算方向需要 `return_period + sector_period` 个完整价格帧，默认 45 帧。比较品种使用当前单帧收益率，两个领头使用滚动平均；这不是三个品种各自 MA 后互相比大小。缺失帧不补价，因此单帧收益率可能跨越多分钟或休市区间。

### 主力信号与次主力执行

| 环节 | 规则 |
| --- | --- |
| 信号复权 | 三个品种各自的主力真实 close × 各自主力累计因子 |
| 成交报价 | 执行次主力的真实未复权行情；无需次主力复权因子 |
| 提交等待 | 收到不早于信号时刻的执行合约 Bar；可再等待指定数量的后续执行 Bar |
| 同方向去重 | 已提交方向和执行合约均相同时不重复提交 |
| 信号覆盖 | 新方向可替换待执行方向；回到已提交方向时撤销相反的待执行计划 |
| 执行换约 | 动态路由处理旧、新次主力；待执行计划转到新执行合约并重新等待其行情 |
| 停止回放 | 未能获得后续执行行情的计划记为过期，不在停止回调补单 |

该策略没有双品种对冲仓位、独立止盈止损或末日强制平仓。参数中的产业链名称不改变算法，也不验证经济领头关系。验收时先核对完整帧数量和 `comparison_return / sector_ma`，再检查提交等待记录、真实执行 symbol 和成交，确认主力研究信号最终落在指定次主力。

## 3. 使用核心组件描述

### dataprep：输入准备

| 关键接口 | 本例用途 |
|---|---|
| input_session / read_feather | 共享输入缓存、文件版本和诊断 |
| resolve_futures_args | 解析期货目录、角色表和基础条款 |
| prepare_role_research | products 为配置的三个品种，signal_role=main，execution_role=secondary；只要求主力研究因子，保留执行次主力映射 |
| futures.instrument | 校验条款并构造真实合约、行情元信息和乘数 |
| BarReadSpec / bar_paths / add_bar_source | 标准化并注册信号主力、执行次主力及换约旧执行合约的真实行情 |
| write_input_reports | 输出输入版本、覆盖和问题，记录因子合约覆盖 |

### datahub：参考资料查询

| 关键接口 | 本例用途 |
|---|---|
| SectorRoleAssignment / SectorRoleStore | 保存最终日级合约和累计因子，按事件时间查询 |
| snapshot / instrument | 给出三品种主力信号合约，以及执行品种次主力合约 |
| factor(product, "main") | 提供同品种主力累计因子，供主力研究价使用 |

### market：真实行情回放

| 关键接口 | 本例用途 |
|---|---|
| InstrumentMeta / DataType.BAR | 描述真实合约及分钟行情类型 |
| FileReplayFeed / 固定合约 Bar 解析器 | 回放真实 OHLCV 和事件时钟；不把研究复权价变成成交价 |

### trader：策略运行和执行

| 关键接口 | 本例用途 |
|---|---|
| StrategyTemplate / set_target | 提交执行角色逻辑目标，例如 rb_secondary、pp_secondary |
| ScheduledContractResolver / DynamicExecutionRoute | 路由到真实次主力，处理执行角色换约 |
| PositionManager / NetTargetOrderPlanner / PreTradeRiskManager | 维护仓位、规划差额订单并检查真实行情风险 |
| NautilusMarketFeedAdapter / UnifiedStrategyRunner / UnifiedHistoricalRuntime | 将回放、策略和后端连接，推进历史运行 |
| NautilusSimExecutionBackend / CtpFuturesBasicProfile | 模拟真实合约成交，使用基础手续费和保证金模型 |
