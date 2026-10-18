# 中金所股指期权 Delta 双腿买入回测

## 1. 使用方法

### 启动命令

从 pro 根目录、在已安装运行依赖的环境中执行，先设置 `export PYTHONPATH=.`。入口自动读取已有 `.env` 配置；没有配置数据路径时，可通过可选的 `--data-root` 或各文件路径参数指定。示例日期需与实际数据匹配。

**完整执行（显式设置常用参数）：**

```bash
python demos/09_option_delta/run_backtest.py \
  --product MO \
  --start-day 2026-09-01 --end-day 2026-09-11 \
  --quantity 1 \
  --bar-timestamp end --log-level WARNING --no-tearsheet
```

**默认执行（仅必填参数）：**

```bash
python demos/09_option_delta/run_backtest.py \
  --start-day 2026-09-01 --end-day 2026-09-11
```

默认执行采用参数表中的默认值及已有数据路径配置；例如默认启用绩效图，不要求必须产生成交。

### 参数说明

“必选”表示启动时必须传入；“可选”表示有默认值或可由已有配置解析。数据路径虽然是可选参数，但运行前必须能解析到有效文件。

| 参数 | 必选 / 可选 | 含义与默认值 |
| --- | --- | --- |
| `--start-day / --end-day` | 必选 | 必填，请求日期闭区间 |
| `--product / --model` | 可选 | 品种默认 MO；模型默认 bs，可选 black76 |
| `--index-code / --future-product` | 可选 | 覆盖对应定价指数和期货，不提供跨标的对冲 |
| `--data-root` | 可选 | CTP 根目录，下含 role 与 kline/opt、kline/index、kline/fut |
| `--opt-dir / --index-dir / --fut-dir` | 可选 | 显式指定各资产行情目录，优先于环境配置 |
| `--opt-basic / --fut-basic` | 可选 | 显式指定基础条款文件，后者仅 Black76 必需 |
| `--bar-timestamp` | 可选 | 默认 end；源数据为分钟开始标签时传 start，公共层只转换一次 |
| `--sides / --select-slots` | 可选 | 默认 C,P / 13:58；交易入口要求同时启用 C 和 P |
| `--dte-min / --dte-max / --target-dte` | 可选 | 开仓剩余自然日期限范围与目标，默认 20 / 45 / 32.5 |
| `--expiry-month / --min-remaining-days` | 可选 | 显式 YYYYMM 可绕过期限窗口；最低开仓剩余自然日默认 7 |
| `--delta-min / --delta-max / --target-delta` | 可选 | 核心绝对 Delta 区间与目标，默认 0.25 / 0.30 / 0.275 |
| `--fallback-min / --fallback-max / --no-fallback` | 可选 | 无核心候选时在 0.225–0.325 回退，可关闭 |
| `--rate / --dividend-yield` | 可选 | 默认 0.02 / 0，年化小数 |
| `--min-open-interest / --min-volume` | 可选 | 默认 1 / 0；允许零成交分钟选约，可传 min-volume 1 排除 |
| `--max-quote-age-seconds` | 可选 | 默认 60 秒，完整分钟信号帧的最大迟到限制；不能证明真实最后成交时间 |
| `--quantity` | 可选 | 每腿目标手数，默认 1，正整数 |
| `--close-remaining-days / --flatten-slot` | 可选 | 默认 2 个自然日 / 14:50；退出阈值须小于最低开仓剩余天数 |
| `--max-market-age-seconds` | 可选 | 下单相关腿的行情年龄限制，默认 300 秒 |
| `--starting-balance / --commission` | 可选 | 模拟初始资金默认 1000000，固定每手手续费默认 1 |
| `--max-notional` | 可选 | 每合约订单及持仓权利金名义金额上限，默认 10000000，按价格×乘数×手数计算 |
| `--option-margin-init / --option-margin-maint` | 可选 | 默认 1 / 1，基础权利金名义金额保证金近似 |
| `--log-level / --no-tearsheet` | 可选 | 默认 WARNING；可跳过绩效图 |
| `--require-selection / --require-fills` | 可选 | 可要求至少一次完整双腿选约 / 至少一笔成交 |
| `--report-dir` | 可选 | 默认本目录 results，每次以运行 UUID 创建子目录 |

### 数据要求

需要 opt_basic.feather、期权与指数行情；Black76 另需同月期货基础条款和行情。MO 对应 IM / 000852，IO 对应 IF / 000300，HO 对应 IH / 000016。tickNum 为每个期权的最小变动价位，缺失或为空时兼容 minChgPriceNum / price_increment。交易期权需提供 OHLCV、datetime、open_interest，指数及定价期货需 datetime、close；持仓旧腿的后续行情也应覆盖。末日指数须在清仓时点后至少还有两分钟行情。

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
| signals.csv | 双腿买入、换仓、行情等待与退出行为记录 |
| selections.csv / decisions.csv / candidates.csv | Delta 选约结果、时点决策、候选及拒绝原因 |

## 2. 策略详细说明

### 策略思路与模型用途

这是按期限和绝对 Delta 选择虚值 Call、Put 并同时买入的双腿策略。两腿通常具有不同的行权价，可理解为虚值宽跨式买入；它不根据价格方向选择单边，也没有“预测波动率高于 IV 才买入”的择时过滤。

Delta 用于选择距离标的适当的合约，不是要求组合 Delta 为零。Call 的 Delta 为正、Put 为负，分别靠近相同绝对目标；相等手数不保证实际 Delta 完全抵消。IV、Gamma、Vega 用于计算和审计，本版没有按它们动态分配数量。

| 模型 | 定价输入与交易用途 |
| --- | --- |
| BS（Black–Scholes） | 使用指数现价、利率、股息率和期限，结合真实期权价格反解 IV，再计算 Delta |
| Black76 | 使用所选月份的同月期货价格定价；指数仍用于判断 Call/Put 是否虚值 |
| 两种模型共同点 | 只交易认购和认沽；指数不交易，期货不建立对冲仓位 |

### 期限选择与两侧排名

默认每日 13:58 选约，期限统一为距离最后交易日的自然日，不使用交易日计数。

| 步骤 | 当前规则 |
| --- | --- |
| 确定月份 | 从完整静态条款中选生命周期有效、至少剩余 7 日且在 20–45 日窗口内的月份 |
| 月份排序 | 距目标 32.5 日最接近优先；相同时较早最后交易日、月份代码优先 |
| 显式月份 | `expiry-month` 指定月份可绕过 20–45 日窗口，但仍须满足最低剩余日数 |
| 虚值筛选 | Call 行权价 > 指数；Put 行权价 < 指数 |
| 行情筛选 | 必须是信号分钟报价、正且有限的 close、有效成交量与持仓量，并满足流动性门槛 |
| 风险指标 | 权利金通过模型边界校验且能反解 IV，才进入 Delta 候选 |
| 核心区间 | 两侧分别在绝对 Delta 0.25–0.30 中选约，目标 0.275 |
| 回退区间 | 某侧没有核心候选时，可在该侧 0.225–0.325 中回退；可关闭 |
| 侧内排序 | 绝对 Delta 距离较小、持仓量较大、成交量较大、代码较小依次优先 |

月份先由静态期限确定，再筛选该月两腿。目标月缺报价或缺一侧不会改选另一月份补腿。默认持仓量至少 1，成交量允许为零；零成交分钟符合选约规则不代表真实市场能够按该价格成交。

### 组合更新、等待与退出

| 场景 | 仓位处理 |
| --- | --- |
| 完整选出 Call 和 Put | 两腿分别目标 `+quantity`，默认各一手，其余历史期权目标为零 |
| PARTIAL / NO_SELECTION | 不新增单腿，保留已有组合；既有到期或末日退出逻辑仍独立生效 |
| 再次选中相同组合 | 保持相同目标，不每日叠加手数 |
| 旧组合含新组合不再需要的持仓 | 先把组合目标全置零；实际旧仓归零且无在途订单后再推进新组合 |
| 仍有在途订单 | 等待回报，不提前发下一阶段目标 |
| 下单相关腿缺新鲜报价 | 记录 WAIT_FRESH_PRICES，保留真实持仓和已提交目标，等待合格行情 |
| 任一相关腿距最后交易日不超过 2 日 | 发起整个组合退出；实际平仓仍受报价与成交门控 |
| 最后回测日到清仓时刻 | 默认 14:50 起只退出，不再开仓；结束检查实际仓位和订单残留 |

这里的“先平后开”由实际仓位和在途数量推进，不以删除持仓字典或发送零目标作为平仓完成证明。未持有且无需变化的历史合约不要求全部同时有新报价。订单被拒绝、取消或过期时停止，并保留诊断信息。

### 决策时序与结果核对

新分钟到达后评价上一完整分钟，选约帧迟到超过默认 60 秒则不据此发新交易。交易报价年龄单独采用默认 300 秒限制；两个参数分别约束信号帧时序与下单报价，不能互相替代。同一期权行先发原生 Bar 推进撮合，再发 CustomBar 提供信号，避免重复撮合。

最后一帧没有后续行情时不在停止回调补单。完整双腿目标不保证原子成交，先平旧仓也可能出现阶段性空仓。本版没有独立止盈止损或期货对冲，退出主要由组合更新、临近到期及末日清仓决定。

验收时按 `decisions / selections` 核对期限、虚值与 Delta，再按 `signals / orders / fills` 核对双腿买入和换仓阶段，最后用实际持仓、在途数量与清仓摘要确认完整交易闭环。选约成功只证明选出合约，不能替代真实模拟成交。

## 3. 使用核心组件描述

### dataprep：文件与交易行情准备

| 关键接口 | 本例用途 |
|---|---|
| input_session / write_input_reports | 共享读取缓存、记录版本与诊断，输出三类输入报告 |
| resolve_option_args | 统一路径，BS 要求期权/指数，Black76 另要求期货 |
| load_options_basic / load_cffex_futures | 校验条款与 tickNum；静态期权全集用于期限选择 |
| inventory / BarFileKey | 定位资产、合约和交易日，保留可能持有旧腿的后续行情 |
| BarReadSpec | 期权 execution_strict OHLCV 加持仓量；指数/期货 close_strict |
| plan_option_chain / prepare_option_chain | InputPlan 校验、准备 HistoricalInputBundle 并记录覆盖 |
| PreparedFrameReader | 直接消费已经校验的行，不重新读取原文件 |

### datahub：参考资料查询

| 关键接口 | 本例用途 |
|---|---|
| 本例未装配 DataHub Store | 使用公共层静态条款全集，不查询主力角色或复权因子 |

### market：信号与真实行情回放

| 关键接口 | 本例用途 |
|---|---|
| InstrumentId / InstrumentMeta / FileReplayFeed | 注册真实期权与信号标识，按时间回放已准备行情 |
| FixedInstrumentBarParser / ParsedEvent | 将严格 OHLCV 转成真实原生 Bar，一次推进撮合 |
| make_custom_bar | 携带 close、持仓量、成交量、时间等信号字段，不作为另一份执行行情 |

### trader：模拟交易与策略生命周期

| 关键接口 | 本例用途 |
|---|---|
| instrument_factory.make_option / instrument_meta | 按 C/P 创建真实认购/认沽及行情元信息；默认认购保持 Vega 调用兼容 |
| CtpFuturesBasicProfile / NautilusSimExecutionBackend | 注册 CFFEX 模拟账户、资金、手续费与真实期权合约 |
| StrategyTemplate.set_targets / account_position / working_quantity | 提交完整目标，按实际成交和在途订单推进换仓 |
| PositionManager / NetTargetOrderPlanner / SimulationExecutionClient | 从目标生成净持仓订单，接收执行回报并更新实际持仓 |
| MarketReferencePriceStore / PreTradeRiskManager / RiskLimits | 价格年龄、数量、持仓及权利金名义金额限制 |
| DataBinding / ExecutionRoute / UnifiedStrategyRunner | 分发选约事件，将期权目标路由到模拟客户端 |
| NautilusMarketFeedAdapter / UnifiedHistoricalRuntime | 原生 Bar 先推进模拟撮合，再由策略产生新目标，统一回测生命周期 |
