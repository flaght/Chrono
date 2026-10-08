# 公共本地输入层开发说明

2026-10-08 新增 [参考资料数据源工厂](sources/README.md)：统一文件和 DolphinDB 字段适配，MySQL／MongoDB 保留子类扩展契约；原文件加载入口保留，在线组合器独立于历史场景。新增代码及数据库联调待远程验证。

2026-10-05 公共包正式命名为 `dataprep`，代码示例、策略引用及测试模块已同步。内部 Input* 类型、input_* 报告文件名和诊断 schema_version 保持兼容，本次只调整包名及引用。

本模块实现 demos 输入整理的第一阶段：将路径、行情索引、读取与标准化、静态条款、交易日历、角色研究和目标计划集中到公共组件。各策略只声明需求、转换既有业务对象并组装对应场景。现有 Feed、Parser、DataHub 和 Trader 继续承担原有职责。

2026-10-05 补充外部累计因子接口 dataprep.factors：主力 EMA 与黑色板块入口传入 factors_path 并直接使用 pcr_cumfactor，不再自计算。用户确认资料已按使用当天对齐，两入口固定 factor_date_basis="trading"、默认 aligned，不再要求命令行选择日期；有显式 available_ns 时仍校验。角色合约冲突以同角色因子表 symbol 为准，并写覆盖诊断；信号因子缺失、无效或尚未发布仍失败。其他未迁移场景保留原模式。新接口未运行测试，历史 80 项通过和 EMA 成功结果属于切换前证据。

本地仅做代码生成与静态核对，不执行程序。2026-10-05 用户运行环境已通过 80 项 unittest 方法及三个规则脚本，并完成 RB 单品种主力 EMA 的真实行情模拟回测：16 个交易日、5,493 根 Bar、2,275 笔订单和成交、unavailable=0。此结果证明该样本链路可运行，尚不代表换月行为、逐笔交易正确性或九策略整体签收。实施记录见 [第一阶段交接文档](../doc/DEMOS_INPUT_STAGE1_IMPLEMENTATION_2026-10-04.md)。

逐函数参数、返回值、调用顺序和单品种 main 场景边界见 [dataprep 函数使用说明](API_USAGE.md)。当前主力 EMA 直接使用 prepare_role_research，不必先构造 DataRequirements 或 HistoricalInputBundle；这两个契约保留给声明式组装及第二阶段运行工厂。

## 模块职责

| 模块 | 提供的能力 | 边界 |
|---|---|---|
| contracts | DataPaths、BarFileKey、BarReadSpec、DataRequirements、InstrumentSpec、InputPlan、InputContext、HistoricalInputBundle、InputIssue | 仅公共值契约；导入不依赖 pandas 或原生交易环境 |
| paths | resolve_paths、resolve_futures_args、resolve_option_args、resolve_targets_path | 配置与来源记录；不选择交易合约 |
| catalog | scan_bar_files、select_bar_files、bar_paths、files_for_symbol、bar_files、inventory | 文件身份与必需或可选缺失；不因文件存在改选月份 |
| bars | read_bar_frame、PreparedFrameReader、add_bar_source、add_prepared_bar_source | 标准化后的行交给现有 Feed.add_source；不启动回放 |
| metadata | load_futures_basic、load_options_basic、load_cffex_futures、contract_rows | 使用 DataHub 基础条款模型校验，保持静态 InstrumentSpec / 原始行返回接口；不配仓 |
| basic | load_future_basic_provider、load_option_basic_provider | 读取版本化基础条款，共用输入缓存，按调用方声明的时间规则组装两类 Provider |
| calendar | load_calendar、TradingCalendar、infer_market_calendar | 显式日历或带前提的文件日期日历 |
| references | load_role_assignments、load_sector_research、load_role_research | 转换为现有 DataHub Store；不生成可成交复权 Bar |
| schedules | load_target_csv | 转换为现有 TargetScheduleStore；不处理时钟或订单 |
| session | InputSession、input_session、write_input_reports、read_feather | 单次运行缓存、文件变化检查、诊断输出 |
| scenarios | fixed_contracts、role_futures、option_chain | 公共场景组装；不依赖具体 demo 的配置或信号 |
| futures | instrument、make_instrument | 旧入口的期货条款薄转换，调用 trader 公共工厂 |

原生 Instrument 工厂位于 `trader/instrument_factory.py`。它不依赖 dataprep 或 demos。`make_profile_future` 保持原有 Profile 的 UTC 挂牌日与最后交易日次日到期约定；`make_option` 保留 Vega 的 CFFEX 认购工厂，不能据此认定所有期权类型已经支持执行。

## 路径配置

同一个配置项按显式参数、config 字典、新环境变量、CTP 根目录推导、旧环境变量和资料默认文件名的顺序处理。专用资产目录优先于通用 kline 父目录；显式 kline 父目录和新 `KLINE_DATA_DIR` 可覆盖根目录推导。相对输入路径按 pro 项目根目录解析。

| 输入 | 显式字段及兼容别名 | 新环境变量 | 根目录推导 | 旧变量 |
|---|---|---|---|---|
| 资料目录 | role、role_dir | FUT_ROLE_DATA_DIR | CTP_DATA_DIR/role | ROLE_DIR |
| 期货行情 | fut、fut_dir、bars_dir | FUT_KLINE_DATA_DIR | CTP_DATA_DIR/kline/fut | KLINE_DIR |
| 期权行情 | opt、opt_dir | OPT_KLINE_DATA_DIR | CTP_DATA_DIR/kline/opt | KLINE_DIR |
| 指数行情 | index、index_dir | INDEX_KLINE_DATA_DIR | CTP_DATA_DIR/kline/index | KLINE_DIR |
| 行情父目录 | kline、kline_dir | KLINE_DATA_DIR | 使用资产目录推导 | KLINE_DIR |
| 日历文件 | calendar | CTP_CALENDAR_PATH | 不默认生成 | 无 |

资产专用路径的 `layout` 可为 auto、direct、parent。auto 在存在对应 fut、opt、index 子目录时使用子目录；无法判定时按直接目录处理。该配置只作用于显式提供的资产路径，不会对已推导的子目录再次拼接。kline 字段与 KLINE_DATA_DIR 明确表示父目录。旧 KLINE_DIR 保留直接目录与父目录兼容。

只校验本场景要求的目录和文件。BS Delta 不强制期货目录或 fut_basic。目标计划不要求角色表。资料目录与行情目录不能相同。

角色表默认 `fut_contract.feather`，仅其不存在时回退 `fut_contract_data.feather`。两个文件同时存在且场景需要角色表时，必须显式给 contract_struct。fut_basic 和 opt_basic 默认位于 role 目录。API 的 `validate=False` 只供配置检查与夹具使用，正式入口仍校验路径。

## 时间与字段契约

`BarReadSpec` 必须表达源分钟标签 start 或 end、时区、周期、交易日政策、字段要求和坏值政策。期货入口新增 `--bar-timestamp start|end`，默认 end 兼容既有调用；两个期权入口保留原有显式时间参数。

公共输出保留 `source_timestamp`，将 datetime、event_ns、bar_ns 统一为完成分钟时间。start 只加一次周期，end 保持不动。纳秒由 Timestamp.value 取得，避免 Feather 微秒单位被误当纳秒。要求整分钟对齐，重复分钟失败。Parser 不再移动标签。

现有 demo 的绑定周期仍为 1-MINUTE，公共执行注册只接受 60 秒规格。Parser 可显式映射可用时间字符串或整数纳秒列；规范化输入使用 available_ns，避免字符串解析将亚微秒可用时间截断。

未指定 available_column 时，假设完成时刻即可用，并写入覆盖报告。指定时保留 available_ns 和 available_datetime，必须不早于完成时刻。公共执行 Parser 将其映射到 ts_init；Feed 已按 ts_init 排序。单文件按到达时间排序后会导致 event_ns 倒退时，公共层以 UNSUPPORTED_CAPABILITY 拒绝。九策略仍沿用现有跨源同步算法，不能据此声称支持任意延迟轨迹。

`day_session` 要求源自然日、完成自然日与声明交易日一致，用于股指日盘。`exchange` 信任上游按交易日归属的文件身份，允许商品夜盘；有 trade_date 时校验其一致性。缺少交易日历时不能精确审计节假日前后的场次归属，因此报告明确记录这一前提，不使用固定自然日偏移猜测夜盘。带日期的目录与文件名必须一致。

| value_policy | 必需字段 | 值处理 |
|---|---|---|
| execution_strict | 至少 OHLCV | 非有限或非正价格、OHLC 关系错误、负或非整数分钟 volume 失败 |
| close_strict | 由调用方声明，常为 close | 无效 close 失败；适用于角色研究与标的价格 |
| research_audited | Delta 的 close、open_interest、volume | 保留坏值和 InputIssue、input_quality，由既有选约核拒绝候选 |

volume 声明为分钟增量，累计量必须先按明确的导出规则转换；公共层不猜测重置、不重复差分。零成交分钟保留并标记 zero_volume_bar。不会前填价格、补零 Bar、复制旧报价，也不把零成交理解为最后成交时间已更新。

研究 SourceSpec 不允许通过 add_prepared_bar_source 变成执行 Bar。Delta 仍使用原有研究 CustomBar 载体，真实 close 保留在 factors 中，载体价格不送撮合。

## 场景组装接口

固定合约场景是两步：先由业务决定真实合约与必需或可选日期，再由公共层准备输入。

```python
from datetime import date
from bomber.framework.dataprep import BarFileKey, BarReadSpec, InputSession
from bomber.framework.dataprep.scenarios import plan_fixed_contracts, prepare_fixed_contracts

with InputSession() as session:
    plan = plan_fixed_contracts(
        required=(BarFileKey("future", "RB2610", date(2026, 9, 11)),),
        optional=(BarFileKey("future", "RB2609", date(2026, 9, 11)),),
        requested=(date(2026, 9, 11), date(2026, 9, 11)),
    )
    session.coverage.requested = plan.requested
    bundle = prepare_fixed_contracts(paths.fut, plan,
        spec=BarReadSpec(timestamp_label="end"))
    # 调用层准备已有 Feed 和原生 Instrument 后：
    # add_prepared_bar_source(feed, bundle.sources[0].result, instrument.id)
```

上述是开发用例，未在本轮执行。Bundle.context 保存公共事实与服务，sources 引用规范化行，bindings 是声明而非原生 DataBinding；阶段二负责运行层转换。

角色场景的 `prepare_role_research` 兼容当前入口返回现有 LoadedSectorResearch 或 LoadedRoleResearch。参数 products、signal_role、execution_product、execution_role 声明角色需求；minute_roles 非空时要求一个品种及 end_day，返回分钟研究 Store。日终锚点和分钟研究价格保持两种不同需求。

需要完整历史 Bundle 时使用 `plan_role_futures` 与 `prepare_role_futures(plan=..., **research_args)`；调用方先确定执行合约集合，公共层不会替策略选执行腿。角色 Store 沿用 previous_common、最多回看一个共同交易日的原有因果因子政策。只加载声明的角色，main 场景不再要求无关 recent。具有延迟发布时间的角色不能提前用于首个决策，无法构造首个可见快照时明确失败。

`ObservedClose` 当前没有可用时间字段，因此延迟发布的研究收盘价或换月锚点明确拒绝；其 publication-aware Provider 属于后续能力。跨期旧业务模型也没有角色发布门控，本阶段拒绝携带 available_ns 的该类资料。静态条款的 available_ns 或 source_version 如非空同样拒绝，直到接入按可用时间查询的元数据服务，避免最终版本回灌历史。

期权链的 `plan_option_chain` 与 `prepare_option_chain(paths, plan, specs=..., context=...)` 支持按资产指定字段及研究或执行政策。兼容入口用 `read_research_frame` 和 `read_execution_frame` 组合。公共层不做 DTE 选月、Delta 排名、Vega 配仓和对冲资格筛选。需要日历时可将 TradingCalendar 或既有日期序列放入 InputContext.calendar。

## 缓存与诊断

`@input_session` 包裹当前入口，嵌套调用复用同一会话，退出清理缓存。目录清单按根目录扫描一次，不同资产身份由同一清单转换；Feather 按绝对路径、size、mtime_ns 及投影缓存，规范化结果额外按身份和规格缓存。返回帧深拷贝，策略侧改动不会污染缓存。

运行中已读取文件 size 或 mtime_ns 变化会报 SOURCE_CHANGED。该检查不是文件内容哈希；目录中的新增或删除也不是实时监控。缓存生命周期受单次运行约束，没有持久缓存或内存容量上限；真实样本峰值内存和性能仍需验收。

成功报告目录增加 input_manifest.json、input_coverage.json、input_issues.json。manifest 记录实际读取文件版本、路径来源和扫描或读取次数。coverage 区分请求内 actual_days 与区间外 dependency_days，保存必需或可选缺失、时间假设和日历来源。文件出现的日期不是交易所完整覆盖证明，休市与缺文件仍需显式日历确认。

当调用方提供 report_dir 时，失败尝试输出 input_failed_<id> 下的三份诊断，保留原异常；命令行解析错误由 parser.error 报告，未必产生失败目录。研究选约中缺少候选的业务原因继续保存在 candidates.csv 等策略审计中，不能用输入质量报告替代。

## 新策略开发约束

1. 声明所需资产、字段、角色、参考服务和同步边界。
2. 选择固定合约、角色期货或期权链场景；只组合差异参数。
3. 交易候选政策留在 model、selection、signal 或 strategy 模块。
4. 通过 PreparedFrameReader 和现有 Feed.add_source 消费公共准备结果。
5. 保留业务诊断，输出输入诊断，不复制路径判断、Feather 读取和合约工厂。
6. 完成输入与行为等价验收后，再进入 run_live 和 run_backtest 的运行层统一。
