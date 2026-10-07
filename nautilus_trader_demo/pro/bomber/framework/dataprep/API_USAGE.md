# dataprep 函数使用说明

本文按当前源码说明公共输入层的调用方式，重点是单品种动态主力策略。更新日期为 2026-10-05。函数签名是开发接口说明，示例未在本地执行。运行证据来自用户 Linux 环境，详见 [主力 EMA 说明](../demos/01_main_ema/README.md) 和 [实施记录](../doc/DEMOS_INPUT_STAGE1_IMPLEMENTATION_2026-10-04.md)。

dataprep 负责路径、文件身份、读取校验、时间标准化、角色研究资料和诊断。DataHub 提供角色与因子的历史查询；market 提供回放和解析；trader 提供原生合约工厂、路由和执行。dataprep 不计算 EMA、不生成交易目标、不启动撮合，也不负责在线订阅。

## 外部累计因子接口更新

prepare_role_research 与日终 load_sector_research 新增 factors_path=None、factor_date_basis=None、factor_availability="explicit"。提供 factors_path 时，使用外部累计因子，不构造用于自计算的 RolePriceStore；未提供时保留其他调用者的原路径。minute_roles 分支暂不支持外部因子，显式拒绝该组合。用户已确认上游完成使用日期对齐，当前两个编号策略固定传 factor_date_basis="trading"、默认 factor_availability="aligned"，移除 --factor-date-basis，不再要求用户重复选择。

`load_cumulative_factors(path, *, date_basis, availability="explicit", timezone="Asia/Shanghai")` 位于 dataprep.factors，返回 `(日期, 品种, 角色) → (小写真实合约, Decimal 累计因子, available_ns)`。必需 trade_date/code/symbol；值列为 cumulative_factor 或 pcr_cumfactor，前者必须有 role，后者无 role 时仅按项目既有 main 语义处理。pcr_factor 不用来再次累计。有明确 role 的 pcr_cumfactor 可对应其他角色。

date_basis 必须声明 source 或 trading。availability=explicit 要求非负整数 available_ns；source-day-end 仅支持 source，显式假设来源日结束前可用。严格校验正有限值、真实合约身份及冲突重复。输入读取使用共享 read_feather 和报告会话。

availability=aligned 仅支持 trading，信任上游已对齐到使用当天的资料可在该日首个角色生效 Bar 使用；无 available_ns 时用内部 0 哨兵，让最终快照可用时间保持角色生效时刻，并记录该前提。有 available_ns 列时仍验证非负整数及不可提前使用，不忽略该列。aligned 是两个编号入口的默认值，不改变其他公共调用的 explicit 默认值。

load_cumulative_factors 还接受 required_keys=None。角色场景传入所需 `(日期, 品种, 信号角色)` 集合，先按品种及角色过滤，再按日期筛选，再校验因子值、合约及重复。无关品种、角色和日期的坏值不阻止本场景；所需记录的坏值仍失败，错误显示行号、日期、品种、角色、合约及原始值。未传 required_keys 的独立调用保留全表校验。当前所需日期来自全部研究准备日期，并非仅回测请求区间。

`build_external_sector_store(assignments, first_by_day, *, signal_role, factors, date_basis)` 接收品种→日期→RoleAssignment 映射，返回 SectorRoleStore。按声明日期、品种、角色匹配因子，冲突时用因子 symbol 覆盖对应角色合约，并通过 record_issue 保存 WARNING / FACTOR_CONTRACT_OVERRIDE。不修改原始 RoleAssignment 的只读映射。信号因子必需，执行专用角色的因子可选；有提供时也覆盖该角色合约并校验可用时间，没有则保留基础角色。因子与角色在首个完成 Bar 必须已可用。信号因子缺失则 MISSING_FACTOR，不补 1、不调用计算路径。各品种共同角色来源日的规则继续保留，source_day 仍表示基础角色来源，覆盖信息另见输入问题报告。

角色场景加载器筛选集合现在包含所需信号及执行角色，因此黑色板块同时提供 main/secondary 时均可参与同角色覆盖。覆盖后的 Store 是入口注册合约、加载行情和构造动态路由的依据，不能只替换数值而保留原合约身份。

下文自计算的 previous_common 和日终锚点描述适用于未传 factors_path 的兼容调用者；两个编号策略现使用外部路径，具体参数见各自 README。扫描历史范围仍未裁剪，外部文件也需要覆盖这部分准备日期。此前运行证据不能替代外部模式复测。

## 单品种主力场景的调用顺序

当前入口使用以下顺序，而不是把所有公共函数都调用一遍：

1. `@input_session` 建立本次运行的缓存和诊断上下文。
2. `resolve_futures_args(args)` 得到行情目录、角色表和基础条款文件。
3. `prepare_role_research(products=(product,), signal_role="main", execution_product=product, execution_role="main", ...)` 构造可按时间查询的主力角色和复权因子 Store。
4. 入口从 `loaded.day_end_ns` 选择回测日期，从 `loaded.store.snapshot(...)` 得到真实主力合约；发生切换时，额外准备切换当日旧主力行情。
5. `read_feather(fut_basic)`、`metadata.venue(...)` 和 `futures.instrument(...)` 准备真实合约、交易所和乘数。
6. `bar_path(...)` 定位必需文件，`add_bar_source(...)` 标准化并注册真实 OHLCV 数据。
7. 入口组装既有执行器和动态路由并启动回放。策略只用当前主力真实 Bar 收盘价乘以历史因子更新 EMA，提交逻辑目标 `rb_main`。
8. `write_input_reports(run_directory)` 与业务报表一起写出本次输入诊断。

单品种表示一个品种，不表示整个区间只有一个合约月份。main 表示角色，实际合约可能从 rb2605 切到 rb2610。复权价格仅供信号使用；撮合使用原始真实合约价格。

## 公共数据契约

这些名称是类，不是加载函数。可从 `dataprep` 导入；未列入顶层导出的函数需从其所属子模块导入。

| 类型 | 主要字段 | 使用方式 |
|---|---|---|
| DataPaths | role、fut、opt、index、contract_struct、fut_basic、opt_basic、calendar、sources | 路径解析结果；sources 记录选择来源，未配置项可为 None |
| BarFileKey | asset_kind、symbol、trading_day、venue | 标识一个真实合约某交易日文件；symbol 去交易所后缀并转大写 |
| BarReadSpec | required_fields、timestamp_column、timezone、timestamp_label、interval_seconds、trading_day_policy、value_policy、available_column、volume_semantics、require_integer_volume、require_minute_alignment、field_mapping | 指定读取规则；默认执行 OHLCV、上海时区、end、60 秒、exchange、execution_strict、分钟增量量 |
| DataRequirements | 必填 assets；products、contracts、roles、fields、bar_spec、reference_services、synchronization | 描述需求，不读取文件，不自动驱动现有 main_ema；单品种可声明 assets=("future",)、products=("RB",)、roles=("main",) |
| InstrumentSpec | 资产、合约、品种、交易所、币种、上市及最后交易日、tick、multiplier 等 | 公共静态条款，不是原生可撮合 Instrument |
| BarLoadResult | key、path、frame、spec、issues；first_ns、last_ns | 规范化数据及身份；两个属性返回最小、最大事件纳秒 |
| InputPlan | required、optional、requested、bindings | 业务决定后的文件需求；不会自动选择主力 |
| InputContext | instruments、references、calendar | 组装场景共享事实与服务，映射外层只读 |
| BindingSpec | data_key、instrument_key、data_type、bar_spec | 声明绑定，不是 trader.DataBinding |
| SourceSpec | result、purpose；reader() | 保存准备结果；reader() 创建 PreparedFrameReader |
| HistoricalInputBundle | context、sources、bindings、coverage | 完整场景准备结果，不自动启动回放 |
| CoverageReport | requested、actual_days、dependency_days、required_missing、optional_missing、files、assumptions、calendar_source | 区分运行范围和研究依赖范围，不能替代交易所完整日历审计 |
| InputIssue、InputError | 错误代码、说明、文件及行等；InputError.issue | 结构化诊断；InputError 继承 ValueError |

`BarReadSpec(timestamp_label="start")` 将源时间加一次周期，输出完成分钟时间；end 不移动。`exchange` 信任上游文件的交易日归属，允许商品夜盘；`day_session` 要求自然日一致。累计成交量需在上游按明确规则转换，公共层不猜测差分。

## 路径函数

### resolve_paths

`resolve_paths(overrides=None, env=None, required_kinds=(), project_root=None, config=None, required_files=(), validate=True) -> DataPaths`

overrides 和 config 是配置字典，env 默认当前环境，project_root 默认 pro 根目录。required_kinds 使用 future、option、index 或 role；required_files 使用 fut_basic、contract_struct、opt_basic 或 calendar。只校验声明必需的路径。validate=False 不检查存在性，仅用于配置检查和测试夹具；正式回测保持 True。

同一配置项按显式字典、config、新环境变量、根目录推导、旧环境变量及默认文件处理；专用资产目录覆盖通用父目录。相对路径按 pro 根目录解析。详细环境变量和 layout 规则见 [README](README.md)。角色表两个文件同时存在且被场景需要时必须显式选择，避免静默选错资料。

### resolve_futures_args

`resolve_futures_args(args, *, require_roles=True) -> DataPaths`

args 为 argparse.Namespace，内部传递 vars(args)。默认要求期货行情目录、fut_basic 和 contract_struct；require_roles=False 仅要求期货目录和 fut_basic。返回对象，不回写 args。单品种主力使用默认 True。

### resolve_option_args 和 resolve_targets_path

`resolve_option_args(args, *, require_futures, validate=True)` 要求期权和指数目录及 opt_basic；require_futures=True 还要求期货目录和 fut_basic。将解析结果写回 args 的 opt_dir、index_dir、fut_dir、opt_basic、fut_basic 及已有 calendar 字段，返回同一个 args。

`resolve_targets_path(path, project_root=PROJECT_ROOT) -> Path` 查找目标计划文件；相对路径先查项目根目录，再查当前工作目录。不读取 CSV。EMA 场景不用这两个函数。

## 会话和报告函数

### input_session 和 InputSession

`@input_session` 在调用函数时创建 InputSession，嵌套被装饰函数复用当前会话；只在调用时执行，导入不会启动程序。退出时清理缓存。根据调用传入的 start_day/end_day 或位置参数中的 Namespace 记录请求区间。它管理输入生命周期，不创建账户、策略或执行器，也不自动写成功报告。

需要手动组织准备过程可使用 `with InputSession() as session:`。同一个实例不能在活动状态重入；不要用手动嵌套新会话来假定共享缓存。`current_session()` 返回当前会话或 None。无会话时单独读取仍可使用，但没有本次运行共享缓存和集中诊断。

`InputSession.fingerprint(file)` 返回绝对路径、大小和 mtime_ns，记录版本并检查同次运行的变化；不是内容哈希。`InputSession.write_reports(directory)` 写三份输入报告。缓存没有持久化和容量上限，不是在线目录监控。

### read_feather 和 write_input_reports

`read_feather(path, columns=None) -> DataFrame` 懒加载 pandas，按文件版本和投影缓存；已缓存完整帧时直接取对应列。返回深拷贝。只读取原始资料，不执行行情标准化；真实 Bar 应调用 read_bar_frame。

`write_input_reports(directory)` 将当前会话写入指定目录；没有活动会话时不写。成功时应在会话退出前调用。文件为 input_manifest.json、input_coverage.json、input_issues.json，记录文件版本、路径来源、读取次数、日期覆盖、假设和问题。schema_version 保留 inputs-v1 兼容已有消费者。

装饰器遇到异常且调用参数能找到 report_dir 时，会尝试在 input_failed_<id> 写诊断，然后重新抛出原异常。argparse 错误未必有失败报告。JSON 报告存在不等于数据完整；空 issues 也不等于交易规则验收通过。

`record_issue(issue)` 向活动会话添加问题；`fail(code, message, **details)` 添加问题后抛 InputError；`json_value(value)` 将数据类、路径、日期、Decimal 等转换为报告可序列化形式。这些是公共层实现辅助接口，策略一般无需直接使用。

## 文件索引函数

全部位于 dataprep.catalog。目录扫描递归识别 `真实合约_YYYYMMDD.feather`，忽略其他命名文件；同一身份多个文件或日期目录与文件日期冲突时失败。

| 函数签名 | 返回和用途 |
|---|---|
| symbol(value) | 去后缀、去空白、转大写的身份 |
| scan_bar_files(root, asset_kind="future", start_day=None, end_day=None) | dict[BarFileKey, Path]；一次会话同根目录复用清单，日期过滤含边界 |
| select_bar_files(index, required, optional=(), requested=(None, None)) | (已找到文件映射, CoverageReport)；必需缺失失败，可选缺失记录，不改选合约 |
| bar_paths(root, required, optional=None) | {(date, 原传入合约): Path}；required/optional 为 (date, code) 集合，用于期货 |
| bar_path(root, code, day) | 一份必需期货文件 Path，缺失明确失败 |
| files_for_symbol(root, code, start_day, end_day) | {date: Path}；整个区间无文件失败 |
| bar_files(root, symbols, start_day, end_day) | {小写合约: 文件路径元组}；按指定合约收集 |
| product_inventory(root, product, start_day, end_day) | {date: 小写真实合约集合}；只是文件库存，不是主力排名 |
| inventory(root, start_day, end_day, predicate, asset_kind="option") | {(date, 大写合约): Path}；predicate 接收合约名，业务过滤由调用者提供 |

## 行情读取和注册函数

### read_bar_frame

`read_bar_frame(path, spec=None, key=None) -> BarLoadResult`

默认使用 BarReadSpec，未给 key 时从文件名推导期货身份。可显式给其他资产身份。读取并校验空文件、字段、身份、时间、重复分钟、数值及 OHLCV；execution_strict 要求正且有限价格、合法 OHLC、非负整数分钟成交量。close_strict 用于研究收盘价；research_audited 保留坏值及质量标记供既有选约规则拒绝。

输出 frame 保留 source_timestamp，datetime/event_ns/bar_ns 表示完成分钟，available_ns/available_datetime 表示可用时间。默认完成时即可用；显式 available_column 不得早于完成时间。单文件按到达排序后事件时间倒退会失败，不支持任意迟到轨迹。返回帧独立拷贝，不污染缓存。归入 actual_days 或 dependency_days 依据当前会话请求区间。

### add_bar_source 和 add_prepared_bar_source

`add_bar_source(feed, path, instrument_id, *, spec=None, key=None) -> BarLoadResult` 先调用 read_bar_frame，再注册准备结果；未传 key 时从文件名生成并补 Instrument 交易所。当前 EMA 使用此接口。

`add_prepared_bar_source(feed, result, instrument_id)` 注册已有结果，返回 None，不再次读文件。要求 execution_strict、60 秒、合约及交易所匹配；创建 PreparedFrameReader 和 FixedInstrumentBarParser，映射 available_ns，然后调用现有 Feed.add_source。不会创建或启动 Feed。

这条接口需要同步支持 available_nanoseconds 的 market/replay/parsers/bar/fixed.py 和 mapped.py，不能删除参数来绕过版本差异。研究帧不能通过此接口作为真实成交价源。

`PreparedFrameReader(frame).read(path)` 输出 (从 1 开始的行号, 字典行)，path 为 Feed 协议参数，不用来重新读文件。`first_ns(path, spec=None)` 读取并返回首个事件时间，默认严格研究 close。`timestamp_column(path)` 选择 datetime 或 timestamp，否则失败；`key_from_path(path, asset_kind="future")` 推导身份；`timestamps(values, timezone="Asia/Shanghai")` 转换时区。这三个是低层辅助函数。

## 主力角色场景函数

### prepare_role_research

`prepare_role_research(*, bars_dir, contract_struct_path, products, signal_role="main", execution_product=None, execution_role="main", end_day=None, minute_roles=(), timezone="Asia/Shanghai", bar_timestamp="end")`

这是当前单品种主力入口最重要的场景函数。products 必须是非空序列；单品种使用 ("RB",)，而不是字符串 "RB"。日终研究分支要求规范化后的品种唯一，执行品种属于该集合；建议传大写品种代码。

| 参数 | 当前 main 用法及意义 |
|---|---|
| bars_dir | 真实期货分钟文件根目录 |
| contract_struct_path | 含 trade_date、code、main 的角色表 |
| products | (product,)；仅此品种，无需板块其他品种 |
| signal_role、execution_role | 均为 main，仅校验 main 角色，不要求 second、recent、far |
| execution_product | 显式传当前 product；省略时为 products[0] |
| end_day | 限制研究行情至该日；本函数没有 start_day 参数 |
| minute_roles | 当前为空，使用日终换月锚点；不是 EMA 的分钟价格来源 |
| timezone、bar_timestamp | 必须与真实执行数据口径一致 |

minute_roles 为空返回 LoadedSectorResearch，字段为 store、day_end_ns 和 bars_dir。这里的 Sector 类型是复用容器，可以只含一个品种，并不强制板块策略。day_end_ns 是 (date, 最后事件纳秒) 元组；store.snapshot(as_of_ns) 查询角色，返回 assignment.instrument(product, "main") 和 assignment.factor(product, "main") 等资料。不存在首个可见角色或因子时明确失败。

minute_roles 非空时要求恰好一个品种且 end_day 非空，调用分钟研究分支，返回 LoadedRoleResearch，字段为 store、day_end_ns、bar_count、file_count；适用于多角色价差信号，不是当前 EMA 所需分支。

**当前加载范围限制：**日终分支扫描所选品种 end_day 及以前的全部匹配真实合约文件，用日终收盘价构造换月锚点，而非仅仅读取请求区间的主力文件。区间前数据可作为因子依赖，但大量历史或无关月份会增加准备成本，也会接受相应输入校验；目前没有按最小主力依赖集合裁剪的优化。扫描到的最早交易日也需能构造有效角色快照。运行区间筛选由入口随后完成，EMA 本身不使用区间前分钟 Bar 预热。

### references 模块的组成函数

| 函数 | 作用与返回 |
|---|---|
| load_role_assignments(path, products, required_roles) | 返回过滤后 DataFrame；required_roles 可为角色序列或按品种的字典；校验 trade_date/code/所需角色和冲突重复。main 场景无需其他角色有效 |
| load_roll_anchor_closes(files, time_spec=None) | files 是 BarFileKey→Path 映射；返回每文件最后收盘的 ObservedClose 元组 |
| load_role_minute_closes(files, time_spec=None) | 同样映射，返回逐分钟 ObservedClose 元组 |
| build_role_store(assignments, closes, *, roll_policy="previous_common", max_anchor_lookback=1) | 创建 DataHub RolePriceStore，沿用最多回看一个共同交易日的换月锚点政策 |
| build_sector_store(role_stores, days, first_by_day, signal_role, execution_product, execution_role) | 构造 SectorRoleStore；每个日期首根完成 Bar 必须有该日可见角色和因子 |
| load_sector_research(*, bars_dir, contract_struct_path, signal_products, signal_role, execution_product, execution_role, end_day=None, timezone="Asia/Shanghai", bar_timestamp="end") | 日终锚点研究底层接口；参数是 signal_products，不同于场景包装器 products |
| load_role_research(*, bars_dir, contract_struct_path, product, end_day, timezone="Asia/Shanghai", bar_timestamp="end", signal_roles=("main", "secondary", "far")) | 分钟多角色研究底层接口，返回 LoadedRoleResearch |

角色来源采用严格早于目标交易日的最近资料记录。未接入显式日历时，这不能独立证明源记录就是交易所上一交易日，不能把陈旧表自动视为完整每日资料。角色 available_ns 会参与可见性校验；不能在首个决策使用尚未可见的资料。ObservedClose 尚无发布时间字段，延迟发布的研究收盘价明确拒绝。

## 静态条款和真实合约函数

当前主力入口读取原始基础表，再调用下列薄适配，而不是要求全部合约先转换成 InstrumentSpec。

| 函数 | 参数及返回 |
|---|---|
| metadata.validate_basic(frame, required=REQUIRED_BASIC) | 检查列唯一及所需字段，返回原 frame |
| metadata.venue(basic, product) | 由基础表解析品种交易所；无记录或一个品种多个交易所失败 |
| metadata.venue_of(row)、symbol_of(row) | 单行交易所映射、保留源大小写且去后缀的合约代码 |
| metadata.selected_products(basic, products) | 返回品种→交易所映射 |
| metadata.contract_rows(basic, keys) | 返回请求合约→原始行；每个真实合约必须唯一，校验品种及可选交易所后缀 |
| metadata.future_spec(row, *, require_execution=True) | 返回 InstrumentSpec；校验生命周期，执行模式要求正 tick 和乘数 |
| metadata.load_futures_basic(path, products=None, *, require_execution=True) | 返回大写合约→InstrumentSpec 字典，可按品种过滤，重复合约失败 |
| futures.instrument(basic, profile, product, symbol, margin_init=Decimal("0.10"), margin_maint=Decimal("0.08")) | 定位唯一行、校验品种，返回 (原生 Instrument, InstrumentMeta, Decimal 乘数) |
| futures.make_instrument(row, profile, margin_init=Decimal("0.10"), margin_maint=Decimal("0.08")) | 校验条款和 Profile 交易所，调用 trader.instrument_factory.make_profile_future，返回同样三元组 |

基础表必需字段为 code、symbol、exchangeCD、contMultNum、minChgPriceNum、listDate、lastTradeDate。工厂保持既有 UTC 上市日和最后交易日次日到期口径；含非空 available_ns/source_version 的版本化条款暂不支持，必须接入按时查询服务后才可用。

`load_cffex_futures(path, product, *, require_execution=False)` 返回合约→已规范化原始行，限定 CFFEX；`load_options_basic(path, product, index_code, tick_override=None, *, kinds=("C", "P"), require_month_dates=True, require_currency=False)` 返回合约→InstrumentSpec，校验类型、行权价、到期月份、标的及交易所。两者常用于期权场景，但前者准备的是期货资料。这些不是 main 场景依赖。`positive(value, name)` 和 `contract_date(value, code, field)` 为数值与日期校验辅助函数。静态期货与期权加载内部共用 DataHub 的 FutureBasic / OptionBasic 模型进行条款校验。

### 版本化基础条款文件适配

| 函数 | 参数与返回值 |
| --- | --- |
| dataprep.basic.load_future_basic_provider | `(path, *, record_factory)`，返回 FutureBasicProvider |
| dataprep.basic.load_option_basic_provider | `(path, *, record_factory)`，返回 OptionBasicProvider |

两者复用 read_feather 的输入缓存与来源记录。record_factory 接收不可变条款对象和未丢弃来源字段的原始行，必须返回匹配数据集、合约键和值类型的 ReferenceRecord，并明确提供生效时间、可用时间和来源时间；不会自动解释 date 或上市日为发布时间。Provider 使用 `basic_at(symbol, AsOfQuery(...))` 查询。

这两个接口是版本化资料的显式准备入口，不替代现有静态场景函数。现有静态函数的签名、返回结构及对非空 available_ns / source_version 的拒绝规则保持不变。Provider.from_feather 兼容入口内部委托本模块。

## 完整 Bundle 场景函数

当前 EMA 可以直接使用 prepare_role_research 和 add_bar_source，不必迁移到完整 Bundle 才能测试。未来需将输入准备与运行器完全分离时使用下列接口。

| 函数签名 | 作用 |
|---|---|
| plan_fixed_contracts(required, optional=(), *, requested=(None, None), bindings=()) | 去重排序，返回 InputPlan；required/optional 为 BarFileKey 集合 |
| prepare_fixed_contracts(root, plan, *, spec=None, context=None) | 校验文件、准备规范化 sources，返回 HistoricalInputBundle；默认期货目录和执行规格 |
| plan_role_futures(required, optional=(), **kwargs) | 调用固定合约计划函数；角色选择仍由调用层完成 |
| prepare_role_futures(*, plan, context=None, **research_args) | 先准备角色 Store，写入 context.references 的 role_research/day_end_ns，再准备 plan 指定真实行情；不会自动从角色表生成 plan |
| plan_option_chain(required, optional=(), *, requested=(None, None), bindings=()) | 返回期权链输入计划，不决定期限或排名 |
| prepare_option_chain(paths, plan, *, specs=None, context=None) | paths 为资产种类→根目录，specs 为资产种类→BarReadSpec；返回 Bundle |
| option_chain.read_execution_frame(path, expected_symbol, day, bar_timestamp, *, asset_kind="option") | 返回 day_session 执行规范化 frame |
| option_chain.read_research_frame(path, key, day, kind, bar_timestamp) | key 为合约代码，kind 为 option/future/index；期权使用 close/open_interest/volume 的 research_audited，其他严格 close |

角色 Bundle 调用前，业务必须声明新主力和换月当日旧主力所需文件；公共层不会因缺文件改变月份。BindingSpec 仍需运行器转为实际 DataBinding。准备得到 sources 后，用 add_prepared_bar_source 消费，不再 add_bar_feather 重新读取原文件。

## 日历和目标计划函数

`load_calendar(path, start_day=None, end_day=None)` 读取含 date/is_trading_day 的 CSV 或 Feather，返回排序交易日元组，拒绝重复、空日期及超范围请求。它不会自动返回 TradingCalendar 对象。需要对象时用 `TradingCalendar(days, source, inferred=False)`；其 previous(day) 查询严格早于 day 的最近交易日，require_range(start_day, end_day) 检查对象交易日首末边界。

`infer_market_calendar(roots, start_day, end_day)` 返回文件日期推导的日期元组并记录完整性假设；缺文件不能证明休市。`infer_calendar_from_inventory(index, coverage_assumptions)` 要求明确假设，返回 inferred=True 的 TradingCalendar。当前 EMA 没有接入显式交易日历，14 自然日缺口检查不能发现所有单日缺失。

`load_target_csv(path, *, timezone_name="Asia/Shanghai")` 返回 DataHub TargetScheduleStore。CSV 必需 timestamp、target_qty 和 target_key 或 instrument_id，手数必须有限整数，同一时点同一目标不能重复；无时区时间按配置时区解释。仅供计划目标场景，不读取行情、不触发订单。当前 EMA 目标来自 EMA，不使用 CSV。

## 单品种主力当前是否满足

结论：当前公共层满足这个例子的输入准备和模拟回测接入需求，用户样本已跑通；尚不能标记为全部业务验收完成。

| 需求 | 当前状态 | 验收范围 |
|---|---|---|
| 一个入口选择一个品种 | 已实现 products=(product,) | RB 样本已运行，其他品种待验证 |
| 只需要 main 角色 | 已实现按需角色校验 | 公共角色夹具已通过，无需 second/recent/far |
| 按历史角色决定真实主力 | 已实现按时查询 | 样本未出现 unavailable；每日来源需结合角色表审计 |
| 分钟主力收盘价计算 EMA | 已实现；复权仅用于信号 | 5,493 根 Bar 参与 EMA；逐笔信号仍待核对 |
| 真实未复权合约成交 | 已接入模拟执行 | 2,275 笔订单、2,275 笔成交；终端条数不是逐笔正确性证明 |
| 换月因子和新旧合约路由 | 有实现，切换日要求旧主力行情 | 本次末尾主力 rb2605 不足以证明有换月，需单独跨换月验收 |
| start/end 标签转换 | 已实现，共同口径 | 夹具覆盖；真实样本仅验 end |
| 输入诊断和报表 | 入口已调用写出 | 已打印报表目录；三份 JSON 内容尚未审阅 |
| 实盘与回测统一 | 未实现 | 属于第二阶段，不纳入此次单品种回测通过结论 |

下一步按顺序核对 summary、orders/fills/positions/account 和三份输入 JSON，再选确定发生主力切换的日期，审查旧合约平仓、新合约目标、因子锚点及时间。结束时 last_target=1 表示策略最后目标，不直接等同账户实际持仓；策略没有结束强制平仓规则，不应仅因末尾持仓非零判失败。

本地目录已改名 demos/01_main_ema，但 run_backtest.py 仍导入 demos.main_ema.strategy。用户成功记录对应旧目录运行版本。改名后的本地入口尚未验收；开发者需统一目录名和导入路径后再同步。本次仅更新文档，没有修改该导入或执行任何代码。
