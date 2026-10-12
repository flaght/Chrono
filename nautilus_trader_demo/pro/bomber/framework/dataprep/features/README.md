# GAP-08 通用外部数据

## 当前扩展方式：PF2026101203

普通的已计算特征只新增JSON声明。Provider按存储协议/源格式划分，业务数据集通过dataset_id和Schema版本区分。
`schema.py` 从声明字段生成带slots的frozen dataclass，校验标量类型、有限值、可空性、数值边界和白名单字段关系；不导入配置指定的类，不执行配置表达式。
字段、关系、频率、日历ID及计算版本进入Schema指纹；同一声明重新装配得到兼容类型，字段或约束变化需要管理新版本。
原有手写frozen Payload仍适用于需要专门算法的扩展，普通特征优先用声明。

默认[index_1min配置](configs/features.json)只有数据声明，真实Source字段/结束标签保持用户确认值；
[多特征样例](configs/features.example.json)在同一清单声明五分钟基差和分钟波动率，示例表名不是已核验的生产表。
一个业务清单可继续往datasets加入多个特征，不需要新建Provider、Feed、Channel或Payload Python文件。
内置两份配置是固定验收样例；业务清单可以复制为自己的JSON，再通过load_feature_bundle/--config/--index-feature-config选择。
自定义清单无需加入Python验收入口的摘要；只读报告记录选定配置摘要，运行会话保存数据集、映射和Schema身份。

```python
from bomber.framework.dataprep.features import load_feature_bundle, FeatureQueryCache
from bomber.framework.dataprep.sources.dolphindb import DolphinDbReferenceConfig

bundle = load_feature_bundle("my_features.json", DolphinDbReferenceConfig.from_env())
provider = bundle.provider("basis.five_minute")  # 与指数使用同一通用Provider
# bundle.registry可装配多个Provider/查询缓存，共享一个DataHub。
```

来源支持file标准导出、dolphindb/MongoDB标准信封、dolphindb-raw原表投影。文件默认用于离线，数据库用于Live/导出，不隐式切换。
MongoDB连接通过 `{"mongodb": {"uri": uri, "timeout_seconds": 15}}` 注入；DolphinDB通过 `{"dolphindb": connection}` 或原ReferenceConfig注入。
JSON只声明来源和映射，不写凭据。文件相对路径按配置文件目录解析。
原表适配由RawTimestamp声明DATE+日内时间或完整DolphinDB时间，也支持明确UTC整数unix_ns；频率偏移读取Schema.period_ns。
业务键支持任意正常字符串，保留前导零；数值、字符串、布尔及可空字段按声明解码，不硬绑定指数/一分钟/六位代码。

`FeatureQueryCache` 对任意已声明特征刷新DataHub，`FeatureQueryChannel`检查缓存就绪；
需要真实OHLCV的策略才使用 `FeatureQueryBarFeed` 和显式bar_projection。普通因子直接读DataHub，无需伪造Bar。
Bar投影支持声明的固定SECOND/MINUTE/HOUR/DAY频率，要求源OHLCV不可空、频率匹配、精度无损；请求Bar使用EXACT，普通特征对齐由Schema/策略需求选择。
原表默认observed只在实际查询完成后可见；有来源发布证据才声明source-time。旧CLI值update-time仅作为source-time别名。
迟到和修订保留原约束，不能凭最后覆盖值重建丢失历史。
映射和Schema身份进入来源/游标校验；换映射或定义须重新装配，不能沿用旧游标静默跳过记录。

远程部署需同步运行源码及两份内置JSON；doc目录不需上传。移除PF1202四个指数专用运行文件，入口会列出残留文件。
当前验收版本PF1203包含definitions配置扩展组；原features/index/legacy/entry组保留。当前entry6项、definitions10项、Bar场景9项是静态方法计数，未运行。

```bash
python -u -m tests.run_feature_datahub
python -u -m tests.run_cross_section_acceptance \
  --only feature-query-live --only option-vega --only option-vega-live --only live-common
python -u -m scripts.integration.features.read_only --connect --inspect \
  --dataset index.minute --key 000852
```

业务新特征探针/读取只需增加 `--config my_features.json --dataset <id> --key <key>`；
读取另加有实际数据的带时区--start/--end，标准导出加--export新路径。旧index_readonly命令是默认参数兼容壳。
PF1202声明指纹、来源身份已改变，旧导出/旧会话不能被默认为新版本兼容，保留旧包/状态并按当前定义独立导出/装配。
本机仍仅静态审阅和SHA，不执行Python、测试、编译或数据库；真实接入与GAP08验收保持开放。

以下为PF1201/1202历史接口与验证记录；指数专用运行类在PF1203已退役，当前以本节为准。

PF2026101201，2026-10-12。源码已交付，**本机未运行 Python、测试、编译或数据库连接**；远程验证待执行。首期分钟特征及指数同值输入输出，公共能力同时提供完整 WINDOW 和日频 ASOF。08 Demo 的策略政策和日历接入另行实施，不因本模块交付关闭 GAP-08。

## 层次与来源

`DatasetSpec/Payload → ExternalDataRecord → FeatureProvider → FeatureStore → DataHub.feature_snapshot → FeatureFrameAssembler`。

| 用途 | 来源 |
| --- | --- |
| offline_feature | CSV/Parquet 文件，默认验证 manifest |
| live_feature | DolphinDB/MongoDB 或注册的 database Provider |
| feature_export | 数据库读取后使用 export_features 导出 |

工厂在装配层选择来源；构造和导入不连接。调用方管理 `open/read/read_updates/close`。查询、轮询是首期公共生命周期，原生流订阅是后续可选适配，不能把当前轮询描述为流订阅。特征来源策略是新增用途；旧参考资料和历史行情 Provider 保持原契约，旧 MongoDB 参考/历史占位并未因此实现。

## Schema 与频率

Payload 必须为本地已注册的 frozen dataclass，字段声明为 float/int/decimal/str/bool，支持显式 nullable。无执行来自消息的 Python、pickle 或动态类型导入。Schema 版本禁止原地覆盖；改字段、计算公式或发布政策需要新版本，消费者明确绑定版本。数据修订 revision 与 Schema/计算版本分开。

`FrequencySpec` 声明整数纳秒周期、UTC锚点、时区及日历身份。分钟首期为60秒，支持 start/end/inclusive-end 标准化。非连续市场必须注入 `slots(start_ns,end_ns)`；日历负责午休、夜盘、节假日的预期槽，公共层逐槽核验而不生成假价格。样例日频现货用上海业务日期零点作有效标签；当天11点观测/发布的报价从11点才可见，频率为日频不等于日终发布。

`WINDOW` 返回完整有序记录，不自动猜测 sum/mean/last，也不把特征值伪装成 OHLC。特征算法或已注册聚合器消费窗口。`IndexPrice(close: Decimal)` 是原值标准化；MO/IM→000852 关系由消费方显式声明，无指数执行路由。

## 文件格式

单文件一个 `(dataset_id,schema_version)`，允许多个 key 及多个修订。必需外壳列：

```text
dataset_id,schema_version,key,ts_event,available_ns,source_ns,revision,source,quality_flags
```

Payload 使用 `payload.<字段名>`，`received_ns` 可选。时间固定为 UTC Unix 整数纳秒，不接收浮点纳秒或自动推断秒/毫秒。指数键以字符串保留前导零。CSV bool 为 true/false，Decimal 为十进制文本；可空字段为空。Parquet 使用等价标准列和整数时间。禁止将业务标签、文件修改时间或本次导出时间填作未知的历史发布时间。

`FeatureColumnMapping` 显式映射标准名→实际列及标签规则；文件缺字段、非法类型、NaN、重复冲突或未注册 Payload 字段均失败。原始外部文件可显式 `require_manifest=False` 导入校验；正式导出采用 `export_features`，生成 `.manifest.json`，记录 Schema/计算配置指纹、内容SHA256、行数和时间单位。重新读取核验指纹、身份、摘要及行数。目标已存在拒绝覆盖；manifest是包完成标记，半包不能被正式消费者接受。

格式样例位于 `doc/data/samples/gap08/`，使用合成纳秒时钟，不能当作真实市场数据或验收证据。

## 查询例子

```python
from bomber.framework.datahub import DataHub
from bomber.framework.datahub.features import FeatureHub, FeatureQuery, FeatureRequirement
from bomber.framework.dataprep.features import FileFeatureProvider, FeatureRequest
from bomber.framework.dataprep.features.samples import sample_registry

registry = sample_registry()
hub = DataHub(features=FeatureHub(registry))
request = FeatureRequest("index.price", 1, ("000852",), 0, 60_000_000_000, 60_500_000_000)
with FileFeatureProvider("doc/data/samples/gap08/index_price.csv", registry,
                         require_manifest=False) as provider:
    hub.features.store.publish(provider.read(request).records)
snapshot = hub.feature_snapshot(
    (FeatureRequirement("index", "index.price", 1, "000852"),),
    FeatureQuery(event_ns=60_000_000_000, as_of_ns=60_500_000_000),
)
assert str(snapshot.latest("index").payload.close) == "6400.1250"
```

EXACT 只取相同规范业务槽；缺失不退回旧分钟。ASOF 先按可见性筛选，每槽取最高可见 revision，再选最新业务槽；未来发布的下一条记录不阻断仍有效旧报价。WINDOW 使用 `(event-window,event]` 的完整收盘槽，起止须对齐频率，拒绝部分区间；年龄检查末槽而不是要求整个历史窗口都像最新记录一样年轻。

可见性为 `ts_event <= event_ns` 且 `max(available_ns,received_ns) <= as_of_ns`。重复刷新不改变首次接收或 source_ns。最新可见修订若质量不合格，不退回旧修订掩盖问题。快照固定一个 Store generation；不是数据库跨表事务保证。

## 数据库映射

两个数据库适配器读取显式配置的标准表/视图。必须具备整数纳秒外壳列及真实发布时间；`DatasetTable` 对 Payload 列作显式映射。例如 `FeatureColumnMapping({"payload.close": "close"})`。不猜测 `index_1min` 的字段、DFS路径、是否可订阅或时间语义。若真实表使用 date+minTime/updateTime，须先按核验结果提供标准视图或对应来源适配，不能把其原始字符串直接当整数纳秒。

```python
from bomber.framework.dataprep.features import (
    DatasetTable, FeatureColumnMapping, FeatureProviderFactory,
    DolphinDbFeatureConfig, MongoDbFeatureConfig,
)
from bomber.framework.dataprep.sources.policy import DataSourcePurpose

table = DatasetTable("index.price", 1, "index_feature_records",
    FeatureColumnMapping({"payload.close": "close"}))
# 凭据从运行环境配置注入；这里只描述配置形状。
# ddb = DolphinDbFeatureConfig(host, port, username, password, database, (table,))
# mongo = MongoDbFeatureConfig(uri, "features", (table,), source_id="index-primary")
# provider = FeatureProviderFactory.create_for(DataSourcePurpose.LIVE_FEATURE,
#     "mongodb", mongo, registry)
```

MongoDB用独立 source_id 区分实例，URI不进入repr和来源摘要。SDK在open时加载；缺SDK/连接/查询/超限分别报来源错误，不解释为空数据、不换后端、不返回截断结果。SDK用法依据 [PyMongo查询](https://www.mongodb.com/docs/languages/python/pymongo-driver/current/crud/query/find/)、[MongoClient](https://www.mongodb.com/docs/languages/python/pymongo-driver/current/connect/mongoclient/) 和 [DolphinDB Session](https://docs.dolphindb.com/en/pydoc/BasicOperations/Session/Constructor.html)；真实两库联调仍待验。

`read_updates(request,cursor)` 有界范围重读，以全部版本身份及内容摘要去重；因此可以发现旧业务槽的迟到插入和新修订，不使用最大ts_event作恢复游标。范围/来源变化、可见截止点回退或同修订内容变化明确失败。版本表应追加修订；删除或原地更新历史版本不是受支持的生产格式。长会话由控制层按配置管理读取范围及归档，超行数时失败而不静默截断。

## 在线缓存与决策帧

`FeatureRefreshService` 在控制线程调用 Provider；整批查询和校验成功后一起发布，查询完成时才记录 received_ns。连接错误使请求范围的缓存不可消费，恢复后须成功回读；成功轮询仅更新来源观测期限，不能刷新原始数据年龄。Hub区分 MISSING、QUALITY、STALE、SOURCE_ERROR、REFRESH_STALE。

`FeatureFrameAssembler` 接受已完成的市场输入，按策略声明的需求等待；既不阻塞行情指标，也不查询数据库。每次调用 offer 后由运行层推进 poll：齐全则输出一次 READY；截止点缺失则 SKIPPED；实际poll晚于期限也SKIPPED，不补发旧分钟目标。期限边界包含，同一时点先发布特征/市场数据，再poll。Schema/版本冲突作为错误向上抛出。消费方对 FrameOutcome 的 READY 才调用自己的策略决策入口；是否取消旧目标、保护性平仓或允许ASOF仍属于策略/执行政策。

`replay_feature_frames` 合并发布时间、市场可用时刻、期限事件，使用整数模拟时钟，不sleep，不等下一根Bar再触发超时。Live使用同一个Assembler，由实际运行时钟推进。复现某次Live时，文件还需要该次 received_ns；只有源发布时间的研究文件不能证明网络延迟完全一致。

恢复可以数据库有界回读后按版本去重，并对新市场帧重新装配；本模块不自动恢复/重发旧pending交易帧。数据游标不是账户/订单检查点，也不解除GAP02/05/06执行恢复约束。日历来源和08 Demo接入另行验收。

## 验证

当前PF2026101202默认使用验收Python入口内置的被测源码SHA256，远程只部署代码即可，**不需要上传doc目录、样例文件或部署清单**。新增真实指数原表适配后须同步新版本源码和入口；混版预检会列具体代码文件。入口自身SHA记录到报告，避免内置自引用；`--manifest PATH`只用于显式外部清单覆盖，仍会拒绝不匹配。下文REV2/3/4描述保留PF1201失败历史。

在 Linux 项目 `pro` 目录执行，环境需要 pandas、Parquet引擎（项目已有pyarrow可复用）；数据库SDK不是无网络测试的前置条件：

```bash
python -u -m tests.run_feature_datahub
```

记录项目/模块加载路径、源码SHA256、各组日志与summary.json。失败只复测对应组：

```bash
python -u -m tests.run_feature_datahub --only features
python -u -m tests.run_feature_datahub --only index
python -u -m tests.run_feature_datahub --only legacy
```

公共 MinuteFrameBuffer 的期限修复影响08，补跑已存在组：

```bash
python -u -m tests.run_cross_section_acceptance --only option-vega
```

集中入口也增加 `--only feature-datahub`，但该入口原有部署预检需要原生SDK；独立入口无需行情/交易扩展。首轮远程35项中期限检查1项失败，08原有33项通过；legacy/D0尚未执行。本地期限修复与交付摘要一致，远端加载版本仍待核验。REV2独立入口先核对部署摘要并打印market.frames实际路径/SHA256，默认使用PF2026101201_REV2清单；详见[GAP08交接](../../../../doc/HANDOFF/gaps/HANDOFF_GAP08_2026-10-12.md)。真实数据库读取、一分钟策略消费及本地/Live一致性需要实际数据和匹配版本验收，不用假SDK测试代替。

第二轮只输出报告目录，尚无GAP08测试结果。REV2遗漏了预检异常的终端输出，实际原因在该次summary.json的deployment_error。REV3补错误类型/消息/报告路径打印，默认匹配PF2026101201_REV3清单，增加entry组的三个无网络防回归用例。当前同步入口、tests/test_feature_acceptance_entry.py及REV3摘要后再运行；无须重复已过且运行组件未变化的08组。本机仍未运行REV3。

第三轮明确错误为REV3文档清单不存在；用户确认远程不上传文档。REV4解除文档运行依赖，entry组现在5项（新增无doc目录仍可正常启动，以及内置摘要仍拒绝混版）。features仍35项；本机未执行REV4测试，远程完整features/legacy/entry/D0待验。已过08组无需因本次入口修正重复。

REV4远程预检现已定位唯一混版为market/frames.py，远端摘要与本地删除期限拒绝四行后的摘要完全相同。补同步这一个代码文件，再跑GAP08；由于远端公共运行组件改变，此时补跑08受影响组。无需同步文档或更新验收入口。证据详见交接末尾。

## PF2026101202 指数原表

`DolphinDbIndexConfig` 默认 `dfs://min_bar/index_1min`、上海时区、分钟结束标签，显式列映射
覆盖用户确认的12字段。`open()` 核验 `schema(...).colDefs` 的 `name/typeString`，不靠字段名
猜类型；`Code` 要求 STRING/SYMBOL，保留 `000852` 前导零。`date` 要求 DATE，`minTime`
和 `updateTime` 支持 MINUTE/SECOND/TIME/NANOTIME 或 DATETIME/TIMESTAMP/NANOTIMESTAMP/DATEHOUR。
时间转换依据 [DolphinDB concatDateTime](https://docs.dolphindb.com/en/Functions/c/concatDateTime.html)
及 [schema](https://docs.dolphindb.com/en/Functions/s/schema.html)；未知或字符串时间类型明确报错。
`IndexMinute` 保留真实OHLC、量额、prev_close及pct_change，不造结算价等期货字段，不改变百分比口径。

`FeatureProviderFactory.create_for(LIVE_FEATURE, "dolphindb-index", config, registry)` 返回此适配器；
相同 `index_minute_registry()` 用于数据库查询、公共对齐、CSV/Parquet文件回测。
`availability="observed"` 默认以实际查询完成时间为available_ns及观测revision，质量位2标识观测证据。
为适配在线查询开始到完成的耗时，返回记录可晚于请求as_of_ns；Hub仍严格按实际visible_ns屏蔽早期查询。
相同内容重复读取保持首次观测，改变内容产生新版本；批内同分钟冲突快照失败，不能推测过去可用时间。
`availability="update-time"` 只在来源确认updateTime可证明发布时选用，available_ns/revision采用真实
updateTime。不是原始修订序号；同时间内容变化仍失败。原表覆盖历史无法重建消失版本。
实际SDK接收时间由 `FeatureRefreshService` 在查询结束后另记received_ns；轮询不能刷新源分钟年龄。
文件manifest的publication_evidence为record_available_ns，实际证据含义由source身份/质量位区分。

在远程Linux先只读查看Schema和3条样例（凭据从项目 `.env` 或已有环境变量读取）：

```bash
python -u -m scripts.integration.features.index_readonly --connect --inspect
```

再用实际有数据的业务区间读取，并选择一个尚不存在的导出路径：

```bash
python -u -m scripts.integration.features.index_readonly --connect \
  --key 000852 --start "$INDEX_START" --end "$INDEX_END" \
  --availability observed --timestamp-label end --export "$INDEX_EXPORT"
```

INDEX_START/INDEX_END须ISO时间且带 `+08:00`，如分钟结束09:31/09:33；这里不预设真实数据日期。
读取空数据明确失败；导出后逐记录比较文件往返，保留所有版本/来源/可用时间，不改写为业务分钟即刻可用。
observed导出只能在下载完成后可见，不能用于下载之前决策的历史可见性研究。
历史 `--as-of` 仅允许已确认update-time的读取。探针只验证结构，读取和文件往返只验证选定区间，
不证明全天完整、真实低延迟或策略消费。报告独立写入tests/results/index-readonly-*/summary.json。

原生Bar与08 Recording装配的离线组：

```bash
python -u -m tests.run_cross_section_acceptance \
  --only index-query-live --only option-vega --only option-vega-live --only live-common
```

本机仅静态源码审阅，新增用例未执行；真实数据库类型、查询时延、导出和08在线分钟消费待远程验收。
