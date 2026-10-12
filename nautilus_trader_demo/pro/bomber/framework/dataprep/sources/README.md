# 参考资料数据源

参考资料通过统一接口读取，数据源工厂选择文件、DolphinDB 或后续数据库子类。策略消费角色快照或条款模型，不查询数据库。构造和导入不建立连接，调用方显式管理 open／close 生命周期。

2026-10-09正式边界：DolphinDB提供在线参考／特征和回测／研究文件的上游导出；导出任务可连接，离线执行读本地。在线启动和运行中的辅助数据均从库获取。参考用途仍为offline_reference/file_reference_test仅file、live_reference/export仅dolphindb。PF2026100911按用户补充接入独立 [历史行情Provider工厂](../history/README.md)：offline_market_history仅file，live_market_history可用file或数据库，live_recovery仅数据库；盘前文件不足时回退配置数据库，盘中恢复读取数据库。MongoDB历史驱动尚未实现，不等于已能联网读取。构造不联网，回退由历史协调器显式执行；01及联合参考入口默认数据库，file须显式测试标志。通用特征Provider仍归GAP-08；GAP-03专项与所选回归、真实参考导出及纯本地一致性已由用户远程验收通过并关闭，见 [GAP-03交接](../../../../doc/HANDOFF/gaps/HANDOFF_GAP03_2026-10-09.md)。历史新增源码验证归PF2026100911/GAP-04，不改写此前验收证据。

## GAP-03时效与参考导出

`ReferenceFreshnessPolicy` 默认角色／因子距TD日最多14自然日、最近成功观测最多30秒、单轮刷新最多30秒。年龄预算不推断上一交易日；一次性在线数据库入口必填按日历核对的expected-source-day，持续模式由每个会话日历提供。预期日不符、来源／观测过期、刷新过长或真实available_ns尚未来到均不得放行。基础条款检查生命周期和成功观测，不用上市日判断陈旧。

`sources.artifacts.export_reference_bundle` 对声明的ReferenceQuery范围做两轮稳定读取，原子输出不覆盖旧文件的JSON参考包。清单保存源库／表、脱敏连接身份、范围、Schema、字段类型、源版本／修订、内容指纹和可用时间。explicit须有真实available_ns；缺失时失败。observed仅提供当前观测快照，可用时间不早于稳定导出完成时刻，不能回测此前历史。两轮一致不是事务证明，导出只覆盖实际源版本，不补造发布历史。

`ExportedReferenceSource(path, as_of_ns=...)` 只读本地包，校验Schema和内容后按业务键、available_ns及revision选可见版本，未来修订不泄漏；同键、同可用时间及同修订冲突时拒绝。它供统一ReferenceDataSource消费者使用；九个回测入口仍消费既有本地文件，本轮不迁移全部demo，也不替代GAP-08通用特征格式。CLI及验证见 [参考导出说明](../../../../scripts/integration/reference_data/README.md)。

## 公共组件

| 组件 | 职责 |
| --- | --- |
| ReferenceDataSource | open、read、close，以及四种资料查询入口 |
| ReferenceSourceFactory | create、register；不同后端采用独立配置 |
| DolphinDbReferenceSource | 官方 SDK 会话和限定范围的 DFS 只读查询 |
| FileReferenceSource | 委托 read_feather，再进入共同字段标准化 |
| MySqlReferenceSource／MongoDbReferenceSource | 同一父类契约；open／read 明确抛 NotImplementedError |
| ReferenceQuery | 品种、真实合约、来源日期范围及合约有效日；不是历史 as-of 查询 |
| ReferenceBatch | 标准字段、只读记录、来源和内容 SHA-256；frame() 返回独立表 |
| dataprep.live_references.LiveFuturesReferences | 将期货角色、累计因子和条款组合为单交易日、多品种／角色快照 |
| dataprep.live_role.FileRoleReferences | 单角色文件适配，提供snapshot、spec、instrument_id及instrument_meta |
| dataprep.live_role.SourceRoleReferences | 任意ReferenceSource的单角色适配，与文件适配共享策略读取接口 |

load_role_assignments／load_cumulative_factors 保留文件函数签名，新增 parse_role_assignments／parse_cumulative_factors 接受标准 DataFrame，文件和数据库共用业务规则。

metadata.parse_future_terms 解析纯期货条款值，future_spec_from_basic 将已校验的 FutureBasic 转为规格；发布时间由在线快照或版本化Provider管理。原 future_spec 静态入口仍拒绝 available_ns／source_version，不能用静态回测入口直接解释版本化资料。

## DolphinDB 表级字段映射

默认数据库为 `dfs://bomber_daily`。Code 的含义按表定义。

| 数据集 | 默认表 | Code | contractObject | date |
| --- | --- | --- | --- | --- |
| futures_basic | fut_basic | symbol：真实期货合约 | code：品种 | listDate：上市日 |
| options_basic | opt_basic | symbol：真实期权合约 | 当前没有，不制造 | listDate：上市日 |
| adjustment_factors | fut_adjustment_factors | symbol：真实期货合约 | code：品种 | trade_date：因子适用日 |
| contract_structure | fut_contract | code：品种 | 不依赖 | date：角色来源日 |

opt_basic 的品种筛选按合约代码的字母前缀进行，仅用于筛选，输出不增加 code 或 contractObject。无法识别格式时使用明确 symbols 查询。OptionBasic.from_mapping 可直接消费标准行；交易所映射和交易合约构造仍由执行装配完成。

main／second／recent／far 源字段保留，公共角色为 main／secondary／near／far。在线组合器将角色来源 date 显式转交给角色解析器的 trade_date 字段，不将其改成当天因子适用日。

源字段和标准字段同时存在且冲突时失败。结果再次按请求范围筛选。超过行数上限失败，不返回截断资料。期限结构按 products 查询，active_on 只用于基础条款。

## 配置和装配

复用 DDB_HOST、DDB_PORT（默认8848）、DDB_USERNAME、DDB_PASSWORD。密码不放在命令参数、配置 repr 或结果文件中。可选环境变量为 DDB_REFERENCE_DATABASE、DDB_FUT_BASIC_TABLE、DDB_OPT_BASIC_TABLE、DDB_FACTOR_TABLE、DDB_CONTRACT_TABLE、DDB_ENABLE_SSL；实际表名不同时显式配置，不自动尝试 opt_info 或单数因子表。

```python
from bomber.framework.dataprep.sources import DolphinDbReferenceConfig, ReferenceSourceFactory
from bomber.framework.dataprep.live_references import LiveFuturesReferences

config = DolphinDbReferenceConfig.from_env()
with ReferenceSourceFactory.create("dolphindb", config) as source:
    references = LiveFuturesReferences(
        source, products=("RB",), trading_day=td_trading_day,
        started_ns=started_ns, roles=("main",), factor_roles=("main",),
        factor_date_basis="source",
    )
    strategy = MainEmaStrategy("main-ema", references, strategy_config)
    # source 的生命周期覆盖策略整个运行期。
```

示例变量由运行入口提供：交易日来自核对后的 TD 会话，started_ns 为启动时间，策略类和配置来自 demo。不从自然日期推断夜盘交易日。多品种使用 products，roles 和 factor_roles 也可接受按品种映射的字典。仅执行的次主力可不请求复权因子，主力因子不覆盖次主力角色。

原始数据库日表应显式向公共LiveFuturesReferences传入 `factor_date_basis="source"`；其默认trading只为保留原直接调用兼容。01数据库适配与只读探针默认source，01文件模式默认trading；run_live新增 --factor-date-basis 可显式切换。source取此前角色来源日的同日因子，trading取已对齐TD适用日的因子；它们是不同输入契约，不通过更改源日期来伪装对齐。

## 时间和刷新边界

2026-10-08 最新澄清：回测文件是已对齐适用日的trading/aligned输入；本次实盘数据库是原始来源日表，须用source。先选择严格早于TD日的最新完整角色来源日，再精确读取同日因子；多品种要求同一来源日，缺该日因子不能退回其他日期。不是自然日减一天；未接入交易所日历时，最近来源日等于前一交易日仍依赖上游日表完整性，该假设写入manifest。

source默认采用source-day-end（按来源自然日结束声明资料可用）；上游已有available_ns时，改用其实际发布时间门控。该日终假设在来源日午夜之前不会使资料可见，夜盘需要准确available_ns时使用explicit。trading默认aligned；不兼容的日期／发布时间组合报错。同角色合约冲突继续以选中因子表symbol为准，pcr_factor不再累计，pcr_cumfactor不补1，条款available_ns也参与门控。

PF2026100810增加在线ReferenceSource可显式选择的`factor_availability="observed-on-read"`，用于资料已入库、但源表没有`available_ns`的实时夜盘。角色、因子和条款两轮读取指纹一致后，首次稳定读取完成的墙钟时间作为当前进程最早可用时刻；不能用于更早的Bar或历史回测。同一指纹刷新保持首次时间，内容变化重新计时；源表如有真实`available_ns`仍可进一步推迟可用时刻。manifest记录`first_observed_ns`、`available_ns`和观测证据，指纹不构成正式发布版本。文件入口拒绝此口径；01入口仅在DolphinDB实时会话接受，默认规则不变。跨进程重新启动将重新观察并计时。

默认每5秒按需检查，未到间隔时读取内存；每轮读取角色、因子和有效条款，再重复读取核对内容。失败后旧快照禁止用于决策，后续成功刷新才恢复资料可用性。初版刷新为同步查询，可能等待数据库；01 在查询后再检查行情及会话时效，读取失败会停止入口，不自动恢复授权。

更新版从观察到更新的时刻生效，不能用于较早事件。本服务不保留完整历史快照。内容指纹不是上游发布版本，重复读取一致也不等于数据库事务快照。没有发布历史时，不承诺事后修订资料的历史可见性；历史版本管理及跨表原子发布须另行约定。角色来源日保持原文件规则，未增加新的最大年龄限制。

多品种参考快照不代表多腿执行已验收。01 的 CNY／SHFE／INE、固定路由、独占账户、冷启动和 SimNow 授权范围仍由通道装配保持。

PF2026100806将01原适配文件移入公共dataprep/live_role.py，demo只在run_live.py组装。FileRoleReferences与SourceRoleReferences均通过role选择main／secondary／near／far；交易所及币种限制可由allowed_venues／required_currency显式传入。其他策略不需要EMA配置即可消费snapshot；SourceRoleReferences委托已有LiveFuturesReferences，文件默认trading、ReferenceSource默认source。现有pcr_cumfactor没有role时仍仅解释为main；次主力等角色必须提供对应因子记录，不复制主力因子。工厂、回测解析接口及现有发布时间规则没有修改。配套固定单角色运行组件位于trader/live_roles.py，不提供多腿自动执行。

SDK 使用官方 [连接参数](https://docs.dolphindb.com/en/pydoc/BasicOperations/Session/Connect.html) 和 [查询接口](https://docs.dolphindb.com/en/pydoc/BasicOperations/Session/OtherParams.html)，运行环境须支持 connect 的 readTimeout／writeTimeout，不通过移除超时兼容旧版本。

## 验证

用户 Linux uv-nautilus 环境的 pro 根目录：

```bash
python -u -m tests.run_main_ema_acceptance \
  --only references --only reference-regression --only entry
```

增量覆盖字段、工厂、期权无 contractObject、数据库／文件一致的 EMA 信号、时间门控、读取中变化、刷新失败、多品种／角色及假 MD／TD 的完整入口。集中入口先打印实际模块路径与SHA-256，并检查必要接口。实际数据库联调见 [只读核验](../../../../scripts/integration/reference_data/README.md)。本机不运行 Python 或测试。

2026-10-08 首轮远程 references 为21/23通过：01模块缺少 DatabaseMainReferences，在线组合器误用静态 future_spec。修正版拆出共用纯条款解析、保留在线发布时间门控和静态拒绝规则；须同步 metadata.py、公共 live_references.py、01入口及配套测试再复验。reference-regression／entry 因集中入口停止，本次尚未运行。

2026-10-08 最新远程验证：第二轮references23/23通过；第四轮公共回归32/32、角色研究与角色信号、entry21/21全部通过，最新四组报告为 `tests/results/main-ema-acceptance-1791443131209031509`。部署路径与摘要一致，新增静态条款拒绝、角色价格冲突／去重及假数据库完整入口均已验证。当前相关离线验收完成，源码未改无需重复；真实DolphinDB只读及当前策略SimNow Recording待运行。测试中的数据库和柜台均为夹具，未取得本次真实网络或成交证据。

PF2026100804日期口径修正版新增6项source回归，references现29项，entry增加CLI契约用例现22项。上述旧通过证据不覆盖此修正版；重新验证references、reference-regression、entry即可，未改动的角色研究／信号不重跑。本机未执行修正版。

PF2026100804最新证据：远程三组29/29、32/32、22/22全部通过，报告 `tests/results/main-ema-acceptance-1791444863148068678`；真实DolphinDB RB核验也通过，指定TD日2026-10-08、选中角色／因子来源日2026-09-30、主力rb2701.SHFE、累计因子1.181055、tick1、multiplier10，结果文件 `tests/results/reference-data-20261008.json`。相关离线兼容和真实RB三表读取均已验证，当前策略SimNow Recording及真实柜台成交继续独立验收；未覆盖期权或其他品种。
