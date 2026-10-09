# 参考数据库只读核验

本目录核验参考资料读取和字段契约，不连接 CTP MD／TD，也不报单。公共实现位于 [dataprep sources](../../../bomber/framework/dataprep/sources/README.md)，入口加载项目 .env、接收参数并展示结果。

## GAP-03当前入口与导出

readonly新增必填`--expected-source-day`，检查声明来源日、来源年龄、成功观测及真实发布时间。下文旧命令保留历史记录，执行时补充当前已核对的来源日。当前在线默认DolphinDB，文件须显式测试标志；新增22个GAP-03无网络测试方法及代码待用户远程验收，见 [专项交接](../../../doc/HANDOFF/HANDOFF_GAP03_2026-10-09.md)。

已有 `examples/opts2/kline_task.py:export_day` 的Feather导出及上游 `code/bomber/bomber/examples/bomber_adapter/tools/sync_data.py` 的Parquet导出继续负责行情文件，本轮只补参考审计包。核对当前TD及来源日后示例（日期随会话更新）：

```bash
python -u -m scripts.integration.reference_data.readonly \
  --connect --trading-day 20261009 --expected-source-day 20261008 --products CU \
  --database dfs://bomber_daily --factor-date-basis source \
  --factor-availability observed-on-read --report tests/results/gap03-reference-cu-20261009.json
python -u -m scripts.integration.reference_data.export \
  --connect --products CU --start-day 20261008 --end-day 20261008 \
  --database dfs://bomber_daily --visibility observed \
  --destination tests/results/gap03-cu-reference-20261009.json
python -u -m scripts.integration.reference_data.verify_export \
  --bundle tests/results/gap03-cu-reference-20261009.json \
  --trading-day 20261009 --expected-source-day 20261008 --products CU
```

导出每次选新文件，不覆盖旧版本。observed只提供稳定导出观测后的快照，不证明此前历史发布时间；历史回放用真实available_ns的explicit导出，缺失时明确失败。verify_export只读本地，默认as-of为导出观测时刻，校验内容／Schema／来源日并打印合约、因子及manifest；与当前只读结果核对并保留输出。两次查询间若发生更新，核查版本差异，不强行认定一致。

Python消费者用 `ExportedReferenceSource(path, as_of_ns=...)` 或 `read_as_of(dataset, query, as_of_ns)`；晚到修订按实际可用时间可见。源码保留source_version／revision，未提供时只有内容指纹，不声称完整上游发布历史。现有demo本地文件入口保持，参考包不替代GAP-08通用特征格式。

PF2026100804最新日期口径：`--trading-day`是TD当前交易日，不是数据库查询来源日。原始DolphinDB日表默认 `--factor-date-basis source`：读取此前最新期限结构来源日及同日因子，合约有效性仍按当前TD日验证。若期限结构此前最近记录为2026-09-30，TD为20261008，则因子精确查2026-09-30，不要求2026-10-08因子。已有available_ns优先门控；缺同日因子拒绝，不退回更老因子。最近来源日选择依赖上游日表完整性，不由自然日减一计算。

最新验证已通过：远程 `tests/results/main-ema-acceptance-1791444863148068678` 显示references29/29、公共回归32/32、entry22/22，代码摘要一致；真实DolphinDB只读报告 `tests/results/reference-data-20261008.json` 显示TD指定日2026-10-08，source_day=factor_date=2026-09-30，RB主力rb2701.SHFE，累计因子1.181055、tick1、multiplier10，ready=true。数据和日期修正版验收已齐，源码未改无需重跑下列历史复验命令；直接继续本页末尾SimNow Recording。只读核验未连接柜台，实际MD行情／策略目标及柜台成交仍待验证。

同步公共 `bomber/framework/dataprep/live_references.py`、01 `live_references.py`／`run_live.py`、本目录 `readonly.py`，以及 `tests/test_reference_sources.py`、`run_main_ema_live.py`、`run_main_ema_acceptance.py`。日期口径修正版references29项、entry22项及既有公共回归32项待远程验证，先运行下面三组，再重试数据库只读：

```bash
python -u -m tests.run_main_ema_acceptance \
  --only references --only reference-regression --only entry
```

2026-10-08 当前结论：相关离线验收已完成，references23/23、公共回归32/32、角色研究与信号两组及entry21/21均有远程通过证据。最新四组报告为 `tests/results/main-ema-acceptance-1791443131209031509`，数据源证据为此前 `main-ema-acceptance-1791441568092793677`。测试使用假DB／MD／TD，不是实际柜台交易；源码没有变化时无需重复这些离线组，直接继续本页数据库只读和实时Recording。

远程 Linux uv-nautilus 环境的 pro 根目录先执行无网络增量：

同步完整公共 sources 目录、dataprep 的 live_references.py／metadata.py／factors.py／references.py、01 的 live_references.py／run_live.py，以及 tests 的 test_reference_sources.py／test_basic_alignment.py／run_main_ema_live.py／run_main_ema_acceptance.py。修正版集中入口打印加载路径和摘要；远程缺少 DatabaseMainReferences 表示01装配尚未同步完整，不要在测试中回退到旧文件实现。

```bash
python -u -m tests.run_main_ema_acceptance \
  --only references --only reference-regression --only entry
```

2026-10-08 第二轮远程结果：数据源23/23通过，加载路径与修正版摘要一致；公共回归30项中29项通过，RolePriceStore的冲突收盘记录用例失败，entry未执行。角色研究价修正版增加同事件冲突拒绝及完全相同记录去重，不修改01策略或数据库资料加载。同步 `bomber/framework/datahub/role_prices.py`、`tests/test_datahub_option_basic.py`、`tests/run_main_ema_acceptance.py`，保留 `tests/run_role_research.py` 与 `tests/test_demo_role_cross_three_roles.py`，补跑：

```bash
python -u -m tests.run_main_ema_acceptance \
  --only reference-regression --only role-research --only role-signal --only entry
```

公共回归现在32项；角色研究覆盖四角色换约、发布时间和缺价锚点，角色信号覆盖三角色演示及四角色兼容。修正版待远程验证，已通过且源码未改的数据源组不必重复执行。

第三轮远程结果：公共回归30项通过，但尚未证明包含新增的2项；角色研究在M0/M1/M2通过后因旧examples导入中断，角色信号与entry未执行。测试入口已改用公共key_from_path及prepare_role_research，不恢复旧examples。请同步 `tests/run_role_research.py`、`tests/run_main_ema_acceptance.py`、`tests/test_datahub_option_basic.py`，并核对 `tests/test_basic_alignment.py`；继续执行上面的四组命令。集中入口现在检查新增测试方法并打印测试模块实际路径和SHA-256，缺少方法则在运行前提示部署不完整。公共运行代码此次未修改，数据源23项不重跑。

通过后，沿用项目 .env 或环境变量中的 DDB_HOST、DDB_PORT、DDB_USERNAME、DDB_PASSWORD。以下交易日须替换为实际已核对的 TD 交易日：

```bash
python -u -m scripts.integration.reference_data.readonly \
  --connect --trading-day 20261008 --products RB \
  --factor-date-basis source \
  --database dfs://bomber_daily \
  --report tests/results/reference-data-20261008.json
```

先核验RB期货链即可继续01接入；可选追加 --option-product MO 核验opt_basic，不作为01前置。--products 可传多个品种，所有请求品种须有同一来源日的角色、按所选口径精确匹配的因子及本次TD日有效的条款。缺失、冲突、未来发布时间或读取中变化都会失败，不改写源因子日期。交易日以TD登录返回为准，夜盘不能按自然日期直接指定。

结果包含TD日、factor_date_basis、factor_date、期限结构来源日、真实主力、累计因子、乘数、价位、资料指纹及可选期权核验，不输出密码。若角色来源日为2026-09-30且有同日因子，source口径可用于2026-10-08；若角色来源日已更新但对应日因子缺失，仍须补齐对应来源日，不能自动取更早因子。只有已对齐TD适用日的数据才显式选择 --factor-date-basis trading。

随后验证当前策略实时 Recording：

```bash
python -u -m demos.01_main_ema.run_live \
  --connect --mode recording --product RB --seconds 600 \
  --reference-source dolphindb --reference-database dfs://bomber_daily \
  --factor-date-basis source
```

该命令连接 SimNow MD／TD并核验账户，Recording 不报单。数据源选择只影响参考资料，实时行情仍来自 CTP，交易日仍由 TD 确认。数据库模式不要求角色、因子或条款文件路径。完整交易授权及停机边界按 [01 接入验收](../../../demos/01_main_ema/SIMNOW_ACCEPTANCE.md) 执行。
