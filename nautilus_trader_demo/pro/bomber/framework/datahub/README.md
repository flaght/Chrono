# DataHub 参考资料

2026-10-12 PF2026101201新增通用特征入口 `DataHub(features=FeatureHub(registry)).feature_snapshot(...)`：强类型Schema、可见修订、EXACT/WINDOW/ASOF、频率/日历槽及一次性等待帧；文件/DolphinDB/MongoDB经公共Provider发布缓存。旧参考接口保持。源码已交付、远程验收待执行，详见[特征协议与使用说明](../dataprep/features/README.md)和[GAP08交接](../../../doc/HANDOFF/gaps/HANDOFF_GAP08_2026-10-12.md)。既有“通用特征Provider待实现”应结合本次源码状态阅读，不能解读为整项已关闭。

DataHub 定义不可变参考资料，并按事件时间查询可见版本。文件读取由 dataprep 完成；真实行情回放属于 market，原生合约、订单及持仓属于 trader。

2026-10-09数据源边界：回测／研究执行消费从DolphinDB导出的本地资料，导出／准备任务可连接数据库。在线启动及运行中的参考／特征与辅助窗口经dataprep／DataHub从库获取；历史行情Tick／Bar预热读本地。已有参考源用途约束、来源／时效检查和保留可用时间／修订的导出包及纯本地as-of读取，GAP-03专项与所选回归、真实参考读取及导出一致性已远程通过并关闭；通用DolphinDB特征Provider仍归GAP-08。来源选择不进入策略业务代码，见 [GAP-03交接](../../../doc/HANDOFF/gaps/HANDOFF_GAP03_2026-10-09.md)。

## 基础条款

| 能力 | 期货 | 期权 |
| --- | --- | --- |
| 数据集 | future_basic | option_basic |
| 不可变条款 | FutureBasic | OptionBasic |
| 查询服务 | FutureBasicProvider | OptionBasicProvider |
| 源字段适配 | FutureBasic.from_mapping | OptionBasic.from_mapping |
| 指定时刻查询 | basic_at(symbol, query) | basic_at(symbol, query) |
| 文件准备入口 | dataprep.basic.load_future_basic_provider | dataprep.basic.load_option_basic_provider |

两类模型均保存代码、交易所、币种、生命周期、乘数及最小变动价位；期货另有品种，期权另有标的、认购/认沽、行权价及到期日。代码和交易所标识原样保留，不在参考模型中进行交易所映射或根据代码猜测条款。

期货信号专用条款允许价位及乘数缺失；创建执行合约时必须齐备。期权乘数必须为正，参考条款价位可以缺失，交易场景仍要求有效价位。期权价位优先级为 tickNum → minChgPriceNum → price_increment；期货为 minChgPriceNum → tickNum → price_increment。非空无效值报错，不回退绕过。

## 时间查询与文件适配

两类 Provider 共用 BasicProvider 和既有 ReferenceRecord / AsOfQuery 时间契约：生效、发布时间、来源年龄、日终交易日隔离及重复版本校验保持一致。最新生效版本尚未发布时拒绝查询，不悄悄回退旧版本。已到期合约的历史条款仍可查询，交易资格由调用方判断。

```python
from bomber.framework.datahub import AsOfQuery, DataHub, ReferenceRecord
from bomber.framework.dataprep.basic import load_future_basic_provider, load_option_basic_provider

def record_factory(info, row):
    dataset = "future_basic" if hasattr(info, "product") else "option_basic"
    return ReferenceRecord(
        dataset, info.symbol, info,
        event_ns=int(row["effective_ns"]),
        available_ns=int(row["available_ns"]),
        source_ns=int(row["source_ns"]),
    )

futures = load_future_basic_provider(futures_path, record_factory=record_factory)
options = load_option_basic_provider(options_path, record_factory=record_factory)
hub = DataHub({"future_basic": futures, "option_basic": options})
info = futures.basic_at("rb2605", AsOfQuery(decision_ns))
```

示例时间列必须由调用方根据真实发布时间资料提供，不自动从 date、上市日或自然日推导。文件入口复用 dataprep.session.read_feather 的缓存和输入来源记录。两个 Provider 的 from_feather 作为兼容入口保留，内部委托 dataprep 文件适配器。

现有静态策略继续由 dataprep.metadata 返回 InstrumentSpec 或已规范化原始行；内部使用上述基础条款模型统一校验，但仍拒绝非空 available_ns / source_version。拥有查询 Provider 不代表静态回测入口已经支持盘中条款变更。

## 其他参考资料

外部参考数据库读取归入 [dataprep sources](../dataprep/sources/README.md)。数据库和文件适配器输出标准记录，数据源工厂不放在策略目录；在线期货组合器沿用 SectorRoleAssignment，期权标准行可进入既有 OptionBasic 模型。新增能力不将上市日当作发布时间，也不使静态回测自动获得完整历史发布版本。

| 模块 | 用途 |
| --- | --- |
| temporal | 通用版本、发布及 as-of 时间门控 |
| sector_roles | 多品种角色映射与累计因子快照 |
| role_prices / core | 单品种多角色研究价快照 |
| target_schedule | 外部完整目标计划及可用时间查询 |

这些模型描述不同数据，不要求机械地拥有相同字段。详细时间架构见 [参考资料核心说明](../../../doc/data/DATAHUB_MINIMAL_CORE.md)。

RolePriceStore 按合约与事件时间识别真实收盘记录。完全相同的重复记录只保留一次；同一事件的交易日或价格不同则抛出 ValueError，不依赖输入顺序选择价格。不同时间的分钟序列及不同合约同时报价继续有效，换约锚点仍采用来源交易日最后一条真实收盘。

## 运行环境验证

```bash
python -m unittest tests.test_basic_alignment tests.test_datahub_option_basic
python tests/test_dataprep_common.py
```
