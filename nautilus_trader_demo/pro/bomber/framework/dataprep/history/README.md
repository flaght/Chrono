# 历史分钟行情Provider

PF2026100911第一阶段源码，远程验证待完成。`HistoryRequest → HistoryProviderFactory → HistoryProvider.read → HistoryBar` 独立于参考资料工厂；策略不连接数据库。导入、构造不联网，协调器显式管理open/read/close，失败也关闭连接。

| 阶段 | 来源规则 |
| --- | --- |
| offline_market_history | 本地file |
| live_market_history / preopen | 文件优先；文件缺失或不足时从配置数据库补足 |
| live_recovery / recovery | 数据库；加载期间已收到的完整实时分钟可补齐对应分钟 |

`HistoryService` 按固定截止点读取，筛选真实合约和区间，完全重复去重、同键冲突拒绝，报告条数、缺口及来源。`missing_policy="fail"` 默认拒绝缺失；显式allow允许仅用实际有效行或零行继续，记录数据库未配置/不可用等原因，不制造价格。Schema错误、内容冲突、未实现的驱动仍失败。交易时效、账户归属及活动订单对账不受此参数豁免。

JSONL和带表头CSV采用标准字段：instrument_id、ts_event，以及close/adjusted_close；OHLC、volume、cumulative_factor可选。时间戳为完整分钟末最后1ns。EMA需要adjusted_close，或close乘该分钟实际cumulative_factor；不使用启动时当前因子回算历史。文件不包含参考角色资料的替代许可。

DolphinDB配置显式声明DFS路径、表名、字段映射、时间单位（ns/ms/s）、时间标签（start/end_ns）和合约格式（full/symbol，symbol须固定venue）。只读查询按合约/时间限定，多读一行检测行数截断。连接复用DDB环境配置及超时，表名和行情字段不猜测。当前读取分钟表，尚未实现原始Tick聚合、跨表历史因子连接、历史版本as-of选择、跨主力换月连续行情。

后续MongoDB适配继承同一契约并注册，协调器和策略不变：

```python
from bomber.framework.dataprep.history import HistoryProvider, HistoryProviderFactory

class MongoMinuteProvider(HistoryProvider):
    # open/read/close由实际MongoDB驱动实现。
    # read(request)返回经过时间和合约标准化的HistoryBar元组。
    ...

# 仅在完成三个抽象方法后注册；当前mongodb位置是显式未实现的占位。
HistoryProviderFactory.register("mongodb", MongoMinuteProvider, replace=True)
```

CLI及远程验证见 [实施交接](../../../../doc/HANDOFF/HANDOFF_HISTORY_PROVIDER_2026-10-09.md)。
