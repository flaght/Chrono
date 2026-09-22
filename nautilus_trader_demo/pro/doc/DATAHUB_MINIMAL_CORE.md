# DataHub最小核：职责与分阶段验收

DataHub是**参考/研究数据查询边界**，不是行情Feed、策略、仓位账本或交易客户端。
策略拿到的是决策时刻已可见的数据；真实Bar/Tick仍由`market`供给撮合。
当前只实现格式无关的基础核及已有第四类角色研究价，不能标作完整DataHub。

## 从example06～10提炼的能力

| 来源 | 可复用能力 | 应留在DataHub之外的职责 |
|---|---|---|
| example06 | 四角色动态合约映射、源合约时间、复权价、缺失/陈旧检测 | 真实订单、活动委托和仓位查询 |
| example07 | `event_ns`、`available_ns`、`source_ns`、as-of快照及防未来查询 | 策略目标、换月状态机 |
| example08 | 角色/因子预加载、按时点索引、角色专属研究数据 | Feather等具体格式不进入核心 |
| example09 | 决策时间与数据时间分离、交易日窗口；当日日终表不可用于当日盘中 | 由自然日期推导夜盘交易日 |
| example10 | 合约基础信息、精度/乘数/手续费来源、文件原子加载 | 仓位、撮合及策略内硬编码路径 |

共同规则：**数据时间、生效时间、最早可用时间和源数据时间不是同一个概念。**
日终数据还带来源交易日；查询必须由行情/交易日历传入当前交易日，不能用
`timestamp.date()`或减一个自然日代替。复权价仅是信号研究值，不生成可交易
Instrument或Bar。实际账户持仓及委托属于执行状态，不能塞进DataHub。

## 当前第一步D0：格式无关的时间与Provider框架

`datahub/temporal.py`提供：

- `ReferenceRecord`：携带`event_ns / available_ns / source_ns`、来源交易日及
  `INTRADAY / DAY_END`发布规则的标准记录；适配器负责提供不可变值对象。
- `AsOfQuery`：显式的决策时点、可选数据截止时点、当前交易日、窗口和源数据
  最大年龄。数据截止时点不得晚于决策时点。
- `AsOfProviderPort`：任意文件、数据库、DolphinDB或服务端实现相同`read`协议。
- `DataHub`：按数据集装配Provider；读取后再次检查未来事件、尚未发布版本、
  当日日终记录和过旧源数据。多键快照有缺失时整体失败，不返回部分结果。
- `InMemoryAsOfProvider`：只用于第一步测试和明确给出的离线夹具。最新生效
  版本尚不可用时默认fail-closed，不悄悄退回旧合约表。

运行第一步测试：

```bash
export PYTHONPATH=.
python tests/run_datahub.py
```

该测试不需要pandas/pyarrow，也不访问任何真实数据。预期输出以`D0通过`开头。

## 已有第四类数据积木与文件适配器

`datahub/role_prices.py`保存四角色因果复权价；`datahub/core.py`中的
`MinimalDataHub`暂时是第四类只读角色快照门面。它们是已有M1/M2功能，
**尚未并入通用D0多数据集查询门面**。后续在真正需要多品种查询时再统一
接口，不提前实现不使用的功能。

`examples/role_cross/local_input.py`仅用于现有第四类真实文件探针。
它按这批样本现有的Feather格式准备输入，不属于`datahub`包或其稳定API；
其他文件类型、数据库等来源直接提供标准记录/Provider，无需修改DataHub。
运行现有第四类探针：

```bash
python tests/run_role_research.py
python tests/run_role_research.py --real
```

第二条默认读取`/workspace/worker/pj/neutron/tests/temp/role/`中的
`fut_contract_data.feather`、`fut_adjustment_factors.feather`和
`fut_basic.feather`。后者不包含最小变动价位，真实Bar探针须显式传入
`--price-increment`。必须在拥有真实Feather文件的环境运行；主力pcr因子的方向、
记录日期与累计锚点仍待真实数据确认。

## 后续按需推进，不预先宣称完成

| 阶段 | 触发需求 | 待实现/验收 |
|---|---|---|
| D1 | 第四类真实角色数据查询 | 角色/因子接入通用Provider，明确每条记录真实发布时间；源格式可切换 |
| D2 | 多日主力及四角色信号 | 四角色专属因子、旧/新合约价格、换约点连续性与pcr对账的真实文件验收 |
| D3 | 正式回测创建真实Instrument | `fut_basic`元数据标准化、历史有效区间、费用和乘数校验 |
| D4 | 多源与实盘 | 增量更新、版本/重启、数据修订、质量状态及新旧Provider一致性测试 |

M3策略信号和M4正式模拟回测仍属第四类策略阶段，不因D0通过而自动完成。

# 第四类策略接入进度（M3/M4）

- M3a：`examples/role_cross/role_signal.py`实现四角色复权价穿越信号；
  `role_strategy.py`将只读DataHub快照转为`rb_main`逻辑目标。纯信号夹具已通过，
  Runner/Recording完整装配测试见`tests/run_role_cross.py --stage 2`，待
  含Bomber环境验证。
- M3b：`tests/strategies/role_cross/run_bar_probe.py --day YYYY-MM-DD`已用
  2026-07-28真实RB分钟Bar取得1,317根Bar和264个完整帧，价差全负故无信号。
- M4a/M4b：`tests/run_role_cross.py --stage 3/4`分别验证动态换月与模拟成交；
  `examples/role_cross/run_backtest.py`是多日真实Bar基础模拟回测入口，
  代码已接入，服务器测试尚待运行。
- 尚未完成：受控前填、真实跨换月成交、上期所高保真结算及pcr因子口径核对。
  不应将M3探针视为正式回测绩效。
