# 第四类：RB四角色研究数据与穿越策略（M1～M4）

M1/M2建立“真实合约Bar + 前一交易日角色表 → 四条因果复权研究价”。
M3a提取`example06/07`共有的穿越信号：主力角色复权价由下向上穿越
四角色复权均价时目标为+1手，由上向下穿越时为-1手；第一帧只预热。
策略只提交逻辑`rb_main`目标，不自行选择真实下单合约。研究价不会成为
可成交Bar。M4由Runner动态路由负责先平旧主力、再开新主力；模拟订单
由Planner/Risk/Backend处理，策略不接触订单或合约选择。

模块归属：`datahub/core.py`提供策略只读的`RoleSnapshot`和最小门面，
`datahub/role_prices.py`负责因果角色与独立复权计算，
`examples/role_cross/local_input.py`只负责这次试验的文件加载，不是DataHub API；
`datahub/temporal.py`提供格式无关的Provider与时间契约。将来正式DataHub
替换文件Provider即可；
`trader/dynamic_routes.py`仍只负责真实目标路由和换月，不读复权数据。
`examples/role_cross/role_signal.py`是纯信号，`role_strategy.py`是策略模板适配层；
`tests/strategies/role_cross/run_bar_probe.py`只做真实Bar到记录型客户端
的M3b探针；`examples/role_cross/run_backtest.py`才是基础模拟回测装配。
这两个模块使用带角色前缀的文件名，因为直接运行探针时，其目录位于Python
模块搜索路径最前；命名为`signal.py`或`strategy.py`会遮蔽标准库或项目包。

角色表采用`fut_contract_data.feather`的`recent/main/second/far`，对应
`near/main/secondary/far`。`fut_adjustment_factors.feather`的主力
`pcr_factor/pcr_cumfactor`用于核对口径，不共用给其余三个角色。角色发生
切换时，对每个角色分别用来源交易日旧、新真实合约收盘价计算：

```text
single = old_close / new_close
cumulative = previous_cumulative * single
adjusted_close = raw_close * cumulative
```

当前采用从样本起点向前累计的锚点1。若外部累计因子以不同锚点归一，不能
直接比较两者绝对值，应先核对相邻日比值。因子记录日期可能是价格来源日或
实际生效日，所以真实文件探针同时显示两种对齐结果和因子方向；在拿到输出
并确认口径前，不把pcr一致性标为通过。合约表每条记录必须来自当前交易日
以前。文件没有显式发布时间，因此最早可用时间保守设置为当前交易日第一根
真实Bar时间；正式DataHub应返回真实`available_ns`。

在`pro`目录运行：

```bash
export PYTHONPATH=.
python tests/run_role_research.py
python tests/run_role_research.py --real
python tests/run_role_cross.py --stage 1
python tests/run_role_cross.py --stage 2
python tests/strategies/role_cross/run_bar_probe.py --day 2026-07-28 --bars-dir /workspace/data/dev/kd/intelkit/records/raw_data/cn_futures/20260728 --price-increment 1
python tests/run_role_cross.py --stage 3
python tests/run_role_cross.py --stage 4
python examples/role_cross/run_backtest.py --start-day 2026-07-28 --end-day 2026-07-28 --bars-dir /workspace/data/dev/kd/intelkit/records/raw_data/cn_futures/20260728 --price-increment 1
```

`--real`默认读取`/workspace/worker/pj/neutron/tests/temp/role/`中的三张
新表；Bar目录仍默认使用`/workspace/worker/kdwk/.../test_data/bars/mo`。
四个路径均可用`--bars-dir`、`--fut-contract`（兼容旧名
`--contract-struct`）、`--factors`、`--fut-basic`覆盖。
新版`fut_basic`没有`minChgPriceNum`，因此真实Bar探针要求显式
`--price-increment`；上面的`1`仅供已核实RB最小变动价位为1时使用，
它不是从新表推断出来的规则，也不意味着已验证正式撮合精度。
Bar文件需命名为`rb合约_YYYYMMDD.feather`，含`datetime`或`timestamp`、
`close`以及可选的`trading_day`/`date`列。无时区时间按上海时间解释；
有时区时间转换为UTC纳秒。真实文件输出应检查：跨日覆盖、主力切换次数、
`pcr_factor`是直接还是倒数、记录日期的对齐方向、缺失旧新合约价格。

M3a阶段1的纯信号夹具已在本机通过；M3a阶段2需在含Bomber/Nautilus
依赖的环境运行。M3b真实Bar探针需指定实际具有四角色Bar的交易日；
`complete_frames=0`表示当日没有同分钟四角色完整帧，不能算通过。
探针还会输出`first/last/min/max`价差及正、负、零帧数：若264个完整帧
始终在零轴同一侧，`signals=0`是穿越规则的正常结果；若价差同时出现正、
负两侧却无信号，探针会报错，而不会把静默零请求当作通过。
目前严格要求同分钟四张真实Bar全部到达，尚未采用`example06`的受控
前填逻辑；`signals=0`也可能只是当日没有发生穿越。探针的固定
`ExecutionRoute`和`RecordingExecutionClient`不代表真实主力换月或订单成交。

M4a合成测试验证无新穿越时保留逻辑目标，并在旧仓归零前禁止开新主力；
M4b合成测试验证穿越目标进入原生模拟订单和成交。两者需在服务器的
Bomber/Nautilus环境验收。`run_backtest.py`使用真实未复权合约Bar、
`CtpFuturesBasicProfile`、Planner、Risk和模拟Backend；若一天的价差始终
为负，它应报告0信号、0订单，而非伪造成交。要验证真实换月和成交，需
提供覆盖角色切换与穿越的多日数据。为平旧仓，切换日也必须有旧主力真实Bar。
当前Basic Profile不代表上期所完整平今/平昨和结算规则；pcr因子口径及
角色换约连续性也未完成真实验收。Tick输入将来先经现有CTP解析和Bar聚合，
再复用同一DataHub与策略接口。
