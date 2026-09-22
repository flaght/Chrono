# 第三类：期货/期权联动与安全换月

`FuturesOptionSignalStrategy`只订阅连续合约的逻辑Bar，用快慢均线输出
`future/call/put`三个**逻辑目标键**。RB可同时交易期货与方向期权；IM配置
`trade_futures=False`时只输出MO方向期权目标，不会下IM期货单。策略不选择真实
合约、不读DataHub、不撤单，也不判断账户仓位。

执行侧新增`ContractResolverPort`。当前`ScheduledContractResolver`读取显式的
`ContractAssignment(effective_ns, available_ns, revision)`；未来DataHub只需实现
同一`resolve(target_key, as_of_ns)`接口。最新已生效版本若尚未可用，必须拒绝
解析，不能退回可能已过期的旧合约。

`SafeRollCoordinator`按`ACTIVE → CANCELING → CLOSING → ACTIVE`推进；先请求撤旧单，
在旧合约在途量归零后发旧合约零目标，旧仓和在途量均归零后才打开新合约。
换月中收到的新信号只覆盖待执行数量。状态可导出/恢复，但生产环境仍须把它接入
既有持久化检查点和启动对账流程。

统一Runner目前以`DynamicExecutionRoute`支持**单逻辑目标、独占客户端**的
HISTORICAL安全换月验证；LIVE模式明确拒绝动态路由，直到回报驱动、持久化及
重启恢复通过专项验收。固定`ExecutionRoute`及前两类策略保持原样。为避免误把多腿目标
当作原子成交，期货+Call+Put三腿的动态执行、选权、裸腿限制和失败补偿尚未
放开；当前测试只证明信号层和单腿动态路由的边界。真实主力与期权选约仍等
DataHub及合约主数据。

单角色的积木装配形态如下（`assignments`由测试显式给出，将来由DataHub提供）：

```python
resolver = ScheduledContractResolver(tuple(assignments))
route = DynamicExecutionRoute("rb_main", client.client_id, resolver)
runner.add_strategy(strategy, data_bindings=bindings, execution_routes=(route,))
```

策略仅提交`set_target("rb_main", quantity, ts_event)`。Runner持有最新逻辑目标，
选约变化时撤旧单、发旧合约零目标，账户旧仓和在途量均为零后才发新合约目标。
当前通过行情事件或显式`runner.refresh_dynamic_routes(as_of_ns)`推进；尚未自动
绑定所有柜台回报回调，也未完成生产级状态持久化。

在`pro`目录、项目环境中逐步执行：

```bash
export PYTHONPATH=.
python tests/run_dynamic_routes.py
python tests/strategies/futures_option/run_test.py
python tests/run_strategy.py
python tests/run_portfolio.py
```

前两个测试都不连接网络、不使用真实合约行情，也不真实下单。第三类正式回测
需要下一阶段补齐多腿执行契约与模拟期权合约数据，不能把这些测试当作完整
example05策略绩效验收。
