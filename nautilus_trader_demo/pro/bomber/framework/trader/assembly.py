"""共同的策略组装声明；运行模式与路由政策由调用方明确选择。"""

from dataclasses import dataclass
from collections.abc import Callable

from bomber.framework.market.basic.base import DataType
from .contracts import DataBinding, ExecutionRoute
from .live_roles import FixedRoleLiveRunner


@dataclass(frozen=True)
class RoleGuard:
    references: object
    product: str
    role: str
    processed_ns: Callable[[], int]


@dataclass(frozen=True)
class StrategyBindings:
    data: tuple
    execution: tuple
    role_guard: RoleGuard | None = None
    time_feed_ids: tuple = ()


def assemble_strategy(runner, strategy, *, feeds, execution, bindings):
    """两种入口共用：声明Feed、执行组件、数据绑定及目标路由。

    固定角色守卫交给FixedRoleLiveRunner；历史动态路由仍经原Runner注册。
    不推断策略内部属性，不启动连接，不改变回调／撮合顺序。
    """
    if runner.position_manager is not execution.positions:
        raise ValueError("Runner与执行积木须共享PositionManager")
    guard = bindings.role_guard
    fixed = isinstance(runner, FixedRoleLiveRunner)
    if fixed != (guard is not None):
        raise ValueError("固定角色Runner须显式声明RoleGuard，其他Runner不接受该守卫")
    if guard is not None:
        if len(bindings.data) != 1 or len(bindings.execution) != 1 or bindings.time_feed_ids:
            raise ValueError("固定角色绑定仅支持单Bar数据源及单静态执行路由")
        data, route = bindings.data[0], bindings.execution[0]
        if (not isinstance(data, DataBinding) or not isinstance(route, ExecutionRoute)
                or data.instrument_id != route.instrument_id
                or data.data_type is not DataType.BAR
                or data.data_key != str(data.instrument_id)
                or route.client_id != execution.client.client_id):
            raise ValueError("固定角色数据与真实路由不匹配")
        session_references = getattr(runner, "references", guard.references)
        if session_references is not guard.references:
            raise ValueError("会话Runner与RoleGuard须共享参考服务")
    runner.add_market_observer(execution.prices)
    for feed_id, feed in feeds.items():
        runner.add_data_feed(feed_id, feed)
    runner.add_execution_client(execution.client)
    if guard is None:
        runner.add_strategy(strategy, data_bindings=bindings.data,
            execution_routes=bindings.execution, time_feed_ids=bindings.time_feed_ids)
    else:
        runner.add_role_strategy(strategy, feed_id=data.feed_id,
            instrument_id=data.instrument_id, client_id=route.client_id,
            references=guard.references, product=guard.product, role=guard.role,
            target_key=route.target_key, processed_ns=guard.processed_ns, bar_spec=data.bar_spec)
    return runner
