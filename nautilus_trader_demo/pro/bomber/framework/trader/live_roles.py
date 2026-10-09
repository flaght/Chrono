"""单策略固定角色在线路由及会话守卫；不依赖具体策略或柜台实现。"""

from dataclasses import dataclass
import time

from bomber.framework.datahub.sector_roles import SectorDataUnavailable
from bomber.framework.market.basic.base import Bar, DataType
from .contracts import DataBinding, ExecutionRoute, RuntimeMode
from .runner import UnifiedStrategyRunner


@dataclass(frozen=True)
class _RoleBinding:
    strategy: object
    feed_id: str
    instrument_id: object
    client_id: str
    references: object
    product: str
    role: str
    target_key: str
    processed_ns: object


class FixedRoleLiveRunner(UnifiedStrategyRunner):
    """本根Bar处理后推进保留目标；固定角色合约变化即闭闸。

    单策略、单角色、固定真实路由。参考服务、角色、逻辑目标键及已处理
    时间读取器全部显式传入，不读取策略config或引用demo。LIVE动态换约
    继续禁用，不提供连接、交易授权或跨进程恢复。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mode is not RuntimeMode.LIVE:
            raise ValueError("FixedRoleLiveRunner仅用于LIVE固定角色装配")
        self._last_live_bar_ns = {}
        self._role_binding = None
        self._role_change_reason = None

    @property
    def bound_instrument(self):
        return None if self._role_binding is None else self._role_binding.instrument_id

    def add_role_strategy(self, strategy, *, feed_id, instrument_id, client_id,
                          references, product, role, target_key, processed_ns,
                          bar_spec="1-MINUTE"):
        if self._role_binding is not None:
            raise ValueError("固定角色在线装配仅支持一个策略")
        if not callable(processed_ns) or not target_key:
            raise ValueError("须显式提供处理时间读取器和逻辑目标键")
        if role not in {"main", "secondary", "near", "far"}:
            raise ValueError("角色须为main/secondary/near/far")
        self.add_strategy(strategy, data_bindings=(DataBinding(
            str(instrument_id), feed_id, instrument_id, DataType.BAR, bar_spec),),
            execution_routes=(ExecutionRoute(target_key, client_id, instrument_id),))
        self._role_binding = _RoleBinding(strategy, feed_id, instrument_id, client_id,
            references, product.strip().upper(), role, target_key, processed_ns)

    def _validate_configuration(self):
        super()._validate_configuration()
        if self._role_binding is None or len(self._registrations) != 1:
            raise ValueError("请用add_role_strategy装配单策略固定角色路由")

    def close_role_gate(self, reason):
        self._role_change_reason = reason
        binding = self._role_binding
        self.position_manager.mark_recovery_required(binding.client_id)
        client = self._clients[binding.client_id]
        disarm = getattr(client, "disarm", None)
        if callable(disarm):
            disarm(reason)
        client.cancel_strategy(binding.strategy.strategy_id)
        raise RuntimeError(reason)

    def adopt_initial_position(self, quantity):
        """柜台核验后、Runner启动前，把已有仓位归属到明确绑定的唯一策略。"""
        binding = self._role_binding
        if self._started or binding is None:
            raise RuntimeError("须在固定角色绑定完成、Runner启动前接管仓位")
        if self.position_manager.account_position(binding.client_id, binding.instrument_id) != quantity:
            raise RuntimeError("接管策略仓位前须与已对账的权威账户净仓一致")
        self.position_manager.attribution.adopt(binding.client_id, binding.instrument_id,
            binding.strategy.strategy_id, binding.target_key, quantity)

    def publish(self, feed_id, event):
        if not isinstance(event, Bar) or self._role_binding is None:
            return super().publish(feed_id, event)
        with self._submit_lock:
            binding = self._role_binding
            if feed_id != binding.feed_id:
                return super().publish(feed_id, event)
            if self._role_change_reason is not None:
                raise RuntimeError(self._role_change_reason)
            key = (feed_id, event.bar_type.instrument_id)
            if event.ts_event <= self._last_live_bar_ns.get(key, -1):
                return
            self._last_live_bar_ns[key] = event.ts_event
            reference_ready = False
            try:
                assignment = binding.references.snapshot(event.ts_event)
                symbol = assignment.instrument(binding.product, binding.role).lower()
                if symbol != str(binding.instrument_id.symbol).lower():
                    self.close_role_gate(f"当前角色{binding.role}已由{binding.instrument_id.symbol}变为{symbol}；"
                        "固定路由已闭闸，须核对仓位与活动订单并重新装配，LIVE动态换约尚未启用")
                assignment.factor(binding.product, binding.role)
                reference_ready = True
            except SectorDataUnavailable:
                pass
            super().publish(feed_id, event)
            if self._started and reference_ready and binding.processed_ns() == event.ts_event:
                self.continue_execution_target(binding.strategy.strategy_id, binding.target_key,
                    event.ts_event, trigger_instrument_id=event.bar_type.instrument_id)


class SessionRoleLiveRunner(FixedRoleLiveRunner):
    """注入会话核验，排除部分分钟，检查Bar时效及固定合约条款。

    会话回调由通道组装层提供（例如真实MD／TD及健康检查）；本组件不识别
    SimNow端口或回放日期。参考源遵循snapshot／refresh／spec接口。
    """

    def __init__(self, *, references, session_check, orders, clock_ns=None, **kwargs):
        super().__init__(RuntimeMode.LIVE, **kwargs)
        self.references = references
        self.orders = orders
        self._session_check = session_check
        self._clock_ns = clock_ns or (lambda: time.time_ns())
        self.expected_day = references.trading_day.strftime("%Y%m%d")
        self.fixed_spec = references.spec
        minute = 60_000_000_000
        self.first_complete_bar_ns = ((references.started_ns + minute - 1) // minute + 1) * minute - 1
        self.failure = None
        self.accept_bars = False

    def session_ready(self):
        return self._session_check()

    def begin_bars(self):
        minute = 60_000_000_000
        now = self._clock_ns()
        self.first_complete_bar_ns = max(self.first_complete_bar_ns,
            ((now + minute - 1) // minute + 1) * minute - 1)
        self.accept_bars = True

    def publish(self, feed_id, event):
        if not isinstance(event, Bar) or not self.accept_bars:
            return
        if event.ts_event < self.first_complete_bar_ns:
            return
        try:
            if not self.session_ready():
                raise RuntimeError("MD／TD会话失效、交易日不一致或行情停滞")
            if not 0 <= self._clock_ns() - event.ts_event <= 120_000_000_000:
                raise RuntimeError("分钟Bar与墙钟不匹配；默认不接受历史回放")
            self.references.refresh()
            if self.references.spec != self.fixed_spec:
                self.close_role_gate("角色合约或条款变更，固定路由闭闸并要求重新核对")
            if not self.session_ready():
                raise RuntimeError("参考资料刷新后MD／TD会话已失效或行情停滞")
            if not 0 <= self._clock_ns() - event.ts_event <= 120_000_000_000:
                raise RuntimeError("参考资料刷新后分钟Bar已过期")
            return super().publish(feed_id, event)
        except Exception as error:
            self.failure = str(error)
            self.accept_bars = False
            if self.orders and self._role_binding is not None:
                self._clients[self._role_binding.client_id].disarm("bar_dispatch_failed")
            raise
