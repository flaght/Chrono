"""渠道无关的持仓归属与恢复约束；不读取策略信号或连接柜台。"""

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class InstrumentPositionScope:
    """显式声明本次执行管理的合约；范围外账户仓位不产生策略归属。"""
    instruments: frozenset

    def __post_init__(self):
        if isinstance(self.instruments, str):
            raise ValueError("持仓范围须为非空合约集合")
        raw = tuple(self.instruments)
        instruments = frozenset(str(i) for i in raw)
        if not instruments or any(i is None or not str(i).strip() for i in raw):
            raise ValueError("持仓范围须为非空合约集合")
        object.__setattr__(self, "instruments", instruments)

    def contains(self, instrument):
        return str(instrument) in self.instruments

    def select(self, positions):
        return {key: value for key, value in positions.items() if self.contains(key)}

    def external(self, positions):
        return {key: value for key, value in positions.items() if not self.contains(key)}


@dataclass(frozen=True)
class StrategyPositionBinding:
    strategy_id: str
    client_id: str
    targets: Mapping[str, object]  # 逻辑目标键 -> 标准InstrumentId，由入口显式装配。

    def __post_init__(self):
        targets = dict(self.targets)
        if (not self.strategy_id or not self.client_id or not targets
                or any(not isinstance(key, str) or not key for key in targets)
                or any(instrument is None for instrument in targets.values())
                or len(set(targets.values())) != len(targets)):
            raise ValueError("持仓绑定须明确唯一策略、执行客户端及合约路由")
        object.__setattr__(self, "targets", MappingProxyType(targets))


class SessionIdentityCheckpoint:
    """保存入口声明的会话身份；恢复不能偷偷改变路由、时间语义或策略参数。"""
    def __init__(self, identity):
        self.identity = deepcopy(dict(identity))

    def snapshot_state(self):
        return {"version": 1, "identity": deepcopy(self.identity)}

    def restore_state(self, state):
        if state != self.snapshot_state():
            raise ValueError("恢复会话身份、合约、参考版本或策略参数不一致")


def validate_owned_positions(positions, binding, *, account_scope=None):
    """绑定范围内账户仓位须能解释为本策略归属；范围外仍保留账户事实。"""
    snapshot = positions.snapshot()
    allowed = set(binding.targets.values())
    if account_scope is not None and account_scope.instruments != frozenset(str(i) for i in allowed):
        raise RuntimeError("持仓范围与策略执行路由不一致")
    if any(sid != binding.strategy_id or key not in binding.targets
           for sid, key in snapshot.strategy_positions):
        raise RuntimeError("恢复文件存在其他策略或逻辑仓位")
    if any(key.client_id != binding.client_id or (account_scope is None and key.instrument_id not in allowed)
           for key in snapshot.account_positions):
        raise RuntimeError("恢复文件存在其他执行账户或合约")
    result = {}
    for target_key, instrument in binding.targets.items():
        owned = positions.position(binding.strategy_id, target_key)
        account = positions.account_position(binding.client_id, instrument)
        if owned != account or positions.unassigned_position(binding.client_id, instrument):
            raise RuntimeError("恢复时策略归属与柜台净仓不一致或存在未归属仓位")
        result[instrument] = owned
    return result


def validate_finished_orders(session):
    if any(session.runner.position_manager.snapshot().working_quantities.values()) or any(
            not order.status.is_terminal for order in session.client.order_state_machine.states()):
        raise RuntimeError("恢复要求原会话活动订单已结束")


def restored_target_revision(session, binding, *, allow_empty=False):
    targets = session.runner.target_store.all()
    if not targets and allow_empty:
        if session.runner.portfolio_coordinator.state().strategy_revisions:
            raise RuntimeError("恢复文件缺少对应策略目标")
        return 0
    if set(targets) != {binding.strategy_id}:
        raise RuntimeError("恢复文件不是当前单策略目标")
    target = targets[binding.strategy_id]
    if set(target.targets) != set(binding.targets):
        raise RuntimeError("恢复文件目标键与当前合约路由不一致")
    portfolio = session.runner.portfolio_coordinator.state()
    if portfolio.strategy_revisions != {binding.strategy_id: target.revision}:
        raise RuntimeError("恢复文件目标与组合版本不一致")
    contributions = portfolio.contributions
    if set(contributions) != {binding.strategy_id}:
        raise RuntimeError("恢复文件组合贡献存在其他策略")
    expected = {instrument: target.targets[key] for key, instrument in binding.targets.items()}
    contribution = contributions[binding.strategy_id]
    if (any(key.client_id != binding.client_id or key.instrument_id not in expected
            for key in contribution) or
            {key.instrument_id: quantity for key, quantity in contribution.items()} != expected):
        raise RuntimeError("恢复文件组合贡献与当前执行路由或目标不一致")
    return target.revision
