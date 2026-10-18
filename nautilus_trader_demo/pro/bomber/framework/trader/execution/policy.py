"""执行账本来源与物理账户观测政策；不依赖行情、柜台或策略算法。"""

from dataclasses import dataclass
from enum import Enum


class AccountReconciliationMode(str, Enum):
    STRICT = "strict"
    OBSERVE = "observe"
    OFF = "off"


class StrategyPositionSource(str, Enum):
    FILLS = "fills"
    MODEL = "model"


@dataclass(frozen=True)
class ExecutionPolicy:
    account_reconciliation: AccountReconciliationMode = AccountReconciliationMode.STRICT
    position_source: StrategyPositionSource = StrategyPositionSource.FILLS

    def __post_init__(self):
        object.__setattr__(self, "account_reconciliation", AccountReconciliationMode(self.account_reconciliation))
        object.__setattr__(self, "position_source", StrategyPositionSource(self.position_source))
        if self.position_source is StrategyPositionSource.MODEL and self.account_reconciliation is not AccountReconciliationMode.OFF:
            raise ValueError("模型策略账本不关联柜台；账户观测须配置在独立账户执行端")

    def snapshot_state(self):
        return {"version": 1, "account_reconciliation": self.account_reconciliation.value,
                "position_source": self.position_source.value}

    def restore_state(self, state):
        if state != self.snapshot_state():
            raise ValueError("恢复时执行账本来源或账户对账政策不一致")


def add_execution_policy_arguments(parser):
    parser.add_argument("--account-reconciliation", choices=tuple(m.value for m in AccountReconciliationMode),
                        help="strict关联柜台仓位；observe只观测；off不查询柜台仓位")
    parser.add_argument("--strategy-position-source", choices=tuple(m.value for m in StrategyPositionSource),
                        help="fills归属成交；model独立参考价模拟（须--mode model）")


def execution_policy_from_args(args):
    model = args.mode == "model"
    policy = ExecutionPolicy(getattr(args, "account_reconciliation", None) or ("off" if model else "strict"),
                             getattr(args, "strategy_position_source", None) or ("model" if model else "fills"))
    if model != (policy.position_source is StrategyPositionSource.MODEL):
        raise ValueError("model仓位来源须与--mode model同时选择")
    if getattr(args, "adopt_existing_position", None) and policy.account_reconciliation is not AccountReconciliationMode.STRICT:
        raise ValueError("显式接管柜台仓位仅适用于strict政策")
    return policy


def observed_account_positions(client, instruments, *, positions=None):
    """报告物理账户事实；off返回None，不能把独立执行账本标为真实账户。"""
    mode = getattr(client, "account_reconciliation", AccountReconciliationMode.STRICT)
    if mode == AccountReconciliationMode.OFF:
        return None
    if mode == AccountReconciliationMode.OBSERVE:
        snapshot = client.account_observation
        return None if snapshot is None else {str(i): snapshot.positions.get(i, 0) for i in instruments}
    positions = positions if positions is not None else getattr(client, "position_manager", None)
    if positions is None:
        return None
    return {str(i): positions.account_position(client.client_id, i) for i in instruments}
