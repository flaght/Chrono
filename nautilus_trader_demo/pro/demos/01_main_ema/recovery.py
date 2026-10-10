"""01单次会话恢复：保留仓位与目标版本，EMA用本次完整分钟重新预热。"""

import time
from bomber.framework.trader.runtime.live.recovery import restored_target_revision
from bomber.framework.trader.runtime.live.channels.ctp_recovery import (
    validate_ctp_account, validate_ctp_restored_positions)


def recovery_identity(args, references, policy):
    assignment = references.snapshot(time.time_ns())
    product = args.product.strip().upper()
    return {
        "strategy_id": "main-ema-simnow", "client_id": "main-ema-simnow",
        "product": product, "instrument": str(references.instrument_id),
        "target_key": f"{product.lower()}_main",
        "trading_day": references.trading_day.strftime("%Y%m%d"),
        "simnow_environment": policy.environment,
        "replay_md_trading_day": policy.replay_md_trading_day,
        "source_day": str(assignment.source_day), "factor_date": str(references.factor_date),
        "factor_basis": references.factor_date_basis,
        "factor": str(assignment.factor(product, "main")),
        "tick": str(references.spec.tick), "multiplier": str(references.spec.multiplier),
        "currency": references.spec.currency, "venue": references.spec.venue,
        "listed": str(references.spec.listed), "last_trade": str(references.spec.last_trade),
        "fast": args.fast, "slow": args.slow, "quantity": str(args.quantity),
    }


def validate_restored_session(session, day):
    binding = session.recovery_binding
    validate_ctp_restored_positions(session, day, binding)
    revision = restored_target_revision(session, binding, allow_empty=True)
    owned = session.runner.position_manager.position(binding.strategy_id, session.strategy.config.target_key)
    if abs(owned) > session.strategy.config.quantity:
        raise RuntimeError("恢复仓位超过本策略目标手数上限")
    # 恢复执行事实；本次重新计算EMA，首个新目标版本接在原版本后面。
    with session.strategy._revision_lock:
        session.strategy._revision = revision
    session.runner.hold_restored_target(binding.strategy_id, revision)
    return revision


def verify_managed_position(session, transport):
    strategy = session.strategy
    if (strategy.bars_used < strategy.config.slow_period or strategy.last_target is None
            or session.runner.failure or session.client.report_errors):
        raise RuntimeError("接管/恢复后尚未取得有效新EMA目标或存在会话错误")
    validate_ctp_account(session, transport, session.recovery_binding)
    instrument = session.references.instrument_id
    if (strategy.position(strategy.config.target_key) != strategy.last_target or
            session.runner.position_manager.working_quantity(session.client.client_id, instrument)):
        raise RuntimeError("接管/恢复后最新EMA目标尚未完成")
