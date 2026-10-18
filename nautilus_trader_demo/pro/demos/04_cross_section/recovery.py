"""04同交易日、已结束订单的组合恢复；不接管未知仓位。"""

from bomber.framework.trader.runtime.live.channels.ctp_recovery import (
    CtpResumeLifecycle, validate_ctp_account, validate_ctp_restored_positions)
from bomber.framework.trader.runtime.live.recovery import restored_target_revision


def validate_restored_session(session, day):
    runner = session.runner
    keys = {str(i) for i in runner.fixed_ids}
    targets = runner.target_store.all()
    if set(targets) != {session.strategy.strategy_id}:
        raise RuntimeError("恢复文件不是当前单策略组合")
    target = targets[session.strategy.strategy_id]
    environment = "replay" if runner.replay_md_trading_day else "realtime"
    if (target.metadata.get("simnow_environment", "realtime") != environment or
            target.metadata.get("replay_md_trading_day") != runner.replay_md_trading_day):
        raise RuntimeError("恢复文件SimNow环境或原始MD交易日不一致")
    if set(target.targets) != keys:
        raise RuntimeError("恢复文件合约与当前主力组合不一致")
    validate_ctp_restored_positions(session, day, session.recovery_binding)
    revision = restored_target_revision(session, session.recovery_binding)
    # 目标版本延续；动量窗口重新预热，不能重置到revision=1。
    with session.strategy._revision_lock:
        session.strategy._revision = revision
    return revision


def validate_account(session, transport):
    return validate_ctp_account(session, transport, session.recovery_binding)


class ResumeLifecycle(CtpResumeLifecycle):
    def validate_restored_session(self, session, day):
        return validate_restored_session(session, day)
