"""按交易窗口续接同一EMA状态；休市不执行策略，恢复必须权威对账。"""

from bomber.framework.trader.runtime.ctp_account import position_buckets
from contextlib import ExitStack, nullcontext
from copy import copy
from datetime import datetime
from decimal import Decimal
import json
from pathlib import Path
import signal
import threading
import time

from bomber.framework.trader.execution.ctp.ledger import CtpLedgerState, CtpPositionSnapshot
from bomber.framework.trader.live_roles import SessionRoleLiveRunner
from bomber.framework.trader.persistence import JsonStateStore
from bomber.framework.trader.runtime.live.channels.ctp import CtpSessionLifecycle, account_lock, occupied_positions
from bomber.framework.trader.runtime.live.contracts import LiveRunContext
from bomber.framework.trader.runtime.trading_sessions import CopperSessions, SHANGHAI
from bomber.framework.market.basic.base import Bar


def recover_minutes(checkpoint, calendar, before_ns, path):
    if not checkpoint.strategy.bars_used:
        return
    if checkpoint.strategy.last_processed_ns >= before_ns:
        raise RuntimeError("EMA断点晚于恢复边界，拒绝未来状态或行情倒序")
    missing = list(calendar.missing_minutes(checkpoint.strategy.last_processed_ns, before_ns))
    if not missing:
        return
    required = set(missing)
    rows = {}
    if path:
        with Path(path).open() as handle:
            for line in handle:
                item = json.loads(line)
                stamp = item["ts_event"]
                if stamp in required:
                    if stamp in rows or item["instrument_id"] != checkpoint.instrument:
                        raise ValueError("恢复分钟重复或真实合约不一致")
                    price = Decimal(str(item["adjusted_close"]))
                    if not price.is_finite() or price <= 0:
                        raise ValueError("恢复分钟复权价无效")
                    rows[stamp] = price
    absent = [stamp for stamp in missing if stamp not in rows]
    if absent:
        raise RuntimeError(f"EMA恢复缺少{len(absent)}根完整交易分钟，首根ts_event={absent[0]}；"
                           "保持闭闸，须提供--recovery-bars权威完整分钟，不能用当前价格补造")
    for stamp in missing:
        checkpoint.warm_closed(stamp, rows[stamp])


def orders_for_day(payload, day):
    return sum(item["client_order_id"].rsplit("-", 2)[-2] == day
        for rows in payload.get("orders", {}).values() for item in rows)


class CheckpointedRunner(SessionRoleLiveRunner):
    """Bar更新与提交/回报保存共享执行锁，确保EMA和目标同代快照。"""
    def publish(self, feed_id, event):
        with self._submit_lock:
            if (isinstance(event, Bar) and not self.accept_bars
                    and getattr(self, "history_buffering", False)):
                if len(self.history_buffer) >= 10000:
                    self.failure = "历史加载期间实时缓冲溢出"
                    raise RuntimeError(self.failure)
                self.history_buffer.append(event)
                return
            client = next(iter(self._clients.values()))
            with getattr(client, "_submit_lock", nullcontext()):
                try:
                    if (isinstance(event, Bar) and self.accept_bars and event.ts_event >= self.first_complete_bar_ns
                            and event.ts_event > self.ema_checkpoint.strategy.last_processed_ns):
                        boundary = event.ts_event // 60_000_000_000 * 60_000_000_000
                        if getattr(self, "history_service", None) is not None:
                            from .history import warm_history
                            warm_history(self, boundary, phase="recovery")
                        else:
                            recover_minutes(self.ema_checkpoint, self.calendar, boundary, self.recovery_bars)
                    result = super().publish(feed_id, event)
                    if self.accept_bars and self.manager is not None:
                        self.manager.save()
                    return result
                except Exception as error:
                    self.failure = str(error)
                    self.accept_bars = False
                    disarm = getattr(client, "disarm", None)
                    if callable(disarm):
                        disarm("continuous_bar_or_checkpoint_failure")
                    raise

    def mark_history_live_start(self):
        minute = 60_000_000_000
        with self._submit_lock:
            now = self._clock_ns()
            self.first_complete_bar_ns = max(self.first_complete_bar_ns,
                ((now + minute - 1) // minute + 1) * minute - 1)

    def begin_bars(self):
        if getattr(self, "history_service", None) is None:
            return super().begin_bars()
        from .history import flush_buffer
        with self._submit_lock:
            flush_buffer(self)
            if self.failure or not self.session_ready():
                raise RuntimeError(self.failure or "历史/实时衔接后MD/TD不健康")
            self.history_buffering = False
            self.accept_bars = True


class ContinuousLifecycle(CtpSessionLifecycle):
    def __init__(self, *args, state_file, window, stop_event, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_file, self.window, self.stop_event = state_file, window, stop_event
        self.resuming = Path(state_file).exists()

    def prepare(self, resources):
        if not self.resuming:
            return super().prepare(resources)
        for namespace in dict.fromkeys((self.lock_namespace, *self.legacy_lock_namespaces)):
            resources.enter_context(account_lock(self.transport.account_id, namespace=namespace))
        self.driver.start(lambda report: None)
        self._observed_td_day = self.driver.trading_day
        if self.driver.trading_day != self.window.trading_day:
            raise RuntimeError("登录TD交易日与当前日夜盘窗口不一致")
        self.driver.reconcile_account_state()
        # 此时仅查询；关联、策略归属在最终会话restore/start后校验。
        return LiveRunContext(self.driver.trading_day, resources)

    def start(self, session, context):
        if not self.resuming:
            super().start(session, context)
            return
        self.session = session
        session.manager.restore(legacy_component_states=getattr(session, "legacy_component_states", None))
        session.client.start()  # 先恢复活动订单，再查询账户净仓和资金；不继承旧授权。
        self._refresh_ledger(session, context.trading_day)
        session.runner.start()
        deadline = time.monotonic() + self.ready_seconds
        while not session.runner.session_ready():
            if self.stop_event.is_set() or self.lost or time.monotonic() >= deadline:
                raise RuntimeError("恢复后MD/TD未就绪，保持闭闸")
            time.sleep(0.2)
        session.manager.save()
        # 首根完整Bar先检查/补齐断点到当前分钟的缺口，再允许发布新目标。
        if getattr(session.runner, "history_service", None) is not None:
            from .history import warm_history
            session.runner.mark_history_live_start()
            warm_history(session.runner, session.runner._clock_ns(), phase="recovery")
        else:
            recover_minutes(session.runner.ema_checkpoint, session.runner.calendar,
                session.runner._clock_ns(), session.runner.recovery_bars)
        if not session.runner.session_ready():
            raise RuntimeError("历史回补后MD/TD未就绪，保持闭闸")
        session.manager.save()
        session.client.arm_demo(session.client.DEMO_CONFIRMATION)
        session.runner.begin_bars()
        self._next_check = time.monotonic() + self.reconcile_seconds
        print(f"连续会话已恢复：TD={context.trading_day} EMA bars={session.strategy.bars_used} "
              f"策略仓位={session.strategy.position(session.strategy.config.target_key)}", flush=True)

    def _refresh_ledger(self, session, day):
        instrument = session.references.instrument_id
        net = session.runner.position_manager.account_position(session.client.client_id, instrument)
        owned = session.runner.position_manager.position(session.strategy.strategy_id, session.strategy.config.target_key)
        if net != owned or session.runner.position_manager.unassigned_position(session.client.client_id, instrument):
            raise RuntimeError("恢复时柜台净仓与策略归属不一致，不能自动续跑")
        gross = occupied_positions(self.transport)
        expected = {} if not net else {str(instrument): (max(net, 0), max(-net, 0))}
        if gross != expected or abs(net) > 1:
            raise RuntimeError("恢复时全账户总仓与已知单策略一手仓不一致")
        buckets = position_buckets(self.transport.query_position_details(), (instrument,),
            require_cost=True, unique_buckets=True)
        bucket = buckets.get(str(instrument))
        values = bucket.quantities if bucket else (Decimal(0),) * 4
        bases = tuple(cost / (qty * session.references.spec.multiplier) if qty else Decimal(0)
                      for qty, cost in zip(values, bucket.costs if bucket else (Decimal(0),) * 4))
        if (values[0] + values[1], values[2] + values[3]) != expected.get(str(instrument), (0, 0)):
            raise RuntimeError("权威总仓与今昨明细不一致")
        snap = CtpPositionSnapshot(instrument, *values, *bases)
        session.ledger.restore(CtpLedgerState(day, {instrument: snap} if net else {}))

    def poll(self, session):
        if self.stop_event.is_set() or datetime.now(SHANGHAI) >= self.window.end:
            session.runner.accept_bars = False
            session.client.disarm("scheduled_pause_or_stop")
            return False
        return super().poll(session)

    def verify(self, session):
        # 常驻的计划休市/停止不要求每个窗口都有成交；不复用验收的强制成交标准。
        if session.client.report_errors or session.runner.failure:
            raise RuntimeError("连续会话存在未解决的执行/策略错误")

    def shutdown(self, session, context):
        error = None
        try:
            self._save_close_bar(session)
        except Exception as failure:
            error = str(failure)
        result = super().shutdown(session, context)
        if error:
            result["cleanup_errors"].append(error)
        return result

    def _save_close_bar(self, session):
        if session and datetime.now(SHANGHAI) >= self.window.end and not session.runner.failure:
            # 收盘墙钟已越过完整分钟；仅预热最后Bar，休市不发布目标。
            session.runner.accept_bars = False
            session.client.disarm("scheduled_close")
            # 聚合线程持Feed锁再调用Runner；先取出Bar，再获取Runner/客户端锁。
            bars = session.bar_feed.drain_completed(int(self.window.end.timestamp() * 1e9))
            with session.runner._submit_lock, session.client._submit_lock:
                for bar in bars:
                    if (bar.ts_event >= session.runner.first_complete_bar_ns
                            and bar.ts_event > session.strategy.last_processed_ns):
                        boundary = bar.ts_event // 60_000_000_000 * 60_000_000_000
                        if getattr(session.runner, "history_service", None) is not None:
                            from .history import warm_history
                            warm_history(session.runner, boundary, phase="recovery")
                        else:
                            recover_minutes(session.runner.ema_checkpoint, session.runner.calendar,
                                boundary, session.runner.recovery_bars)
                        factor = session.references.snapshot(bar.ts_event).factor(session.strategy.config.product, "main")
                        session.runner.ema_checkpoint.warm_closed(bar.ts_event, bar.close.as_decimal() * factor)


def legacy_ema_state(report_path, checkpoint, persisted, state_file):
    """仅迁移已通过的首次预热窗口；不从旧目标猜测后来EMA变化。"""
    report = json.loads(Path(report_path).read_text())
    targets = persisted.payload["targets"]
    if len(targets) != 1:
        raise ValueError("旧状态须为单EMA目标")
    target, config = targets[0], checkpoint.config()
    meta = target["metadata"]
    if (report.get("status") != "passed" or report.get("signal_source") != "ema"
            or report.get("fixed_instrument") != checkpoint.instrument
            or Path(report["state_file"]).resolve() != Path(state_file).resolve()
            or report.get("initial_position_mode") != "fresh_flat"
            or report["bars_used"] != config["slow"] or report["ema"]["fast"] != config["fast"]
            or report["ema"]["slow"] != config["slow"] or report["ema"]["quantity"] != config["quantity"]
            or str(report["last_target"]) != target["targets"][config["target_key"]]
            or float(report["fast_ema"]) != float(meta["fast_ema"])
            or float(report["slow_ema"]) != float(meta["slow_ema"])
            or report["final_active_orders"] != 0 or report["cleanup_errors"]
            or report["last_order_status"] != "FILLED"):
        raise ValueError("旧报告不能证明准确EMA断点；须使用完整EMA检查点或历史Bar重建")
    orders = [row for rows in persisted.payload["orders"].values() for row in rows]
    if not any(row["client_order_id"] == report["last_order_id"] and row["status"] == "FILLED" for row in orders):
        raise ValueError("旧报告成交与状态文件订单不匹配")
    state = checkpoint.snapshot_state()
    state.update(fast=meta["fast_ema"], slow=meta["slow_ema"], bars_used=report["bars_used"],
        last_processed_ns=target["ts_event"], last_main=meta["research_main"],
        last_target=str(report["last_target"]), revision=target["revision"],
        order_updates_received=report["order_updates_received"], fills_received=report["fills_received"])
    return {"ema": state}


def run_continuous(args, runtime_factory, *, clock=None, stop_event=None):
    calendar = CopperSessions(args.trading_calendar)
    clock = clock or (lambda: datetime.now(SHANGHAI))
    stop = stop_event or threading.Event()
    old_handlers = {}
    if stop_event is None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, lambda *_: stop.set())
    last_status = None
    try:
        # 单独保有常驻进程租约；活动窗口仍持有公共账户锁，防止重复监督器。
        from .run_live import required
        with account_lock(required("CTP_ACCOUNT_ID"), namespace="bomber-main-ema-continuous"):
            while not stop.is_set():
                window = calendar.window(clock())
                persisted = JsonStateStore(args.state_file).load()
                used = orders_for_day(persisted.payload, window.trading_day) if persisted and window else 0
                if window is None or used >= args.max_session_orders:
                    status = "休市等待" if window is None else "本交易日报单额度已用完，等待下一交易日"
                    if status != last_status:
                        print(status, flush=True)
                        last_status = status
                    stop.wait(5)
                    continue
                worker = copy(args)
                worker.seconds = max(0.2, (window.end - clock()).total_seconds())
                worker.expected_trading_day = window.trading_day
                worker.expected_source_day = calendar.previous_trading_day(window.trading_day)
                worker.max_session_orders = args.max_session_orders - used
                worker._continuous_window, worker._stop_event = window, stop
                worker.resume = persisted is not None
                worker._history_stage = "recovery" if persisted else "preopen"
                runtime = runtime_factory(worker)
                result = runtime.run()
                if result.get("status") != "passed":
                    raise RuntimeError("连续会话停机核对失败，不自动重启")
                last_status = None
                stop.wait(1)
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
