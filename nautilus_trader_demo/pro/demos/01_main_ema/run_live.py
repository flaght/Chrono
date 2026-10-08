"""当前主力 EMA 的 SimNow 单交易日入口；默认只记录目标。"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from dotenv import load_dotenv

from bomber.framework.dataprep.paths import resolve_paths
from bomber.framework.market.basic.base import Bar
from bomber.framework.market.stream import TradeTickBarFeed
from bomber.framework.market.stream.ctp import CtpLiveDataFeed, CtpMdConfig
from bomber.framework.market.stream.health import MarketHealthState
from bomber.framework.trader import (
    ControlledLiveExecutionClient, MarketReferencePriceStore, NautilusLiveExecutionBackend,
    PositionManager, PreTradeRiskManager, RecordingExecutionClient, RiskLimits, RuntimeMode)
from bomber.framework.trader.execution.ctp import (
    CtpExecutionAccounting, CtpLimitPlanner, CtpNativeTraderDriver, CtpPositionLedger,
    CtpTdApiTransport)
from bomber.framework.trader.execution.contracts import ExecutionReportType
from bomber.framework.trader.persistence import JsonStateStore, RuntimeStateManager

from .live_references import LiveMainReferences
from .live_runner import MainEmaSimnowRunner
from .strategy import MainEmaConfig, MainEmaStrategy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLIENT_ID = "main-ema-simnow"


def required(name):
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"缺少环境变量: {name}")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect", action="store_true")
    parser.add_argument("--mode", choices=("recording", "simnow"), default="recording")
    parser.add_argument("--product", required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--contract-struct", type=Path)
    parser.add_argument("--fut-basic", type=Path)
    parser.add_argument("--factors", type=Path)
    parser.add_argument("--factor-availability", choices=("aligned", "explicit"), default="aligned")
    parser.add_argument("--expected-trading-day", help="可选核对TD返回的YYYYMMDD交易日")
    parser.add_argument("--fast", type=int, default=3)
    parser.add_argument("--slow", type=int, default=5)
    parser.add_argument("--quantity", type=Decimal, default=Decimal(1))
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--query-timeout", type=float, default=15)
    parser.add_argument("--max-notional", type=Decimal, default=Decimal(50000))
    parser.add_argument("--limit-offset-ticks", type=int, default=1)
    parser.add_argument("--max-session-orders", type=int, default=4)
    parser.add_argument("--enable-orders", action="store_true")
    parser.add_argument("--confirm-simnow", action="store_true")
    parser.add_argument("--state-file", type=Path, help="报单模式必填；本次新检查点文件，不覆盖旧文件")
    parser.add_argument("--report-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args(argv)
    if not args.connect:
        parser.error("须显式 --connect 允许连接MD／TD")
    if not 0 < args.fast < args.slow:
        parser.error("EMA周期须满足0 < fast < slow")
    if not args.quantity.is_finite() or args.quantity <= 0 or args.quantity != args.quantity.to_integral_value():
        parser.error("手数须为正整数")
    if not args.max_notional.is_finite() or args.max_notional <= 0:
        parser.error("名义金额上限须为正且有限")
    if not 0 < args.seconds <= 3600 or not 0 < args.query_timeout <= 120:
        parser.error("运行时长须大于零且不超过3600秒，查询超时须大于零且不超过120秒")
    if args.limit_offset_ticks < 0 or not 1 <= args.max_session_orders <= 20:
        parser.error("限价偏移须非负，会话报单上限须为1至20笔")
    if args.mode == "simnow":
        if not args.enable_orders or not args.confirm_simnow or args.state_file is None:
            parser.error("simnow模式须同时提供--enable-orders --confirm-simnow --state-file")
        if args.state_file.exists():
            parser.error("状态文件已存在：保留旧检查点并先对账，本入口不自动恢复或覆盖")
    elif args.enable_orders or args.confirm_simnow or args.state_file:
        parser.error("recording模式不接受报单授权或交易状态文件")
    return args


def occupied_positions(transport):
    # 总仓查询识别净仓为零的双向仓，不能仅检查driver.reconcile的净值。
    return {str(key): value for key, value in transport.query_gross_positions().items()
            if any(value)}


def assert_flat_account(driver, transport):
    driver.reconcile()
    account = driver.reconcile_account_state()
    if driver.reconcile_active_orders().orders:
        raise RuntimeError("账户已有活动订单，拒绝自动接管")
    gross = occupied_positions(transport)
    if gross:
        raise RuntimeError(f"账户已有总仓，拒绝空仓启动: {gross}")
    if "CNY" not in account.balances or account.balances["CNY"].available <= 0:
        raise RuntimeError("权威查询未取得可用CNY资金")
    return account


def session_ready(driver, upstream, expected_day, *, require_orders=False, now_ns=None):
    received = upstream.latest_receive_monotonic_ns
    age = None if received is None else (time.monotonic_ns() if now_ns is None else now_ns) - received
    return bool(
        driver.trading_day == expected_day and upstream.latest_trading_day == expected_day
        and upstream.health_snapshot.state is MarketHealthState.READY
        and age is not None and 0 <= age <= 10_000_000_000
        and (not require_orders or driver.is_simnow_session))


class SessionMainRunner(MainEmaSimnowRunner):
    """把线程内异常送到主循环；真实Bar与MD／TD会话逐次校验。"""

    def __init__(self, *, references, driver, upstream, orders, **kwargs):
        super().__init__(RuntimeMode.LIVE, **kwargs)
        self.references = references
        self.driver = driver
        self.upstream = upstream
        self.orders = orders
        self.expected_day = references.trading_day.strftime("%Y%m%d")
        self.fixed_spec = references.spec
        minute = 60_000_000_000
        self.first_complete_bar_ns = ((references.started_ns + minute - 1) // minute + 1) * minute - 1
        self.failure = None
        self.accept_bars = False

    def publish(self, feed_id, event):
        if not isinstance(event, Bar) or not self.accept_bars:
            return
        if event.ts_event < self.first_complete_bar_ns:
            # 启动时只观察到部分分钟，不能将该分钟用于EMA或下单。
            return
        try:
            if not session_ready(self.driver, self.upstream, self.expected_day, require_orders=self.orders):
                raise RuntimeError("MD／TD会话失效、交易日不一致或行情停滞")
            age = time.time_ns() - event.ts_event
            if not 0 <= age <= 120_000_000_000:
                raise RuntimeError("分钟Bar与墙钟不匹配；首版不接受历史回放")
            self.references.refresh()
            if self.references.spec != self.fixed_spec:
                self._close_main_gate("主力合约或条款变更，固定路由闭闸并要求重新核对")
            return super().publish(feed_id, event)
        except Exception as error:
            self.failure = str(error)
            self.accept_bars = False
            if self.orders:
                self._clients[CLIENT_ID].disarm("bar_dispatch_failed")
            raise


def assemble(args, references, driver, upstream, *, positions=None):
    positions = PositionManager() if positions is None else positions
    prices = MarketReferencePriceStore()
    instrument = references.instrument_id
    expected_day = references.trading_day.strftime("%Y%m%d")
    ledger = CtpPositionLedger(expected_day)
    orders = args.mode == "simnow"
    feed = TradeTickBarFeed("MAIN_EMA_1M", upstream)
    feed.register_instrument(references.instrument_meta())
    if orders:
        backend = NautilusLiveExecutionBackend(CLIENT_ID, driver)
        client = ControlledLiveExecutionClient(
            CLIENT_ID, CtpLimitPlanner(ledger, positions, prices, references.spec.tick, args.limit_offset_ticks),
            backend, positions, PreTradeRiskManager(
                CLIENT_ID, positions, prices, instrument_limits={instrument: RiskLimits(
                    max_order_quantity=args.quantity, max_abs_position=args.quantity,
                    max_order_notional=args.max_notional, max_abs_position_notional=args.max_notional,
                    max_market_age_ns=120_000_000_000, contract_multiplier=references.spec.multiplier)}),
            account_id=driver.account_id,
            demo_environment_check=lambda: session_ready(driver, upstream, expected_day,
                                                         require_orders=True),
            max_request_wall_age_ns=120_000_000_000)
        backend.register_report_handler(CtpExecutionAccounting(
            ledger, {instrument: references.spec.multiplier}).on_report)
    else:
        client = RecordingExecutionClient(CLIENT_ID)
    strategy = MainEmaStrategy("main-ema-simnow", references, MainEmaConfig(
        args.product, references.spec.venue, args.fast, args.slow, args.quantity))
    runner = SessionMainRunner(references=references, driver=driver, upstream=upstream,
                               orders=orders, position_manager=positions)
    runner.add_market_observer(prices)
    runner.add_data_feed("ctp-bars", feed)
    runner.add_execution_client(client)
    runner.add_main_strategy(strategy, feed_id="ctp-bars", instrument_id=instrument, client_id=CLIENT_ID)
    manager = None
    if orders:
        manager = RuntimeStateManager(
            JsonStateStore(args.state_file), runner.target_store, runner.portfolio_coordinator,
            positions, order_machines={CLIENT_ID: client.order_state_machine},
            ctp_ledgers={CLIENT_ID: ledger}, ctp_drivers={CLIENT_ID: driver})
        manager.enable_ctp_autosave(CLIENT_ID, client)
    return SimpleNamespace(runner=runner, strategy=strategy, client=client, driver=driver,
                           upstream=upstream, ledger=ledger, manager=manager, references=references)


@contextmanager
def account_lock(account_id):
    # Linux同一账户的本入口不能并发运行；锁文件保留，避免unlink导致锁失效。
    import fcntl
    key = hashlib.sha256(account_id.encode()).hexdigest()[:16]
    with open(f"/tmp/bomber-main-ema-{key}.lock", "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("同一账户已有主力EMA进程运行") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_session(args):
    if required("CTP_BROKER_ID") != "9999":
        raise ValueError("本入口只允许SimNow BrokerID=9999")
    investor = required("CTP_ACCOUNT_ID")
    holder = {}
    lost = []

    def disconnected(reason):
        lost.append(reason)
        if "session" in holder and args.mode == "simnow":
            holder["session"].client.mark_disconnected(reason)

    transport = CtpTdApiTransport(
        client_id=CLIENT_ID, account_id=investor, front=required("CTP_TD_ADDRESS"),
        broker_id="9999", investor_id=investor, password=required("CTP_PASSWORD"),
        app_id=os.getenv("CTP_APP_ID", ""), auth_code=os.getenv("CTP_AUTH_CODE", ""),
        flow_path=os.getenv("CTP_TD_FLOW_PATH", "/tmp/bomber-main-ema-td"),
        production_mode=os.getenv("CTP_PRODUCTION_MODE", "true").lower() in {"1", "true", "yes", "on"},
        timeout_seconds=args.query_timeout)
    driver = CtpNativeTraderDriver(
        CLIENT_ID, investor, transport, enable_simnow_orders=args.mode == "simnow",
        max_session_orders=args.max_session_orders, disconnect_handler=disconnected)
    session = None
    status = "failed"
    final_gross = None
    final_active = None
    cleanup_errors = []
    report_path = args.report_dir / f"simnow-{time.time_ns()}"
    report_path.mkdir(parents=True, exist_ok=False)
    try:
        driver.start(lambda report: None)
        assert_flat_account(driver, transport)
        expected_day = driver.trading_day
        if args.expected_trading_day and args.expected_trading_day != expected_day:
            raise RuntimeError("TD交易日与指定交易日不一致")
        paths = resolve_paths(vars(args), project_root=PROJECT_ROOT,
                              required_files=("contract_struct", "fut_basic"))
        factors = args.factors or (paths.role / "fut_adjustment_factors.feather" if paths.role else None)
        if factors is None:
            raise ValueError("须配置角色数据目录或显式--factors")
        references = LiveMainReferences(
            product=args.product, trading_day=expected_day, contract_struct=paths.contract_struct,
            factors=factors, fut_basic=paths.fut_basic, started_ns=time.time_ns(),
            factor_availability=args.factor_availability)
        assignment = references.snapshot(time.time_ns())
        upstream = CtpLiveDataFeed(CtpMdConfig(
            front=required("CTP_MD_ADDRESS"), broker_id="9999", user_id=investor,
            password=required("CTP_PASSWORD"),
            flow_path=os.getenv("CTP_MD_FLOW_PATH", "/tmp/bomber-main-ema-md"),
            production_mode=transport.production_mode))
        # 持久化钩子必须在Driver启动前绑定；预检会话结束后才装配执行链。
        driver.stop()
        session = assemble(args, references, driver, upstream)
        holder["session"] = session
        # Backend.start重新登录并绑定正确回报接收器；预检期间从未授权报单。
        print(f"主力EMA: mode={args.mode} TD交易日={expected_day} source_day={assignment.source_day} "
              f"main={references.instrument_id} factor={assignment.factor(args.product.upper(), 'main')} "
              f"tick={references.spec.tick} multiplier={references.spec.multiplier}", flush=True)
        if args.mode == "recording":
            driver.start(lambda report: None)
        else:
            session.client.start()
        assert_flat_account(driver, transport)
        if driver.trading_day != expected_day or lost:
            raise RuntimeError("装配期间TD会话／交易日发生变化")
        session.runner.start()
        ready_deadline = time.monotonic() + 30
        while not session_ready(driver, upstream, expected_day, require_orders=args.mode == "simnow"):
            if lost or time.monotonic() >= ready_deadline:
                raise RuntimeError("30秒内未取得本交易日健康MD／TD会话")
            time.sleep(0.2)
        if args.mode == "simnow":
            session.manager.save()
            session.client.arm_demo(session.client.DEMO_CONFIRMATION)
        session.runner.accept_bars = True
        print("会话就绪，开始当前主力EMA实时分钟验证；停机撤单并报告剩余仓位", flush=True)
        deadline = time.monotonic() + args.seconds
        next_check = time.monotonic() + 10
        seen = 0
        cap_deadline = None
        while time.monotonic() < deadline:
            if session.runner.failure or lost:
                raise RuntimeError(session.runner.failure or lost[-1])
            if not session_ready(driver, upstream, expected_day, require_orders=args.mode == "simnow"):
                raise RuntimeError("MD／TD交易日变化、行情异常或停滞，停止本次运行")
            references.snapshot(time.time_ns())
            if references.spec != session.runner.fixed_spec:
                session.runner._close_main_gate("主力／条款文件变更，闭闸重新核对")
            if args.mode == "simnow":
                if session.client.report_errors:
                    raise RuntimeError("执行回报冲突，停止并要求对账")
                if driver.submitted_orders >= args.max_session_orders and cap_deadline is None:
                    session.runner.accept_bars = False
                    session.client.disarm("session_order_cap")
                    cap_deadline = time.monotonic() + 10
                if cap_deadline is not None and time.monotonic() >= cap_deadline:
                    break
                rows = session.client.backend.reports
            else:
                rows = session.client.requests
            for item in rows[seen:]:
                print(f"{'执行回报' if args.mode == 'simnow' else 'Recording目标'}: {item}", flush=True)
            seen = len(rows)
            if time.monotonic() >= next_check:
                if args.mode == "simnow":
                    session.client.refresh_account_state()
                else:
                    driver.reconcile_account_state()
                # 查询期间新成交会使快照与本地不同；不能在持回报锁时等待网络。
                # 首版保守处理：检测到差异即停止，停机再做完整查询，不用差异自动修正仓位。
                gross = occupied_positions(transport)
                snap = session.ledger.snapshot(references.instrument_id)
                expected = {} if not (snap.long_total or snap.short_total) else {
                    str(references.instrument_id): (snap.long_total, snap.short_total)}
                if gross != expected:
                    raise RuntimeError(f"柜台总仓与本地账本不一致: actual={gross} expected={expected}")
                next_check = time.monotonic() + 10
            time.sleep(0.2)
        if args.mode == "recording" and not session.client.requests:
            raise RuntimeError("未取得当前策略Recording目标；检查行情时段、资料和预热时长")
        if args.mode == "simnow" and not any(r.report_type in {
                ExecutionReportType.PARTIALLY_FILLED, ExecutionReportType.FILLED}
                for r in session.client.backend.reports):
            raise RuntimeError("本次没有成交回报，不能判定SimNow交易通过")
        status = "passed"
    finally:
        if session:
            session.runner.accept_bars = False
            if args.mode == "simnow":
                try:
                    session.client.set_risk_mode("HALTED", cancel_active_orders=True)
                    until = time.monotonic() + 10
                    while driver.reconcile_active_orders().orders:
                        if time.monotonic() >= until:
                            raise RuntimeError("停机仍有活动订单，需要人工核对")
                        time.sleep(1)
                except Exception as error:
                    cleanup_errors.append(str(error))
        try:
            final_active = len(driver.reconcile_active_orders().orders)
            final_gross = occupied_positions(transport)
            driver.reconcile_account_state()
            if session and args.mode == "simnow":
                session.manager.save()
                snap = session.ledger.snapshot(session.runner.references.instrument_id)
                expected = {} if not (snap.long_total or snap.short_total) else {
                    str(session.runner.references.instrument_id): (snap.long_total, snap.short_total)}
                if final_gross != expected:
                    raise RuntimeError("停机柜台总仓与本地账本不一致")
        except Exception as error:
            cleanup_errors.append(str(error))
        try:
            if session:
                session.runner.stop()
        finally:
            driver.stop()
            if cleanup_errors:
                status = "failed"
            result = {
                "status": status, "mode": args.mode, "product": args.product,
                "bars_used": session.strategy.bars_used if session else 0,
                "orders_submitted": driver.submitted_orders, "final_gross": final_gross,
                "final_active_orders": final_active, "cleanup_errors": cleanup_errors,
                "state_file": str(args.state_file) if args.state_file else None,
                "trading_day": session.runner.expected_day if session else None,
                "reference_files": [str(p) for p in session.references.paths] if session else [],
                "fixed_instrument": str(session.runner._main_binding[2]) if session else None,
                "factor_availability": args.factor_availability,
                "ema": {"fast": args.fast, "slow": args.slow, "quantity": str(args.quantity)},
            }
            (report_path / "summary.json").write_text(json.dumps(result, ensure_ascii=False,
                indent=2, default=str) + "\n", encoding="utf-8")
            if session:
                rows = session.client.backend.reports if args.mode == "simnow" else session.client.requests
                (report_path / "events.txt").write_text("\n".join(map(str, rows)) + "\n", encoding="utf-8")
            print(f"停机结果: {result}\n记录目录: {report_path}", flush=True)
        if cleanup_errors:
            raise RuntimeError("停机查询／撤单未完成: " + "; ".join(cleanup_errors))


def main(argv=None):
    args = parse_args(argv)
    load_dotenv(PROJECT_ROOT / ".env")
    import bomber
    import bomber.framework
    print(f"Python模式=module Bomber={getattr(bomber, '__version__', 'unknown')} "
          f"framework={bomber.framework.__file__} 策略={Path(__file__).with_name('strategy.py')}", flush=True)
    with account_lock(required("CTP_ACCOUNT_ID")):
        run_session(args)


if __name__ == "__main__":
    main()
