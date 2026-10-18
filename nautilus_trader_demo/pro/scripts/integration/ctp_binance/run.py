"""GAP-10固定双端目标联调：默认只读；显式双端授权才报单。"""

import argparse
from contextlib import ExitStack
from datetime import datetime
from decimal import Decimal
from importlib import import_module
import os
from pathlib import Path
import time
from types import SimpleNamespace

from bomber.framework.market.basic.base import DataType, InstrumentId
from bomber.framework.market.stream.bn import BNWSConfig, BNWSStreamDataFeed
from bomber.framework.market.stream.health import MarketHealthState
from bomber.framework.trader.assembly import StrategyBindings
from bomber.framework.trader.contracts import DataBinding, ExecutionRoute
from bomber.framework.trader.execution.builders import build_ctp_execution
from bomber.framework.trader.execution.ctp import CtpNativeTraderDriver, CtpTdApiTransport
from bomber.framework.trader.execution.live.binance_joint import build_binance_joint_execution
from bomber.framework.trader.execution.risk import MarketReferencePriceStore, RiskLimits
from bomber.framework.trader.persistence import JsonStateStore, RuntimeStateManager
from bomber.framework.trader.portfolio import PositionManager
from bomber.framework.trader.runtime.ctp import account_lock, occupied_positions, session_ready
from bomber.framework.trader.runtime.joint import JointExecutionBinding, JointLiveRunner, assemble_joint_strategy
from bomber.framework.trader.runtime.managed import LiveRunContext
from bomber.framework.trader.runtime.ownership import AccountWriterIdentity
from bomber.framework.trader.runtime.reports import LiveRunReport
from bomber.framework.trader.template import StrategyTemplate

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class JointTargetProbe(StrategyTemplate):
    """显式目标验收策略；行情只更新参考价，控制线程就绪后提交一次双端目标。"""
    def __init__(self):
        super().__init__("joint-target-probe")
        self.orders, self.fills = [], []

    def on_order(self, event):
        self.orders.append({"client_id": event.identity.client_id, "order_id": event.identity.client_order_id,
            "status": event.status.value, "target_key": event.metadata.get("attributed_target_key"),
            "position": str(self.position(event.metadata["attributed_target_key"]))})

    def on_fill(self, event):
        key = event.metadata["attributed_target_key"]
        self.fills.append({"client_id": event.identity.client_id, "order_id": event.identity.client_order_id,
            "event_id": event.event_id, "quantity": str(event.quantity), "target_key": key,
            "position": str(self.position(key)), "account_position": str(self.account_position(key))})


def required(name):
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"缺少环境变量: {name}")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect", action="store_true")
    parser.add_argument("--mode", choices=("readonly", "orders"), default="readonly")
    parser.add_argument("--product", required=True)
    parser.add_argument("--expected-trading-day", required=True)
    parser.add_argument("--expected-source-day", required=True)
    parser.add_argument("--reference-source", choices=("file", "dolphindb"), default="dolphindb")
    parser.add_argument("--allow-file-reference-test", action="store_true")
    parser.add_argument("--reference-max-source-age-days", type=int, default=14)
    parser.add_argument("--reference-max-observation-seconds", type=int, default=30)
    parser.add_argument("--reference-max-refresh-duration-seconds", type=int, default=30)
    parser.add_argument("--reference-database")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--contract-struct", type=Path)
    parser.add_argument("--fut-basic", type=Path)
    parser.add_argument("--factors", type=Path)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--ctp-target", type=Decimal, required=True)
    parser.add_argument("--bn-target", type=Decimal, required=True)
    parser.add_argument("--ctp-max-notional", type=Decimal, required=True)
    parser.add_argument("--bn-max-notional", type=Decimal, required=True)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--enable-orders", action="store_true")
    parser.add_argument("--confirm-simnow", action="store_true")
    parser.add_argument("--confirm-demo", action="store_true")
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "tests/results")
    args = parser.parse_args(argv)
    if args.reference_source == "file" and not args.allow_file_reference_test:
        parser.error("文件资料测试须--allow-file-reference-test；正式在线使用DolphinDB")
    if args.allow_file_reference_test and args.reference_source != "file":
        parser.error("--allow-file-reference-test仅用于file")
    if args.reference_source == "file" and args.reference_database:
        parser.error("--reference-database仅用于dolphindb")
    from bomber.framework.dataprep.reference_freshness import ReferenceFreshnessPolicy
    try:
        ReferenceFreshnessPolicy(max_source_age_days=args.reference_max_source_age_days,
            max_observation_age_seconds=args.reference_max_observation_seconds,
            max_refresh_duration_seconds=args.reference_max_refresh_duration_seconds)
    except ValueError as error:
        parser.error(str(error))
    if not args.connect or not 0 < args.seconds <= 600:
        parser.error("须显式--connect；时长须为0至600秒")
    for value in (args.expected_trading_day, args.expected_source_day):
        try:
            if len(value) != 8 or datetime.strptime(value, "%Y%m%d").strftime("%Y%m%d") != value:
                raise ValueError()
        except ValueError:
            parser.error("预期交易日及来源日须为YYYYMMDD")
    if (not args.ctp_target.is_finite() or abs(args.ctp_target) != 1
            or not args.bn_target.is_finite() or not 0 < abs(args.bn_target) <= Decimal("0.001")):
        parser.error("首期双端非零目标：CTP仅±1手；BN绝对值不超过0.001")
    if any(not value.is_finite() or value <= 0 for value in (args.ctp_max_notional, args.bn_max_notional)):
        parser.error("两端名义额上限须为正且有限")
    args.symbol = args.symbol.upper()
    if not args.symbol.isalnum() or not args.symbol.endswith("USDT"):
        parser.error("首期BN仅支持USDT永续")
    if args.mode == "orders":
        if not (args.enable_orders and args.confirm_simnow and args.confirm_demo and args.state_file):
            parser.error("报单须双端显式确认及新状态文件")
        if args.state_file.exists():
            parser.error("检查点已存在：先对账；联合入口不自动恢复旧进程或覆盖文件")
    elif args.enable_orders or args.confirm_simnow or args.confirm_demo or args.state_file:
        parser.error("只读模式不接受报单授权或状态文件")
    return args


def run(args):
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    if required("CTP_BROKER_ID") != "9999":
        raise ValueError("联合首期仅允许CTP SimNow BrokerID 9999")
    investor = required("CTP_ACCOUNT_ID")
    # 真实物理账户的稳定部署别名。同账户的全部API key必须配置同一值。
    bn_owner = required("BINANCE_DEMO_ACCOUNT_OWNER_ID")
    api_key, api_secret = required("BINANCE_DEMO_API_KEY"), required("BINANCE_DEMO_API_SECRET")
    report = LiveRunReport(args.report_dir, prefix="ctp-binance-joint", metadata={
        "mode": args.mode, "ownership_scope": "same_host_cooperating_processes",
        "policy": "DIRECT_HALT_ALL", "proposed_targets": {"ctp": str(args.ctp_target), "bn": str(args.bn_target)}})
    report.begin()
    runner = None
    manager = None
    strategy = JointTargetProbe()
    status, fields, errors = "failed", {}, []
    with ExitStack() as resources:
        legacy_leases = ExitStack()
        resources.callback(lambda: legacy_leases.close() if runner is None or not runner._lease_held else None)
        # 与既有CTP入口共用旧锁，避免新旧入口绕过彼此；新物理账户锁由Runner持有。
        legacy_leases.enter_context(account_lock(investor))
        legacy_leases.enter_context(account_lock(investor, namespace="bomber-main-ema"))
        holder = {}
        transport = CtpTdApiTransport(client_id="joint-ctp-simnow", account_id=investor,
            front=required("CTP_TD_ADDRESS"), broker_id="9999", investor_id=investor,
            password=required("CTP_PASSWORD"), app_id=os.getenv("CTP_APP_ID", ""),
            auth_code=os.getenv("CTP_AUTH_CODE", ""),
            flow_path=os.getenv("CTP_TD_FLOW_PATH", "/tmp/bomber-joint-td"),
            production_mode=True, timeout_seconds=15)
        driver = CtpNativeTraderDriver(transport.client_id, investor, transport,
            enable_simnow_orders=args.mode == "orders", max_session_orders=1,
            disconnect_handler=lambda reason: holder["runner"].trip(f"ctp_disconnect:{reason}") if holder else None)
        resources.callback(driver.stop)
        try:
            driver.start(lambda report: None)
            if driver.trading_day != args.expected_trading_day or occupied_positions(transport):
                raise RuntimeError("本次CTP交易日不符或存在总仓；联合入口只接收空仓新会话")
            if driver.reconcile_active_orders().orders:
                raise RuntimeError("CTP已有活动订单，禁止联合启动")
            input_args = SimpleNamespace(**vars(args))
            input_args.factor_date_basis = "source" if args.reference_source == "dolphindb" else "trading"
            input_args.factor_availability = "observed-on-read" if args.reference_source == "dolphindb" else "aligned"
            input_args.reference_timeout, input_args.reference_refresh_seconds = 15, 5
            inputs = import_module("demos.01_main_ema.run_live").prepare_inputs(input_args,
                LiveRunContext(driver.trading_day, resources), SimpleNamespace(
                    transport=transport, md_front=required("CTP_MD_ADDRESS")))
            references, upstream = inputs.references, inputs.upstream
            fixed_spec, fixed_id = references.spec, references.instrument_id
            upstream.register_instrument(references.instrument_meta())
            driver.stop()
            positions, prices = PositionManager(), MarketReferencePriceStore()

            def ctp_ready():
                references.snapshot(time.time_ns())
                return references.spec == fixed_spec and references.instrument_id == fixed_id and session_ready(
                    driver, upstream, args.expected_trading_day, require_orders=args.mode == "orders")

            ctp = build_ctp_execution(driver, instrument_id=fixed_id, trading_day=args.expected_trading_day,
                price_increment=fixed_spec.tick, multiplier=fixed_spec.multiplier,
                risk_limits=RiskLimits(max_order_quantity=1, max_abs_position=1,
                    max_order_notional=args.ctp_max_notional, max_abs_position_notional=args.ctp_max_notional,
                    max_market_age_ns=10_000_000_000, contract_multiplier=fixed_spec.multiplier),
                session_check=ctp_ready, orders=True, positions=positions, prices=prices)
            bn_id = InstrumentId.from_str(f"{args.symbol}-PERP.BINANCE")
            bn = build_binance_joint_execution(instrument_id=bn_id, account_id=bn_owner,
                positions=positions, prices=prices, api_key=api_key, api_secret=api_secret,
                risk_limits=RiskLimits(max_order_quantity=abs(args.bn_target), max_abs_position=abs(args.bn_target),
                    max_order_notional=args.bn_max_notional, max_abs_position_notional=args.bn_max_notional,
                    max_market_age_ns=10_000_000_000))
            resources.callback(bn.driver.stop)
            bn_feed = BNWSStreamDataFeed(BNWSConfig(ws_base_url="wss://demo-fstream.binance.com", market_type="futures"))
            runner = JointLiveRunner(position_manager=positions, max_orders_per_client=1, bindings=(
                JointExecutionBinding(ctp, AccountWriterIdentity("CTP", "simnow", f"9999:{investor}"), ctp_ready),
                JointExecutionBinding(bn, AccountWriterIdentity("BINANCE", "demo", bn_owner),
                    lambda: bn.driver.node.is_running() and bn_feed.health_snapshot.state is MarketHealthState.READY)))
            holder["runner"] = runner
            runner._resources.callback(legacy_leases.close)
            assemble_joint_strategy(runner, strategy, feeds={"ctp": upstream, "bn": bn_feed},
                bindings=StrategyBindings(data=(DataBinding("ctp", "ctp", fixed_id, DataType.TRADE_TICK),
                    DataBinding("bn", "bn", bn_id, DataType.TRADE_TICK)), execution=(
                    ExecutionRoute("ctp", ctp.client.client_id, fixed_id), ExecutionRoute("bn", bn.client.client_id, bn_id))))
            if args.mode == "orders":
                manager = RuntimeStateManager(JsonStateStore(args.state_file), runner.target_store,
                    runner.portfolio_coordinator, positions, order_machines={
                        ctp.client.client_id: ctp.client.order_state_machine, bn.client.client_id: bn.client.order_state_machine},
                    ctp_ledgers={ctp.client.client_id: ctp.ledger}, ctp_drivers={ctp.client.client_id: driver})
                manager.enable_ctp_autosave(ctp.client.client_id, ctp.client)
            runner.start()
            if occupied_positions(transport) or any(positions.snapshot().account_positions.values()):
                raise RuntimeError("双端新会话必须全账户空仓；不接受接管")
            if args.mode == "orders":
                native = bn.driver.node.cache.instrument(bn_id)
                if Decimal(str(native.make_qty(abs(args.bn_target)))) != abs(args.bn_target):
                    raise RuntimeError("BN目标不满足合约数量精度，拒绝自动舍入")
                deadline = time.monotonic() + 30
                while not all(runner.readiness(require_armed=False).values()):
                    if runner.failure or time.monotonic() >= deadline:
                        raise RuntimeError("双端在30秒内未就绪，未授权报单")
                    time.sleep(0.2)
                runner.refresh_authority()
                manager.save()
                runner.arm("AUTHORIZE_JOINT_DEMO_ORDERS")
                strategy.set_targets({"ctp": args.ctp_target, "bn": args.bn_target}, time.time_ns())
                manager.save()
                deadline = time.monotonic() + args.seconds
                while time.monotonic() < deadline:
                    if runner.failure or not all(runner.readiness().values()):
                        raise RuntimeError(runner.failure or "必需端不再就绪")
                    if (strategy.position("ctp") == args.ctp_target and strategy.position("bn") == args.bn_target
                            and not any(positions.snapshot().working_quantities.values())):
                        break
                    # 只刷新资金心跳；不重发目标。完整对账留到撤单确认后。
                    for item in runner.bindings:
                        item.execution.client.refresh_account_state()
                    manager.save()
                    time.sleep(1)
                runner.cancel_and_confirm()
                manager.save()
                if strategy.position("ctp") != args.ctp_target or strategy.position("bn") != args.bn_target:
                    raise RuntimeError("双端目标未全部成交，不能标记联合交易通过")
                if {row["client_id"] for row in strategy.fills} != {ctp.client.client_id, bn.client.client_id}:
                    raise RuntimeError("缺少某一端实际策略成交回调")
            status = "passed"
            fields["authority"] = {item.execution.client.client_id: {
                "account_revision": item.execution.client.account_state.revision,
                "active_orders_revision": runner._orders_revisions[item.execution.client.client_id]}
                for item in runner.bindings}
            fields["account_positions"] = {f"{key.client_id}:{key.instrument_id}": str(qty)
                for key, qty in positions.snapshot().account_positions.items()}
        except BaseException as error:
            fields["failure"] = str(error)
            if runner:
                runner.trip(type(error).__name__)
            raise
        finally:
            if runner:
                if args.mode == "orders" and runner._started:
                    try:
                        runner.cancel_and_confirm()
                    except Exception as error:
                        errors.append(str(error))
                fields["positions"] = {item.execution.client.client_id: {
                    "strategy": str(positions.position(strategy.strategy_id, key)),
                    "account": str(positions.account_position(item.execution.client.client_id, instrument)),
                    "unassigned": str(positions.unassigned_position(item.execution.client.client_id, instrument)),
                    "working": str(positions.working_quantity(item.execution.client.client_id, instrument))}
                    for item, key, instrument in ((runner.bindings[0], "ctp", fixed_id),
                        (runner.bindings[1], "bn", bn_id))}
                fields["order_attempts"] = runner._order_attempts
                if runner._started:
                    try:
                        fields["final_ctp_gross"] = occupied_positions(transport)
                        fields["final_active_orders"] = {item.execution.client.client_id:
                            len(item.execution.client.backend.reconcile_active_orders().orders)
                            for item in runner.bindings}
                    except Exception as error:
                        errors.append(str(error))
                if manager is not None and runner._started:
                    try:
                        manager.save()
                    except Exception as error:
                        errors.append(str(error))
                fields.update(audit=runner.audit, order_callbacks=strategy.orders, fill_callbacks=strategy.fills)
                try:
                    runner.stop()
                except Exception as error:
                    errors.append(str(error))
            fields["cleanup_errors"] = errors
            report.write(status="failed" if errors else status, fields=fields, session=runner,
                events=tuple(strategy.orders) + tuple(strategy.fills))
            if errors:
                raise RuntimeError("联合清理未完成: " + "; ".join(errors))
    return report.path


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
