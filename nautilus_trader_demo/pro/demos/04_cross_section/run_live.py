"""多品种横截面动量SimNow实时入口；默认Recording，SHFE/INE固定主力会话。"""

from __future__ import annotations

import argparse
from datetime import datetime
from decimal import Decimal
import math
from pathlib import Path
import time
from types import SimpleNamespace

from dotenv import load_dotenv
from bomber.framework.trader.runtime.live.channels.ctp import required, build_simnow_transport, build_md_config
from bomber.framework.trader.runtime.live.channels.ctp import add_environment_arguments, validate_environment_arguments

from bomber.framework.dataprep.live_references import LiveFuturesReferences, live_factor_policy
from bomber.framework.dataprep.reference_freshness import ReferenceFreshnessPolicy
from bomber.framework.dataprep.sources import DataSourcePurpose, DolphinDbReferenceConfig, ReferenceSourceFactory
from bomber.framework.market.basic.base import InstrumentId
from bomber.framework.market.stream.ctp import CtpLiveDataFeed, CtpMdConfig
from bomber.framework.trader.execution.ctp import CtpTdApiTransport
from bomber.framework.trader.runtime.live.runtime import ManagedLiveRuntime
from bomber.framework.trader.runtime.reports import LiveRunReport

from .live_runtime import CLIENT_ID, CrossSectionLifecycle, PortfolioReferences, assemble

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect", action="store_true")
    parser.add_argument("--mode", choices=("recording", "simnow"), default="recording")
    add_environment_arguments(parser)
    parser.add_argument("--products", required=True, help="至少两个不同品种，逗号分隔；首轮RB,HC")
    parser.add_argument("--expected-instrument", action="append", default=[],
                        help="可选手工核对PRODUCT=symbol.VENUE；省略则自动读取当前主力，传入时须覆盖全部品种")
    parser.add_argument("--expected-trading-day", required=True)
    parser.add_argument("--expected-source-day", required=True)
    parser.add_argument("--reference-database")
    parser.add_argument("--reference-timeout", type=int, default=15)
    parser.add_argument("--reference-refresh-seconds", type=float, default=5)
    parser.add_argument("--reference-max-source-age-days", type=int, default=14)
    parser.add_argument("--reference-max-observation-seconds", type=int, default=30)
    parser.add_argument("--reference-max-refresh-duration-seconds", type=int, default=30)
    parser.add_argument("--factor-date-basis", choices=("source", "trading"), default="source")
    parser.add_argument("--factor-availability", choices=("source-day-end", "aligned", "explicit", "observed-on-read"),
                        default="observed-on-read")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--rebalance-interval", type=int, default=5)
    parser.add_argument("--group-fraction", type=Decimal, default=Decimal("0.30"))
    parser.add_argument("--target-notional", type=Decimal, default=Decimal(100000), help="每侧总名义金额预算")
    parser.add_argument("--target-quantity-cap", type=Decimal,
                        help="可选：预算取整后每合约目标手数上限；不会把不足一手提升为一手")
    parser.add_argument("--require-full-groups", action="store_true",
                        help="所有多空入选品种都能分配非零手数才建仓，否则全组合归零")
    parser.add_argument("--max-quantity", type=Decimal, default=Decimal(1), help="每合约订单及持仓手数上限，超限拒绝而非裁剪")
    parser.add_argument("--max-notional", type=Decimal, default=Decimal(100000), help="每合约订单及持仓名义金额上限")
    parser.add_argument("--limit-offset-ticks", type=int, default=1)
    parser.add_argument("--max-session-orders", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=1800)
    parser.add_argument("--query-timeout", type=float, default=15)
    parser.add_argument("--enable-orders", action="store_true")
    parser.add_argument("--confirm-simnow", action="store_true")
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--resume", action="store_true", help="恢复原状态文件，同交易日且无活动订单；实时重新预热")
    parser.add_argument("--report-dir", type=Path, default=Path(__file__).with_name("results"))
    args = parser.parse_args(argv)
    if not args.connect:
        parser.error("须显式--connect；默认不连接柜台")
    args.products = tuple(item.strip().upper() for item in args.products.split(","))
    if len(args.products) < 2 or len(set(args.products)) != len(args.products) or any(not p.isalpha() for p in args.products):
        parser.error("products须为至少两个不重复字母品种")
    try:
        expected = {}
        for value in args.expected_instrument:
            product, instrument = value.split("=", 1)
            product = product.strip().upper()
            parsed = InstrumentId.from_str(instrument.strip())
            if product in expected or str(parsed.venue) not in {"SHFE", "INE"}:
                raise ValueError("重复品种或不支持的交易所")
            expected[product] = str(parsed)
        if expected and set(expected) != set(args.products):
            raise ValueError("预期合约须完整覆盖品种")
        args.expected_instruments = expected
        for value in (args.expected_trading_day, args.expected_source_day):
            if datetime.strptime(value, "%Y%m%d").strftime("%Y%m%d") != value:
                raise ValueError("日期须为YYYYMMDD")
        validate_environment_arguments(args)
        args.factor_availability = live_factor_policy(args.factor_date_basis, args.factor_availability)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    if args.lookback < 1 or args.rebalance_interval < 1 or args.limit_offset_ticks < 0 or args.max_session_orders < 1:
        parser.error("窗口/间隔/订单上限须为正，跳价偏移须非负")
    if not args.group_fraction.is_finite() or not 0 < args.group_fraction <= Decimal("0.5"):
        parser.error("group-fraction须在(0,0.5]")
    if any(not v.is_finite() or v <= 0 for v in (args.target_notional, args.max_notional, args.max_quantity)):
        parser.error("预算/名义限额/数量限额须为有限正数")
    if args.max_quantity != args.max_quantity.to_integral_value():
        parser.error("max-quantity须为整数手数")
    if args.target_quantity_cap is not None and (
            not args.target_quantity_cap.is_finite() or args.target_quantity_cap <= 0 or
            args.target_quantity_cap != args.target_quantity_cap.to_integral_value() or
            args.target_quantity_cap > args.max_quantity):
        parser.error("target-quantity-cap须为不超过max-quantity的正整数")
    if any(not math.isfinite(v) or v <= 0 for v in (args.seconds, args.query_timeout)):
        parser.error("运行时长和查询超时须为有限正数")
    if (not math.isfinite(args.reference_refresh_seconds) or not 0 <= args.reference_refresh_seconds <= 60
            or args.reference_timeout <= 0):
        parser.error("参考刷新须在[0,60]秒，读取超时须为正")
    if any(v <= 0 for v in (args.reference_max_source_age_days,
                           args.reference_max_observation_seconds,
                           args.reference_max_refresh_duration_seconds)):
        parser.error("参考来源日/观测/刷新时效限额须为正")
    if args.mode == "simnow":
        if not args.enable_orders or not args.confirm_simnow or args.state_file is None:
            parser.error("SimNow报单须enable-orders、confirm-simnow和独立state-file")
        if args.resume != args.state_file.is_file():
            parser.error("恢复须--resume及已有原state-file；新会话须使用不存在的文件")
        if args.state_file.exists() and not args.state_file.is_file():
            parser.error("state-file不是普通文件")
    elif args.enable_orders or args.confirm_simnow or args.state_file or args.resume:
        parser.error("Recording不接受报单开关或交易状态文件")
    return args


def prepare_inputs(args, context, lifecycle):
    source = ReferenceSourceFactory.create_for(DataSourcePurpose.LIVE_REFERENCE, "dolphindb",
        DolphinDbReferenceConfig.from_env(database=args.reference_database,
            read_timeout_seconds=args.reference_timeout))
    context.resources.callback(source.close)
    source.open()
    references = PortfolioReferences(LiveFuturesReferences(source, products=args.products,
        trading_day=context.trading_day, started_ns=time.time_ns(),
        factor_date_basis=args.factor_date_basis, factor_availability=args.factor_availability,
        refresh_seconds=args.reference_refresh_seconds,
        freshness=ReferenceFreshnessPolicy(
            expected_source_day=datetime.strptime(args.expected_source_day, "%Y%m%d").date(),
            max_source_age_days=args.reference_max_source_age_days,
            max_observation_age_seconds=args.reference_max_observation_seconds,
            max_refresh_duration_seconds=args.reference_max_refresh_duration_seconds)))
    context.references = references
    actual = {p: str(i) for p, i in references.instrument_ids.items()}
    if args.expected_instruments and actual != args.expected_instruments:
        raise ValueError(f"本次主力不符: expected={args.expected_instruments} actual={actual}")
    assignment = references.snapshot(time.time_ns())
    for p, spec in references.specs.items():
        print(f"横截面参考: product={p} instrument={actual[p]} TD={context.trading_day} "
              f"source_day={assignment.source_day} factor={assignment.factor(p, 'main')} "
              f"tick={spec.tick} multiplier={spec.multiplier}", flush=True)
    upstream = CtpLiveDataFeed(build_md_config(CtpMdConfig, lifecycle,
        flow_path="/tmp/bomber-cross-section-md"))
    return SimpleNamespace(references=references, upstream=upstream)


def describe(session):
    if session is None:
        return {}
    s = session.strategy
    positions = session.runner.position_manager
    return {"synchronized_frames": s.synchronized_frames, "rebalances": s.rebalances,
        "last_frame_ns": s.last_emitted_ns,
        "last_targets": dict(s.last_targets) if s.last_targets is not None else None,
        "scores": dict(s.last_signal.scores) if s.last_signal else None,
        "long_products": s.last_signal.long_products if s.last_signal else (),
        "short_products": s.last_signal.short_products if s.last_signal else (),
        "order_updates_received": s.order_updates_received, "fills_received": s.fills_received,
        "last_order_status": s.last_order_update.status.value if s.last_order_update else None,
        "fill_positions": s.fill_positions,
        "strategy_positions": {str(i): positions.position(s.strategy_id, str(i)) for i in session.runner.fixed_ids},
        "account_positions": {str(i): positions.account_position(CLIENT_ID, i) for i in session.runner.fixed_ids},
        "unassigned_positions": {str(i): positions.unassigned_position(CLIENT_ID, i) for i in session.runner.fixed_ids}}


def build_runtime(args):
    transport = build_simnow_transport(CtpTdApiTransport, client_id=CLIENT_ID,
        flow_path="/tmp/bomber-cross-section-td", timeout_seconds=args.query_timeout)
    lifecycle = CrossSectionLifecycle(transport, md_front=required("CTP_MD_ADDRESS"),
        orders=args.mode == "simnow", max_session_orders=args.max_session_orders,
        environment=args.simnow_environment, replay_md_trading_day=args.replay_md_trading_day,
        resume=args.resume,
        scope_to_reference_instruments=True,
        expected_trading_day=args.expected_trading_day, legacy_lock_namespaces=("bomber-main-ema",))
    report = LiveRunReport(args.report_dir, prefix="cross-section-simnow", describe=describe, metadata={
        "mode": args.mode, "strategy_id": CLIENT_ID, "products": args.products,
        "simnow_environment": args.simnow_environment,
        "replay_md_trading_day": args.replay_md_trading_day,
        "expected_instruments": args.expected_instruments, "reference_source": "dolphindb",
        "instrument_selection": "assert_expected" if args.expected_instruments else "automatic_main",
        "lookback": args.lookback, "rebalance_interval": args.rebalance_interval,
        "group_fraction": args.group_fraction, "target_notional_per_side": args.target_notional,
        "max_quantity": args.max_quantity, "max_notional": args.max_notional,
        "max_session_orders": args.max_session_orders,
        "factor_date_basis": args.factor_date_basis, "factor_availability": args.factor_availability,
        "expected_source_day": args.expected_source_day,
        "state_file": str(args.state_file) if args.state_file else None,
        "initial_position_mode": "resume" if args.resume else "fresh_flat", "history_minutes": 0,
        "instrument_max_age_seconds": args.instrument_max_age_seconds,
        "target_quantity_cap": args.target_quantity_cap,
        "require_full_groups": args.require_full_groups})
    return ManagedLiveRuntime(CLIENT_ID, controller=lifecycle,
        prepare_inputs=lambda context: prepare_inputs(args, context, lifecycle),
        assemble_session=lambda inputs, context: assemble(args, inputs.references, lifecycle.driver, inputs.upstream),
        report=report, seconds=args.seconds)


def main(argv=None):
    args = parse_args(argv)
    load_dotenv(PROJECT_ROOT / ".env")
    import bomber
    print(f"Bomber={getattr(bomber, '__version__', 'unknown')} source={Path(__file__).resolve()}", flush=True)
    runtime = build_runtime(args)
    try:
        runtime.run()
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
