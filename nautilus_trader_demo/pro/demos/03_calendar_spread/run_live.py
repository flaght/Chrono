"""单品种两期限价差Live；默认仅MD连接与Recording，显式选择Model/SimNow。"""

from __future__ import annotations

import argparse
from datetime import datetime
from decimal import Decimal
import math
from pathlib import Path
import time
from types import SimpleNamespace

from dotenv import load_dotenv
from bomber.framework.dataprep.live_references import LiveFuturesReferences
from bomber.framework.dataprep.reference_freshness import ReferenceFreshnessPolicy
from bomber.framework.dataprep.sources import DataSourcePurpose, DolphinDbReferenceConfig, ReferenceSourceFactory
from bomber.framework.market.basic.base import InstrumentId
from bomber.framework.trader.execution.policy import (
    add_execution_policy_arguments, execution_policy_from_args, observed_account_positions)
from bomber.framework.trader.runtime.live.channels.ctp import (
    add_environment_arguments, validate_environment_arguments, build_simnow_transport, build_md_config, required)
from bomber.framework.trader.runtime.live.runtime import ManagedLiveRuntime
from bomber.framework.trader.runtime.reports import LiveRunReport
from .live_references import CalendarReferences
from .live_runtime import CLIENT_ID, assemble
from .recovery import CalendarLocalLifecycle, CalendarCtpLifecycle
from .strategy import CalendarSpreadConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect", action="store_true")
    parser.add_argument("--mode", choices=("recording", "model", "simnow"), default="recording")
    add_execution_policy_arguments(parser)
    add_environment_arguments(parser)
    parser.add_argument("--product", required=True)
    parser.add_argument("--leg1-role", "--near-role", dest="leg1_role", choices=("main", "secondary"), default="secondary")
    parser.add_argument("--leg2-role", "--far-role", dest="leg2_role", choices=("secondary", "far"), default="far")
    parser.add_argument("--expected-instrument", action="append", default=[], help="ROLE=symbol.VENUE；可省略，指定时须覆盖双腿")
    parser.add_argument("--expected-trading-day", required=True, help="参考交易日YYYYMMDD；SimNow须与TD一致")
    parser.add_argument("--expected-source-day", required=True, help="严格早于参考交易日的角色来源日YYYYMMDD")
    parser.add_argument("--reference-database")
    parser.add_argument("--reference-timeout", type=int, default=15)
    parser.add_argument("--reference-refresh-seconds", type=float, default=5)
    parser.add_argument("--reference-max-source-age-days", type=int, default=14)
    parser.add_argument("--reference-max-observation-seconds", type=int, default=30)
    parser.add_argument("--reference-max-refresh-duration-seconds", type=int, default=30)
    parser.add_argument("--quantity", type=Decimal, default=Decimal(1))
    parser.add_argument("--lookback", type=int, default=120)
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument("--rebalance-interval", type=int, default=5)
    parser.add_argument("--max-notional", type=Decimal, default=Decimal(1000000), help="每腿订单和持仓名义金额限额")
    parser.add_argument("--limit-offset-ticks", type=int, default=1)
    parser.add_argument("--max-session-orders", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=9000)
    parser.add_argument("--query-timeout", type=float, default=15)
    parser.add_argument("--enable-orders", action="store_true")
    parser.add_argument("--confirm-simnow", action="store_true")
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--resume", action="store_true", help="原文件同交易日恢复，保留归属成交并重新预热")
    parser.add_argument("--require-fills", action="store_true", help="额外要求本次取得归属成交")
    parser.add_argument("--report-dir", type=Path, default=Path(__file__).with_name("results"))
    args = parser.parse_args(argv)
    if not args.connect:
        parser.error("须显式--connect；构造和help不连接")
    args.product = args.product.strip().upper()
    try:
        if not args.product.isascii() or not args.product.isalpha():
            raise ValueError("product须为英文字母品种")
        CalendarSpreadConfig(args.leg1_role, args.leg2_role, args.quantity, args.lookback,
            args.entry_z, args.exit_z, args.rebalance_interval)
        days = []
        for value in (args.expected_trading_day, args.expected_source_day):
            day = datetime.strptime(value, "%Y%m%d")
            if day.strftime("%Y%m%d") != value:
                raise ValueError("日期须为YYYYMMDD")
            days.append(day)
        if days[1] >= days[0]:
            raise ValueError("角色来源日须严格早于参考交易日")
        validate_environment_arguments(args)
        expected = {}
        for value in args.expected_instrument:
            role, instrument = value.split("=", 1)
            role, instrument = role.strip(), instrument.strip()
            parsed = InstrumentId.from_str(instrument)
            if role in expected or role not in (args.leg1_role, args.leg2_role):
                raise ValueError("预期角色重复或不属于配置双腿")
            if str(parsed.symbol).rstrip("0123456789").upper() != args.product:
                raise ValueError("预期合约与品种不符")
            from bomber.framework.dataprep.live_portfolio import ctp_futures_symbol
            ctp_futures_symbol(str(parsed.symbol), str(parsed.venue))
            expected[role] = instrument
        if expected and set(expected) != {args.leg1_role, args.leg2_role}:
            raise ValueError("预期合约须完整覆盖两个角色")
        args.expected_instruments = expected
        if not args.max_notional.is_finite() or args.max_notional <= 0:
            raise ValueError("max-notional须为有限正数")
        if args.limit_offset_ticks < 0 or args.max_session_orders < 1:
            raise ValueError("跳价偏移须非负，订单上限须为正")
        if any(not math.isfinite(v) or v <= 0 for v in (args.seconds, args.query_timeout)):
            raise ValueError("运行时长和查询超时须为有限正数")
        if not math.isfinite(args.reference_refresh_seconds) or not 0 <= args.reference_refresh_seconds <= 60:
            raise ValueError("参考刷新间隔须在[0,60]秒")
        if any(v < 1 for v in (args.reference_timeout, args.reference_max_source_age_days,
                args.reference_max_observation_seconds, args.reference_max_refresh_duration_seconds)):
            raise ValueError("参考读取超时和时效限额须为正")
        if args.mode == "recording":
            if args.enable_orders or args.confirm_simnow or args.state_file or args.resume or args.require_fills:
                raise ValueError("Recording只记录目标，不接受报单授权、交易状态文件或成交验收")
            if args.account_reconciliation not in (None, "off"):
                raise ValueError("Recording不连接TD，对账须为off")
            args.account_reconciliation = "off"
        else:
            if args.state_file is None:
                raise ValueError("Model/SimNow须独立state-file")
            if args.state_file.exists() and not args.state_file.is_file():
                raise ValueError("state-file不是普通文件")
            if args.resume != args.state_file.is_file():
                raise ValueError("恢复须--resume和已有原文件；新会话须使用不存在的文件")
            if args.mode == "simnow" and not (args.enable_orders and args.confirm_simnow):
                raise ValueError("SimNow须同时--enable-orders和--confirm-simnow")
            if args.mode == "model" and (args.enable_orders or args.confirm_simnow):
                raise ValueError("Model不接受物理账户报单授权")
        args.execution_policy = execution_policy_from_args(args)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    return args


def prepare_inputs(args, context, lifecycle):
    from dataclasses import replace
    from bomber.framework.market.stream.ctp import CtpLiveDataFeed, CtpMdConfig
    source = ReferenceSourceFactory.create_for(DataSourcePurpose.LIVE_REFERENCE, "dolphindb",
        DolphinDbReferenceConfig.from_env(database=args.reference_database,
            read_timeout_seconds=args.reference_timeout))
    context.resources.callback(source.close)
    source.open()
    service = LiveFuturesReferences(source, products=(args.product,), trading_day=context.trading_day,
        started_ns=time.time_ns(), roles=(args.leg1_role, args.leg2_role), factor_roles=(),
        factor_date_basis="source", factor_availability="source-day-end",
        refresh_seconds=args.reference_refresh_seconds,
        freshness=ReferenceFreshnessPolicy(
            expected_source_day=datetime.strptime(args.expected_source_day, "%Y%m%d").date(),
            max_source_age_days=args.reference_max_source_age_days,
            max_observation_age_seconds=args.reference_max_observation_seconds,
            max_refresh_duration_seconds=args.reference_max_refresh_duration_seconds))
    references = CalendarReferences(service, args.product, (args.leg1_role, args.leg2_role))
    references.assert_expected(args.expected_instruments)
    context.references = references
    print(f"跨期参考: product={args.product} legs={references.instrument_ids} "
          f"reference_day={context.trading_day} source_day={references.spec[0]} 因子=不使用", flush=True)
    config = build_md_config(CtpMdConfig, lifecycle, flow_path="/tmp/bomber-calendar-spread-md")
    upstream = CtpLiveDataFeed(replace(config, dce_use_receive_date=args.simnow_environment == "realtime"))
    return SimpleNamespace(references=references, upstream=upstream)


def describe(session):
    if session is None:
        return {}
    strategy, runner = session.strategy, session.runner
    positions = runner.position_manager
    return {"complete_frames": strategy.complete_frames, "lookback": strategy.config.lookback,
        "submissions": strategy.submissions, "last_frame_ns": strategy.last_frame_ns,
        "direction": strategy.direction, "last_z": strategy.last_z,
        "active_pair": strategy.active_pair, "last_targets": dict(strategy.last_targets or {}),
        "order_updates_received": strategy.order_updates_received, "fills_received": strategy.fills_received,
        "per_instrument_bar_count": {str(i): v for i, v in runner.bar_counts.items()},
        "strategy_positions": {str(i): positions.position(strategy.strategy_id, str(i)) for i in runner.fixed_ids},
        "execution_positions": {str(i): positions.account_position(CLIENT_ID, i) for i in runner.fixed_ids},
        "working_quantities": {str(i): positions.working_quantity(CLIENT_ID, i) for i in runner.fixed_ids},
        "unassigned_positions": {str(i): positions.unassigned_position(CLIENT_ID, i) for i in runner.fixed_ids},
        "account_positions": None if not runner.orders else observed_account_positions(
            session.client, runner.fixed_ids, positions=positions),
        "portfolio_failure": getattr(session.client, "portfolio_failure", None)}


def build_runtime(args, *, account_routes=()):
    if account_routes and args.mode != "model":
        raise ValueError("跟随账户装配须使用model策略主账本")
    if args.mode == "simnow":
        from bomber.framework.trader.execution.ctp import CtpTdApiTransport
        transport = build_simnow_transport(CtpTdApiTransport, client_id=CLIENT_ID,
            flow_path="/tmp/bomber-calendar-spread-td", timeout_seconds=args.query_timeout)
        lifecycle = CalendarCtpLifecycle(transport, md_front=required("CTP_MD_ADDRESS"), orders=True,
            state_file=args.state_file, resume=args.resume, require_fills=args.require_fills,
            expected_trading_day=args.expected_trading_day, environment=args.simnow_environment,
            replay_md_trading_day=args.replay_md_trading_day, max_session_orders=args.max_session_orders,
            scope_to_reference_instruments=True, account_reconciliation=args.execution_policy.account_reconciliation)
    else:
        lifecycle = CalendarLocalLifecycle(mode=args.mode, require_fills=args.require_fills,
            expected_trading_day=args.expected_trading_day, environment=args.simnow_environment,
            replay_md_trading_day=args.replay_md_trading_day, resume=args.resume, state_file=args.state_file)
    report = LiveRunReport(args.report_dir, prefix=f"calendar-spread-{args.mode}", describe=describe,
        metadata={"mode": args.mode, "strategy_id": CLIENT_ID, "product": args.product,
            "leg1_role": args.leg1_role, "leg2_role": args.leg2_role,
            "quantity": args.quantity, "lookback": args.lookback, "entry_z": args.entry_z,
            "exit_z": args.exit_z, "rebalance_interval": args.rebalance_interval,
            "expected_source_day": args.expected_source_day,
            "expected_trading_day": args.expected_trading_day,
            "expected_instruments": args.expected_instruments, "reference_source": "dolphindb",
            "execution_policy": args.execution_policy.snapshot_state(),
            "state_file": str(args.state_file) if args.state_file else None,
            "initial_position_mode": "resume" if args.resume else "fresh",
            "history_minutes": 0, "session_scope": "fixed_pair_fixed_trading_day"})
    return ManagedLiveRuntime(CLIENT_ID, controller=lifecycle,
        prepare_inputs=lambda context: prepare_inputs(args, context, lifecycle),
        assemble_session=lambda inputs, context: assemble(args, inputs.references, lifecycle.driver,
            inputs.upstream, account_routes=account_routes), report=report, seconds=args.seconds)


def main(argv=None):
    args = parse_args(argv)
    load_dotenv(PROJECT_ROOT / ".env")
    print(f"Calendar Live source={Path(__file__).resolve()}", flush=True)
    runtime = build_runtime(args)
    try:
        runtime.run()
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
