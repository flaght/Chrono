"""当前主力 EMA 的 SimNow 单交易日入口；默认只记录目标。"""

from __future__ import annotations

import argparse
from datetime import datetime
from decimal import Decimal
import os
from pathlib import Path
import time
from types import SimpleNamespace

from dotenv import load_dotenv

from bomber.framework.datahub.sector_roles import SectorDataUnavailable
from bomber.framework.dataprep.paths import resolve_paths
from bomber.framework.dataprep.live_references import live_factor_policy
from bomber.framework.dataprep.live_role import FileRoleReferences, SourceRoleReferences
from bomber.framework.dataprep.sources import DolphinDbReferenceConfig, ReferenceSourceFactory
from bomber.framework.market.stream import ReceiveTimeTradeTickBarFeed, TradeTickBarFeed
from bomber.framework.market.stream.ctp import CtpLiveDataFeed, CtpMdConfig
from bomber.framework.market.basic.base import DataType
from bomber.framework.trader import DataBinding, ExecutionRoute, RiskLimits
from bomber.framework.trader.assembly import RoleGuard, StrategyBindings, assemble_strategy
from bomber.framework.trader.execution.builders import build_ctp_execution, attach_ctp_persistence
from bomber.framework.trader.live_roles import SessionRoleLiveRunner
from bomber.framework.trader.execution.ctp import CtpTdApiTransport
from bomber.framework.trader.runtime.ctp import (
    CtpSessionLifecycle, assert_flat_account, session_ready, validate_replay_environment)
from bomber.framework.trader.runtime.managed import ManagedLiveRuntime
from bomber.framework.trader.runtime.reports import LiveRunReport

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
    parser.add_argument("--simnow-environment", choices=("realtime", "replay"), default="realtime",
                        help="replay仅用于第二套工程联调，按实际Tick接收时间聚合Bar")
    parser.add_argument("--replay-md-trading-day", help="replay必填：原始Tick实际TradingDay，YYYYMMDD")
    parser.add_argument("--product", required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--contract-struct", type=Path)
    parser.add_argument("--fut-basic", type=Path)
    parser.add_argument("--factors", type=Path)
    parser.add_argument("--factor-date-basis", choices=("source", "trading"),
                        help="数据库默认source(此前角色来源日)，文件默认trading(已对齐TD日)")
    parser.add_argument("--factor-availability", choices=("aligned", "source-day-end", "explicit", "observed-on-read"),
                        help="默认随口径选择；有available_ns时仍检查实际发布时间")
    parser.add_argument("--reference-source", choices=("file", "dolphindb"), default="file")
    parser.add_argument("--reference-database", help="DolphinDB DFS路径；默认读取环境配置")
    parser.add_argument("--reference-refresh-seconds", type=float, default=5)
    parser.add_argument("--reference-timeout", type=int, default=15)
    parser.add_argument("--expected-trading-day", help="可选核对TD返回的YYYYMMDD交易日")
    parser.add_argument("--expected-source-day", help="报单必填：已核对的角色资料来源日YYYYMMDD")
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
    args.factor_date_basis = args.factor_date_basis or ("source" if args.reference_source == "dolphindb" else "trading")
    try:
        args.factor_availability = live_factor_policy(args.factor_date_basis, args.factor_availability)
    except ValueError as error:
        parser.error(str(error))
    if not args.connect:
        parser.error("须显式 --connect 允许连接MD／TD")
    if args.simnow_environment == "replay":
        try:
            value = args.replay_md_trading_day or ""
            if len(value) != 8 or not value.isdigit():
                raise ValueError()
            datetime.strptime(value, "%Y%m%d")
        except ValueError:
            parser.error("replay模式须显式提供有效--replay-md-trading-day YYYYMMDD")
    elif args.replay_md_trading_day:
        parser.error("--replay-md-trading-day仅用于replay模式")
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
    if not 0 <= args.reference_refresh_seconds <= 60 or not 1 <= args.reference_timeout <= 120:
        parser.error("参考资料刷新间隔须在0至60秒，读取超时须在1至120秒")
    if args.reference_source == "file" and args.reference_database:
        parser.error("--reference-database 仅用于 dolphindb 数据源")
    if args.reference_source == "file" and args.factor_availability == "observed-on-read":
        parser.error("observed-on-read仅用于在线数据库资料")
    if args.simnow_environment == "replay" and args.factor_availability == "observed-on-read":
        parser.error("observed-on-read仅用于实时会话")
    if args.expected_source_day:
        try:
            if datetime.strptime(args.expected_source_day, "%Y%m%d").strftime("%Y%m%d") != args.expected_source_day:
                raise ValueError("来源日格式错误")
        except ValueError:
            parser.error("预期角色来源日须为有效YYYYMMDD")
    if args.mode == "simnow":
        if not args.enable_orders or not args.confirm_simnow or args.state_file is None:
            parser.error("simnow模式须同时提供--enable-orders --confirm-simnow --state-file")
        if args.state_file.exists():
            parser.error("状态文件已存在：保留旧检查点并先对账，本入口不自动恢复或覆盖")
        if not args.expected_source_day:
            parser.error("simnow报单须显式--expected-source-day核对角色资料来源日")
    elif args.enable_orders or args.confirm_simnow or args.state_file:
        parser.error("recording模式不接受报单授权或交易状态文件")
    return args


def prepare_inputs(args, context, lifecycle):
    """选择参考资料与行情积木；交易日来自公共生命周期的只读预检。"""
    if args.reference_source == "file":
        paths = resolve_paths(vars(args), project_root=PROJECT_ROOT,
            required_files=("contract_struct", "fut_basic"))
        factors = args.factors or (paths.role / "fut_adjustment_factors.feather" if paths.role else None)
        if factors is None:
            raise ValueError("须配置角色数据目录或显式--factors")
        references = FileRoleReferences(product=args.product, trading_day=context.trading_day,
            contract_struct=paths.contract_struct, factors=factors, fut_basic=paths.fut_basic,
            started_ns=time.time_ns(), factor_date_basis=args.factor_date_basis,
            factor_availability=args.factor_availability,
            allowed_venues=("SHFE", "INE"), required_currency="CNY")
    else:
        source = ReferenceSourceFactory.create("dolphindb", DolphinDbReferenceConfig.from_env(
            database=args.reference_database, read_timeout_seconds=args.reference_timeout))
        # open或适配器构造失败也由Runtime释放已创建的源。
        context.resources.callback(source.close)
        source.open()
        references = SourceRoleReferences(source, product=args.product,
            trading_day=context.trading_day, started_ns=time.time_ns(),
            factor_date_basis=args.factor_date_basis, factor_availability=args.factor_availability,
            refresh_seconds=args.reference_refresh_seconds,
            allowed_venues=("SHFE", "INE"), required_currency="CNY")
    context.references = references
    assignment = references.snapshot(time.time_ns())
    if args.expected_source_day and assignment.source_day.strftime("%Y%m%d") != args.expected_source_day:
        raise SectorDataUnavailable(
            f"角色资料来源日不符: expected={args.expected_source_day} actual={assignment.source_day}")
    upstream = CtpLiveDataFeed(CtpMdConfig(front=lifecycle.md_front,
        broker_id="9999", user_id=lifecycle.transport.investor_id,
        password=lifecycle.transport.password,
        flow_path=os.getenv("CTP_MD_FLOW_PATH", "/tmp/bomber-main-ema-md"),
        production_mode=lifecycle.transport.production_mode))
    print(f"主力EMA: mode={args.mode} TD交易日={context.trading_day} source_day={assignment.source_day} "
          f"factor_basis={references.factor_date_basis} factor_day={references.factor_date} "
          f"main={references.instrument_id} factor={assignment.factor(args.product.upper(), 'main')} "
          f"tick={references.spec.tick} multiplier={references.spec.multiplier}", flush=True)
    return SimpleNamespace(references=references, upstream=upstream)


def assemble(args, references, driver, upstream, *, positions=None):
    """与run_backtest相同：执行积木→策略→绑定声明→Runner。"""
    if references.spec.venue not in {"SHFE", "INE"} or references.spec.currency != "CNY":
        raise ValueError("首版主力EMA SimNow只支持CNY的SHFE／INE期货")
    instrument = references.instrument_id
    expected_day = references.trading_day.strftime("%Y%m%d")
    orders = args.mode == "simnow"
    replay = getattr(args, "simnow_environment", "realtime") == "replay"
    replay_day = getattr(args, "replay_md_trading_day", None) if replay else None
    if replay and not replay_day:
        raise ValueError("第二套联调缺少固定MD交易日")
    feed_type = ReceiveTimeTradeTickBarFeed if replay else TradeTickBarFeed
    feed = feed_type("MAIN_EMA_1M", upstream)
    feed.register_instrument(references.instrument_meta())
    check = lambda: session_ready(driver, upstream, expected_day,
        require_orders=orders, replay_md_trading_day=replay_day)
    execution = build_ctp_execution(driver, instrument_id=instrument, trading_day=expected_day,
        price_increment=references.spec.tick, multiplier=references.spec.multiplier,
        risk_limits=RiskLimits(max_order_quantity=args.quantity, max_abs_position=args.quantity,
            max_order_notional=args.max_notional, max_abs_position_notional=args.max_notional,
            max_market_age_ns=120_000_000_000, contract_multiplier=references.spec.multiplier),
        session_check=check, orders=orders, limit_offset_ticks=args.limit_offset_ticks, positions=positions)
    strategy = MainEmaStrategy("main-ema-simnow", references, MainEmaConfig(
        args.product, references.spec.venue, args.fast, args.slow, args.quantity))
    runner = SessionRoleLiveRunner(references=references, session_check=check,
        orders=orders, position_manager=execution.positions)
    bindings = StrategyBindings(
        data=(DataBinding(str(instrument), "ctp-bars", instrument, DataType.BAR, "1-MINUTE"),),
        execution=(ExecutionRoute(strategy.config.target_key, execution.client.client_id, instrument),),
        role_guard=RoleGuard(references, args.product, "main", lambda: strategy.last_processed_ns))
    assemble_strategy(runner, strategy, feeds={"ctp-bars": feed}, execution=execution, bindings=bindings)
    manager = attach_ctp_persistence(execution, runner, state_file=args.state_file) if orders else None
    return SimpleNamespace(runner=runner, strategy=strategy, execution=execution, client=execution.client,
        driver=driver, upstream=upstream, ledger=execution.ledger, manager=manager, references=references,
        bar_feed=feed, replay_md_trading_day=replay_day)


def build_runtime(args):
    """选择通道生命周期、策略组装工厂与报告积木；构造期间不连接。"""
    if required("CTP_BROKER_ID") != "9999":
        raise ValueError("本入口只允许SimNow BrokerID=9999")
    investor = required("CTP_ACCOUNT_ID")
    transport = CtpTdApiTransport(client_id=CLIENT_ID, account_id=investor,
        front=required("CTP_TD_ADDRESS"), broker_id="9999", investor_id=investor,
        password=required("CTP_PASSWORD"), app_id=os.getenv("CTP_APP_ID", ""),
        auth_code=os.getenv("CTP_AUTH_CODE", ""),
        flow_path=os.getenv("CTP_TD_FLOW_PATH", "/tmp/bomber-main-ema-td"),
        production_mode=os.getenv("CTP_PRODUCTION_MODE", "true").lower() in {"1", "true", "yes", "on"},
        timeout_seconds=args.query_timeout)
    lifecycle = CtpSessionLifecycle(transport, md_front=required("CTP_MD_ADDRESS"),
        orders=args.mode == "simnow", max_session_orders=args.max_session_orders,
        environment=args.simnow_environment, replay_md_trading_day=args.replay_md_trading_day,
        expected_trading_day=args.expected_trading_day, legacy_lock_namespaces=("bomber-main-ema",))
    report = LiveRunReport(args.report_dir, prefix="simnow", metadata={
        "mode": args.mode, "product": args.product, "reference_source": args.reference_source,
        "state_file": str(args.state_file) if args.state_file else None,
        "expected_source_day": args.expected_source_day,
        "factor_availability": args.factor_availability, "factor_date_basis": args.factor_date_basis},
        describe=lambda session: {"bars_used": session.strategy.bars_used if session else 0,
            "ema": {"fast": args.fast, "slow": args.slow, "quantity": str(args.quantity)}})
    seen = [0]

    def progress(session):
        if session.strategy.bars_used != seen[0]:
            seen[0] = session.strategy.bars_used
            print(f"EMA分钟进度: bars_used={seen[0]} slow={args.slow}", flush=True)

    return ManagedLiveRuntime("main-ema-simnow-runtime", controller=lifecycle,
        prepare_inputs=lambda context: prepare_inputs(args, context, lifecycle),
        assemble_session=lambda inputs, context: assemble(args, inputs.references,
            lifecycle.driver, inputs.upstream), report=report, seconds=args.seconds, progress=progress)


def run_session(args):
    runtime = build_runtime(args)
    try:
        return runtime.run()
    finally:
        runtime.stop()


def main(argv=None):
    args = parse_args(argv)
    load_dotenv(PROJECT_ROOT / ".env")
    import bomber
    import bomber.framework
    print(f"Python模式=module Bomber={getattr(bomber, '__version__', 'unknown')} "
          f"framework={bomber.framework.__file__} 策略={Path(__file__).with_name('strategy.py')}", flush=True)
    run_session(args)


if __name__ == "__main__":
    main()
