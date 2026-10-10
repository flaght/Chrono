"""当前主力 EMA 的 SimNow 单交易日入口；默认只记录目标。"""

from __future__ import annotations

import argparse
from datetime import datetime
from decimal import Decimal
from pathlib import Path
import time
from types import SimpleNamespace

from dotenv import load_dotenv
from bomber.framework.trader.runtime.live.channels.ctp import (
    required, build_simnow_transport, build_md_config)
from bomber.framework.trader.runtime.live.profiles import build_minute_feed
from bomber.framework.trader.runtime.live.contracts import SessionReadiness
from bomber.framework.trader.runtime.live.channels.ctp import (
    CtpLiveProfile, InstrumentHealthGate, CtpMarketChannel, CtpExecutionChannel,
    add_environment_arguments, validate_environment_arguments)

from bomber.framework.datahub.sector_roles import SectorDataUnavailable
from bomber.framework.dataprep.paths import resolve_paths
from bomber.framework.dataprep.live_references import live_factor_policy
from bomber.framework.dataprep.reference_freshness import ReferenceFreshnessPolicy
from bomber.framework.dataprep.live_role import FileRoleReferences, SourceRoleReferences
from bomber.framework.dataprep.live_cached_role import CachedRoleReferences
from bomber.framework.dataprep.sources import DolphinDbReferenceConfig, ReferenceSourceFactory
from bomber.framework.market.stream.ctp import CtpLiveDataFeed, CtpMdConfig
from bomber.framework.market.basic.base import DataType, InstrumentId
from bomber.framework.trader import DataBinding, ExecutionRoute, RiskLimits
from bomber.framework.trader.assembly import RoleGuard, StrategyBindings, assemble_strategy
from bomber.framework.trader.execution.builders import build_ctp_execution, attach_ctp_persistence
from bomber.framework.trader.live_roles import SessionRoleLiveRunner
from bomber.framework.trader.execution.ctp import CtpTdApiTransport
from bomber.framework.trader.runtime.live.channels.ctp import (
    CtpSessionLifecycle, assert_flat_account, session_ready, validate_replay_environment)
from bomber.framework.trader.runtime.live.runtime import ManagedLiveRuntime
from bomber.framework.trader.runtime.live.recovery import StrategyPositionBinding, SessionIdentityCheckpoint
from bomber.framework.trader.runtime.live.channels.ctp_recovery import CtpResumeLifecycle
from bomber.framework.trader.runtime.reports import LiveRunReport

from .strategy import MainEmaConfig, MainEmaStrategy
from .recovery import recovery_identity, validate_restored_session, verify_managed_position

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLIENT_ID = "main-ema-simnow"


class MainEmaSessionLifecycle(CtpResumeLifecycle, CtpSessionLifecycle):
    def validate_restored_session(self, session, day):
        return validate_restored_session(session, day)

    def verify(self, session):
        if self.orders and (self.resuming or self.adopted_position is not None):
            verify_managed_position(session, self.transport)
            return
        if self.orders and not self.driver.submitted_orders:
            target = session.strategy.last_target
            position = session.strategy.position(session.strategy.config.target_key)
            raise RuntimeError(f"成交验收未触发：目标={target} 策略仓位={position} "
                f"signal_source={session.strategy.signal_source}，本次报单0；不算成交通过")
        super().verify(session)

    def snapshot(self, session, context):
        return {**super().snapshot(session, context),
            "signal_state_recovery": "rewarm" if self.resuming else "fresh",
            "verification_basis": ("target_and_position_reconciliation"
                if self.resuming or self.adopted_position is not None else "new_signal_and_execution")}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect", action="store_true")
    parser.add_argument("--mode", choices=("recording", "simnow"), default="recording")
    add_environment_arguments(parser)
    parser.add_argument("--product", required=True)
    parser.add_argument("--expected-instrument", help="核对本次真实合约；IM Recording须显式声明，含交易所后缀")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--contract-struct", type=Path)
    parser.add_argument("--fut-basic", type=Path)
    parser.add_argument("--factors", type=Path)
    parser.add_argument("--factor-date-basis", choices=("source", "trading"),
                        help="数据库默认source(此前角色来源日)，文件默认trading(已对齐TD日)")
    parser.add_argument("--factor-availability", choices=("aligned", "source-day-end", "explicit", "observed-on-read"),
                        help="默认随口径选择；有available_ns时仍检查实际发布时间")
    parser.add_argument("--reference-source", choices=("file", "dolphindb"), default="dolphindb")
    parser.add_argument("--allow-file-reference-test", action="store_true",
                        help="显式文件资料测试／联调；正式在线使用DolphinDB")
    parser.add_argument("--reference-max-source-age-days", type=int, default=14,
                        help="角色／因子来源距TD日的最大自然日数；预期来源日须另行核对交易日历")
    parser.add_argument("--reference-max-observation-seconds", type=int, default=30)
    parser.add_argument("--reference-max-refresh-duration-seconds", type=int, default=30)
    parser.add_argument("--reference-database", help="DolphinDB DFS路径；默认读取环境配置")
    parser.add_argument("--reference-refresh-seconds", type=float, default=5)
    parser.add_argument("--reference-timeout", type=int, default=15)
    parser.add_argument("--expected-trading-day", help="可选核对TD返回的YYYYMMDD交易日")
    parser.add_argument("--expected-source-day", help="在线数据库必填：按日历核对的最近角色资料来源日YYYYMMDD")
    parser.add_argument("--adopt-existing-position", help="显式接管启动前已有的一手主力仓，合约含交易所后缀")
    parser.add_argument("--expected-position", type=Decimal, help="接管预期净仓，仅支持1或-1")
    parser.add_argument("--fast", type=int, default=3)
    parser.add_argument("--slow", type=int, default=5)
    parser.add_argument("--quantity", type=Decimal, default=Decimal(1))
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--run-forever", action="store_true", help="按CU交易日历持续会话，休市暂停并恢复同一检查点")
    parser.add_argument("--resume", action="store_true", help="恢复原状态文件；单次会话同交易日对账后重新预热EMA，持续模式恢复EMA检查点")
    parser.add_argument("--resume-report", type=Path, help="首次迁移旧版首次开仓状态时对应的summary.json")
    parser.add_argument("--recovery-bars", type=Path, help="权威完整分钟JSONL：ts_event/instrument_id/adjusted_close，恢复缺口只预热不报单")
    parser.add_argument("--history-minutes", type=int, default=0, help="声明启动预热窗口；0保留原实时预热，240表示240根完整交易分钟")
    parser.add_argument("--history-db", default="dolphindb", help="注册的历史数据库Provider名称，默认dolphindb")
    parser.add_argument("--history-config", type=Path, help="历史Provider表/字段/时间口径JSON配置，凭据复用环境变量")
    parser.add_argument("--history-file", type=Path, help="盘前优先本地JSONL；无文件或不足则数据库补齐")
    parser.add_argument("--history-stage", choices=("auto", "preopen", "recovery"), default="auto")
    parser.add_argument("--history-missing-policy", choices=("fail", "allow"), default="fail")
    parser.add_argument("--trading-calendar", type=Path,
                        default=Path(__file__).with_name("shfe_cu_2026.json"))
    parser.add_argument("--query-timeout", type=float, default=15)
    parser.add_argument("--max-notional", type=Decimal, default=Decimal(50000))
    parser.add_argument("--limit-offset-ticks", type=int, default=1)
    parser.add_argument("--max-session-orders", type=int, default=4)
    parser.add_argument("--enable-orders", action="store_true")
    parser.add_argument("--confirm-simnow", action="store_true")
    parser.add_argument("--state-file", type=Path, help="报单模式必填；新会话用新文件，--resume使用原文件")
    parser.add_argument("--report-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args(argv)
    if args.product.upper() == "IM":
        if args.mode != "recording" or args.run_forever or args.simnow_environment != "realtime":
            parser.error("IM首阶段仅支持realtime Recording单次观察")
        if not args.expected_instrument:
            parser.error("IM Recording须声明--expected-instrument核对固定真实合约")
    if args.expected_instrument:
        try:
            expected = InstrumentId.from_str(args.expected_instrument)
            if not str(expected.symbol).upper().startswith(args.product.upper()):
                raise ValueError()
            if args.product.upper() == "IM" and str(expected.venue) != "CFFEX":
                raise ValueError()
        except (ValueError, TypeError):
            parser.error("预期合约须匹配品种和交易所并带后缀")
    args.factor_date_basis = args.factor_date_basis or ("source" if args.reference_source == "dolphindb" else "trading")
    try:
        args.factor_availability = live_factor_policy(args.factor_date_basis, args.factor_availability)
    except ValueError as error:
        parser.error(str(error))
    if not args.connect:
        parser.error("须显式 --connect 允许连接MD／TD")
    if args.reference_source == "file" and not args.allow_file_reference_test:
        parser.error("文件资料仅用于明确测试，须--allow-file-reference-test；正式在线使用DolphinDB")
    if args.allow_file_reference_test and args.reference_source != "file":
        parser.error("--allow-file-reference-test仅用于file数据源")
    if args.reference_source == "dolphindb" and not args.expected_source_day and not args.run_forever:
        parser.error("在线数据库资料须--expected-source-day声明已核对的最近来源交易日")
    try:
        ReferenceFreshnessPolicy(max_source_age_days=args.reference_max_source_age_days,
            max_observation_age_seconds=args.reference_max_observation_seconds,
            max_refresh_duration_seconds=args.reference_max_refresh_duration_seconds)
    except ValueError as error:
        parser.error(str(error))
    if args.reference_refresh_seconds > args.reference_max_observation_seconds:
        parser.error("资料刷新间隔不能超过最近观测年龄上限")
    try:
        validate_environment_arguments(args)
    except ValueError as error:
        parser.error(str(error))
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
    if bool(args.adopt_existing_position) != (args.expected_position is not None):
        parser.error("接管须同时提供--adopt-existing-position和--expected-position")
    if args.adopt_existing_position:
        if not args.expected_position.is_finite() or args.expected_position not in {Decimal(1), Decimal(-1)} or args.quantity != 1:
            parser.error("首版接管仅支持一手单向仓和一手目标")
        try:
            instrument = InstrumentId.from_str(args.adopt_existing_position)
            if str(instrument.venue) not in {"SHFE", "INE"}:
                raise ValueError("交易所不支持")
        except (ValueError, TypeError):
            parser.error("接管合约须为真实SHFE／INE合约并带交易所后缀")
        if not args.expected_source_day or not args.expected_trading_day:
            parser.error("接管须声明预期TD日及资料来源日")
        if args.mode == "simnow" and args.max_session_orders < 2:
            parser.error("接管反手至少需要--max-session-orders 2，平仓和开仓各一笔")
    if args.expected_source_day:
        try:
            if datetime.strptime(args.expected_source_day, "%Y%m%d").strftime("%Y%m%d") != args.expected_source_day:
                raise ValueError("来源日格式错误")
        except ValueError:
            parser.error("预期角色来源日须为有效YYYYMMDD")
    if args.mode == "simnow":
        if not args.enable_orders or not args.confirm_simnow or args.state_file is None:
            parser.error("simnow模式须同时提供--enable-orders --confirm-simnow --state-file")
        if args.resume and (not args.state_file.is_file() or args.adopt_existing_position):
            parser.error("恢复须使用已有原state-file，且不能同时接管仓位")
        if args.resume and not args.run_forever and not args.expected_trading_day:
            parser.error("单次恢复须声明--expected-trading-day核对相同TD交易日")
        if args.state_file.exists() and not args.resume:
            parser.error("状态文件已存在：保留旧检查点并先对账，本入口不自动恢复或覆盖")
        if not args.expected_source_day and not args.run_forever:
            parser.error("simnow报单须显式--expected-source-day核对角色资料来源日")
    elif args.enable_orders or args.confirm_simnow or args.state_file or args.resume:
        parser.error("recording模式不接受报单授权、交易状态文件或恢复参数")
    if args.run_forever:
        if (args.mode != "simnow" or args.simnow_environment != "realtime"
                or args.product.upper() != "CU" or args.reference_source != "dolphindb"
                or args.adopt_existing_position or args.expected_position is not None
                or args.expected_trading_day or args.expected_source_day):
            parser.error("持续模式须为realtime SimNow CU/dolphindb，不使用接管或固定TD/来源日；日期由会话权威核对")
        if args.quantity != 1 or not args.trading_calendar.is_file():
            parser.error("首期持续模式限定一手目标且须提供有效交易日历")
        if args.resume and not args.state_file.is_file():
            parser.error("--resume要求已有检查点")
    elif args.resume_report or args.recovery_bars:
        parser.error("resume-report/recovery-bars仅用于--run-forever")
    if args.resume and not args.run_forever and (args.history_minutes or args.history_config or args.history_file):
        parser.error("单次恢复使用当前完整分钟重新预热，不同时装配历史预热")
    if args.resume_report and not (args.resume and args.resume_report.is_file()):
        parser.error("旧报告迁移须--resume且summary.json存在")
    if args.recovery_bars and not args.recovery_bars.is_file():
        parser.error("恢复分钟文件不存在")
    if not 0 <= args.history_minutes <= 10000 or args.history_db == "file":
        parser.error("历史分钟须为0至10000；history-db须为数据库Provider")
    from bomber.framework.dataprep.history import HistoryProviderFactory
    if args.history_db not in HistoryProviderFactory.backends():
        parser.error("历史数据库Provider未注册")
    if args.history_config and not args.history_config.is_file():
        parser.error("历史Provider配置文件不存在")
    if args.history_minutes or args.history_config or args.history_file:
        if args.product.upper() not in {"CU", "IM"}:
            parser.error("历史窗口当前支持CU/IM，须配置对应会话日历")
        from bomber.framework.trader.runtime.trading_sessions import load_sessions
        try:
            load_sessions(args.trading_calendar, args.product)
        except (ValueError, KeyError, OSError) as error:
            parser.error(f"历史会话日历无效：{error}")
    if args.recovery_bars and (args.history_minutes or args.history_config or args.history_file):
        parser.error("旧recovery-bars和统一历史Provider参数不能同时使用")
    return args


def prepare_inputs(args, context, lifecycle):
    """选择参考资料与行情积木；交易日来自公共生命周期的只读预检。"""
    from bomber.framework.dataprep.sources import DataSourcePurpose, validate_source
    purpose = (DataSourcePurpose.FILE_REFERENCE_TEST if args.reference_source == "file"
        and getattr(args, "allow_file_reference_test", False) else DataSourcePurpose.LIVE_REFERENCE)
    validate_source(args.reference_source, purpose)
    if args.reference_source == "dolphindb" and not args.expected_source_day:
        raise ValueError("在线参考准备须有预期来源日；持续模式应先由会话日历确定来源日")
    freshness = ReferenceFreshnessPolicy(
        expected_source_day=datetime.strptime(args.expected_source_day, "%Y%m%d").date()
            if args.expected_source_day else None,
        max_source_age_days=args.reference_max_source_age_days,
        max_observation_age_seconds=args.reference_max_observation_seconds,
        max_refresh_duration_seconds=args.reference_max_refresh_duration_seconds)
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
            allowed_venues=("CFFEX",) if args.product.upper() == "IM" else ("SHFE", "INE"),
            required_currency="CNY", freshness=freshness)
    else:
        source = ReferenceSourceFactory.create_for(purpose, "dolphindb", DolphinDbReferenceConfig.from_env(
            database=args.reference_database, read_timeout_seconds=args.reference_timeout))
        # open或适配器构造失败也由Runtime释放已创建的源。
        context.resources.callback(source.close)
        source.open()
        references = SourceRoleReferences(source, product=args.product,
            trading_day=context.trading_day, started_ns=time.time_ns(),
            factor_date_basis=args.factor_date_basis, factor_availability=args.factor_availability,
            refresh_seconds=args.reference_refresh_seconds,
            allowed_venues=("CFFEX",) if args.product.upper() == "IM" else ("SHFE", "INE"),
            required_currency="CNY", freshness=freshness)
    if args.reference_source != "file":
        references = CachedRoleReferences(references)
    context.references = references
    if getattr(args, "expected_instrument", None) and references.instrument_id != InstrumentId.from_str(args.expected_instrument):
        raise SectorDataUnavailable(f"本次参考合约不符：expected={args.expected_instrument} actual={references.instrument_id}")
    assignment = references.snapshot(time.time_ns())
    if args.expected_source_day and assignment.source_day.strftime("%Y%m%d") != args.expected_source_day:
        raise SectorDataUnavailable(
            f"角色资料来源日不符: expected={args.expected_source_day} actual={assignment.source_day}")
    upstream = CtpLiveDataFeed(build_md_config(CtpMdConfig, lifecycle,
        flow_path="/tmp/bomber-main-ema-md"))
    print(f"主力EMA: mode={args.mode} TD交易日={context.trading_day} source_day={assignment.source_day} "
          f"factor_basis={references.factor_date_basis} factor_day={references.factor_date} "
          f"main={references.instrument_id} factor={assignment.factor(args.product.upper(), 'main')} "
          f"tick={references.spec.tick} multiplier={references.spec.multiplier}", flush=True)
    return SimpleNamespace(references=references, upstream=upstream)


def assemble(args, references, driver, upstream, *, positions=None):
    """与run_backtest相同：执行积木→策略→绑定声明→Runner。"""
    allowed = {"CFFEX"} if args.product.upper() == "IM" and args.mode == "recording" else {"SHFE", "INE"}
    if references.spec.venue not in allowed or references.spec.currency != "CNY":
        raise ValueError("主力EMA交易所/模式不支持：IM仅CFFEX Recording，其他入口SHFE/INE")
    if getattr(args, "expected_instrument", None) and references.instrument_id != InstrumentId.from_str(args.expected_instrument):
        raise ValueError(f"装配合约与显式预期不一致：expected={args.expected_instrument} "
            f"actual={references.instrument_id} spec.symbol={references.spec.symbol!r} "
            f"venue={references.spec.venue!r} reference_type={type(references).__module__}.{type(references).__name__}")
    instrument = references.instrument_id
    expected_day = references.trading_day.strftime("%Y%m%d")
    orders = args.mode == "simnow"
    replay = getattr(args, "simnow_environment", "realtime") == "replay"
    replay_day = getattr(args, "replay_md_trading_day", None) if replay else None
    if replay and not replay_day:
        raise ValueError("第二套联调缺少固定MD交易日")
    policy = CtpLiveProfile("replay" if replay else "realtime", replay_day)
    health = InstrumentHealthGate(upstream, (instrument,), expected_day, policy=policy,
        max_age_seconds=getattr(args, "instrument_max_age_seconds", 10))
    feed = build_minute_feed("MAIN_EMA_1M", upstream, policy,
        max_age_seconds=getattr(args, "instrument_max_age_seconds", 10))
    feed.register_instrument(references.instrument_meta())
    check = SessionReadiness(
        CtpMarketChannel(upstream, expected_day, profile=policy, instrument_health=health),
        CtpExecutionChannel(driver, expected_day, require_orders=orders))
    execution = build_ctp_execution(driver, instrument_id=instrument, trading_day=expected_day,
        price_increment=references.spec.tick, multiplier=references.spec.multiplier,
        risk_limits=RiskLimits(max_order_quantity=args.quantity, max_abs_position=args.quantity,
            max_order_notional=args.max_notional, max_abs_position_notional=args.max_notional,
            max_market_age_ns=120_000_000_000, contract_multiplier=references.spec.multiplier),
        session_check=check, orders=orders, limit_offset_ticks=args.limit_offset_ticks, positions=positions)
    strategy = MainEmaStrategy("main-ema-simnow", references, MainEmaConfig(
        args.product, references.spec.venue, args.fast, args.slow, args.quantity))
    continuous = getattr(args, "run_forever", False)
    history_enabled = bool(getattr(args, "history_minutes", 0) or getattr(args, "history_config", None)
        or getattr(args, "history_file", None))
    if continuous or history_enabled:
        from .continuous import CheckpointedRunner
        from .checkpoint import EmaCheckpoint
    runner_type = CheckpointedRunner if continuous or history_enabled else SessionRoleLiveRunner
    runner = runner_type(references=references, session_check=check,
        orders=orders, position_manager=execution.positions, instrument_health=health)
    if not getattr(references, "refresh_on_event", True):
        references.publication_lock = runner._submit_lock
    bindings = StrategyBindings(
        data=(DataBinding(str(instrument), "ctp-bars", instrument, DataType.BAR, "1-MINUTE"),),
        execution=(ExecutionRoute(strategy.config.target_key, execution.client.client_id, instrument),),
        role_guard=RoleGuard(references, args.product, "main", lambda: strategy.last_processed_ns))
    assemble_strategy(runner, strategy, feeds={"ctp-bars": feed}, execution=execution, bindings=bindings)
    components = ({"ema": EmaCheckpoint(strategy, instrument)} if continuous or history_enabled else
                  {"live_session": SessionIdentityCheckpoint(recovery_identity(args, references, policy))})
    manager = attach_ctp_persistence(execution, runner, state_file=args.state_file,
        state_components=components) if orders else None
    runner.manager = manager
    if continuous or history_enabled:
        from bomber.framework.trader.runtime.trading_sessions import load_sessions
        runner.calendar = load_sessions(args.trading_calendar, "IM" if args.product.upper() == "IM" else "CU")
        runner.ema_checkpoint = components["ema"]
        runner.recovery_bars = getattr(args, "recovery_bars", None)
        runner.history_service = None
        if history_enabled:
            from .history import build_history, warm_history
            runner.history_service = build_history(args)
            runner.history_minutes = args.history_minutes
            runner.history_buffer = []
            runner.history_buffering = True
            runner.history_stage = getattr(args, "_history_stage", args.history_stage)
            if runner.history_stage == "auto":
                from datetime import datetime as history_datetime
                from bomber.framework.trader.runtime.trading_sessions import SHANGHAI
                runner.history_stage = ("recovery" if runner.calendar.window(history_datetime.now(SHANGHAI)) else "preopen")
            def prepare_history():
                from .history import flush_buffer
                warm_history(runner, runner._clock_ns(), phase=runner.history_stage)
                flush_buffer(runner)
            runner.prepare_history = prepare_history
    legacy_states = None
    if continuous and getattr(args, "resume_report", None):
        from .continuous import legacy_ema_state
        from bomber.framework.trader.persistence import JsonStateStore
        persisted = JsonStateStore(args.state_file).load()
        if persisted and not persisted.payload.get("component_states"):
            legacy_states = legacy_ema_state(args.resume_report, components["ema"], persisted, args.state_file)
    return SimpleNamespace(runner=runner, strategy=strategy, execution=execution, client=execution.client,
        driver=driver, upstream=upstream, ledger=execution.ledger, manager=manager, references=references,
        bar_feed=feed, market_profile=policy, readiness=check,
        recovery_binding=StrategyPositionBinding(strategy.strategy_id, execution.client.client_id,
            {strategy.config.target_key: instrument}),
        replay_md_trading_day=replay_day, legacy_component_states=legacy_states)


def build_runtime(args):
    """选择通道生命周期、策略组装工厂与报告积木；构造期间不连接。"""
    transport = build_simnow_transport(CtpTdApiTransport, client_id=CLIENT_ID,
        flow_path="/tmp/bomber-main-ema-td", timeout_seconds=args.query_timeout)
    lifecycle_type, extra = MainEmaSessionLifecycle, {}
    if getattr(args, "run_forever", False):
        from .continuous import ContinuousLifecycle
        lifecycle_type = ContinuousLifecycle
        extra = {"state_file": args.state_file, "window": args._continuous_window,
                 "stop_event": args._stop_event}
    lifecycle = lifecycle_type(transport, md_front=required("CTP_MD_ADDRESS"),
        orders=args.mode == "simnow", max_session_orders=args.max_session_orders,
        environment=args.simnow_environment, replay_md_trading_day=args.replay_md_trading_day,
        adopt_instrument=args.adopt_existing_position, expected_position=args.expected_position,
        **({"resume": args.resume} if not getattr(args, "run_forever", False) else {}),
        expected_trading_day=args.expected_trading_day, legacy_lock_namespaces=("bomber-main-ema",), **extra)
    report = LiveRunReport(args.report_dir, prefix="simnow", metadata={
        "mode": args.mode, "product": args.product, "reference_source": args.reference_source,
        "expected_instrument": getattr(args, "expected_instrument", None),
        "service_mode": "continuous" if getattr(args, "run_forever", False) else "acceptance",
        "initial_position_mode": ("resume" if getattr(args, "resume", False)
            else "adopt_existing" if args.adopt_existing_position else "fresh_flat"),
        "state_file": str(args.state_file) if args.state_file else None,
        "history_minutes": args.history_minutes,
        "history_missing_policy": args.history_missing_policy,
        "expected_source_day": args.expected_source_day,
        "factor_availability": args.factor_availability, "factor_date_basis": args.factor_date_basis,
        "reference_purpose": "file_reference_test" if args.allow_file_reference_test else "live_reference"},
        describe=lambda session: {"history": (getattr(session.runner, "history_service", None).last_audit
                if session and getattr(session.runner, "history_service", None) else None),
            "history_audit": (session.runner.history_service.audit_history
                if session and getattr(session.runner, "history_service", None) else []),
            "bars_used": session.strategy.bars_used if session else 0,
            "signal_source": session.strategy.signal_source if session else None,
            "last_target": session.strategy.last_target if session else None,
            "fast_ema": session.strategy.fast.value if session else None,
            "slow_ema": session.strategy.slow.value if session else None,
            "order_updates_received": session.strategy.order_updates_received if session else 0,
            "last_order_status": (session.strategy.last_order_update.status.value
                if session and session.strategy.last_order_update else None),
            "last_order_id": (session.strategy.last_order_update.identity.client_order_id
                if session and session.strategy.last_order_update else None),
            "last_order_reason": (session.strategy.last_order_update.reason
                if session and session.strategy.last_order_update else None),
            "last_order_position": session.strategy.last_order_position if session else None,
            "fills_received": session.strategy.fills_received if session else 0,
            "last_fill_position": session.strategy.last_fill_position if session else None,
            "strategy_position": session.runner.position_manager.position(
                session.strategy.strategy_id, session.strategy.config.target_key) if session else None,
            "account_position": session.runner.position_manager.account_position(
                CLIENT_ID, session.references.instrument_id) if session else None,
            "unassigned_position": session.runner.position_manager.unassigned_position(
                CLIENT_ID, session.references.instrument_id) if session else None,
            "ema": {"fast": args.fast, "slow": args.slow, "quantity": str(args.quantity)}})
    seen = [0]

    def progress(session):
        if session.strategy.bars_used != seen[0]:
            seen[0] = session.strategy.bars_used
            print(f"EMA分钟进度: bars_used={seen[0]} slow={args.slow}", flush=True)
            print(f"策略目标: source={session.strategy.signal_source} target={session.strategy.last_target} "
                  f"fast={session.strategy.fast.value} slow={session.strategy.slow.value}", flush=True)

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
    if args.run_forever:
        from .continuous import run_continuous
        run_continuous(args, build_runtime)
    else:
        run_session(args)


if __name__ == "__main__":
    main()
