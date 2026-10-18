"""期权Vega在线入口：行情来源、指数来源、执行渠道分别装配。"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta
from dataclasses import replace
from decimal import Decimal
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from zoneinfo import ZoneInfo

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "demos.08_option_vega"

from .vega_signal import OptionVegaConfig


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--connect", action="store_true")
    p.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    p.add_argument("--mode", choices=("recording", "model", "simnow"), default="recording")
    p.add_argument("--market-source", choices=("ctp", "dolphindb"), default="ctp")
    p.add_argument("--index-source", choices=("dolphindb", "dolphindb-dfs"), default="dolphindb")
    p.add_argument("--index-feature-config", type=Path, default=Path(__file__).resolve().parents[2] /
        "bomber/framework/dataprep/features/configs/features.json")
    p.add_argument("--index-dataset", default="index.minute")
    p.add_argument("--index-schema-version", type=int, default=1)
    p.add_argument("--index-database", help="可选覆盖配置中的DFS路径")
    p.add_argument("--index-table", help="可选覆盖配置中的原表名")
    p.add_argument("--index-availability", choices=("observed", "source-time", "update-time"),
        help="update-time仅在源updateTime确为首次可用/修订发布时间时选用")
    p.add_argument("--index-poll-seconds", type=float, default=.2)
    p.add_argument("--index-query-timeout", type=int, default=2)
    p.add_argument("--index-price-precision", type=int, default=4)
    p.add_argument("--index-size-precision", type=int, default=4)
    p.add_argument("--reference-source", choices=("dolphindb", "file"), default="dolphindb")
    p.add_argument("--allow-file-reference-test", action="store_true")
    p.add_argument("--opt-basic", type=Path)
    p.add_argument("--fut-basic", type=Path)
    p.add_argument("--reference-database")
    p.add_argument("--reference-timeout", type=int, default=30)
    p.add_argument("--reference-refresh-seconds", type=float, default=5)
    p.add_argument("--reference-max-observation-seconds", type=float, default=30)
    p.add_argument("--simnow-environment", choices=("realtime", "replay"), default="realtime",
        help="replay只豁免原始历史日期年龄；策略仍按原始行情时间计算")
    p.add_argument("--replay-md-trading-day")
    p.add_argument("--replay-test-start-time", help="08工程测试：将第一份完成Bar映射到估值日HH:MM")
    p.add_argument("--replay-test-source-time", help="可选显式源HH:MM；例如16:00映射到start-time=09:00")
    p.add_argument("--expected-trading-day", required=True, help="执行/参考会话交易日YYYYMMDD")
    p.add_argument("--valuation-day", help="行情估值日；replay默认MD日，realtime默认会话交易日")
    p.add_argument("--calendar-id", default="china.cffex", help="公共编译日历身份；Live不读取日历文件")
    p.add_argument("--option-product", default="MO")
    p.add_argument("--future-product", default="IM")
    p.add_argument("--index-code", default="000852")
    p.add_argument("--index-bar-table", help="DolphinDB指数已收盘分钟流表；也可用DDB_INDEX_BAR_TABLE")
    p.add_argument("--market-bar-table", help="DolphinDB期权/期货已收盘分钟流表；也可用DDB_OPTION_BAR_TABLE")
    p.add_argument("--index-bar-timestamp", choices=("start", "end"), help="DFS默认沿用配置，流默认end")
    p.add_argument("--market-bar-timestamp", choices=("start", "end"), default="end")
    p.add_argument("--ctp-price-type", choices=("last", "mid"), default="last",
        help="last成交Bar；mid报价Bar，零成交量，模型不代表盘口可成交性")
    p.add_argument("--frame-wait-seconds", type=float, default=2)
    p.add_argument("--instrument-max-age-seconds", type=float, default=30)
    p.add_argument("--ready-timeout", type=float, default=90)
    p.add_argument("--seconds", type=float, default=900)
    p.add_argument("--nav", type=Decimal, default=Decimal("40000000"))
    p.add_argument("--vega-budget", type=float, default=1.25)
    p.add_argument("--delta-targets", default="0.2,0.25,0.3,0.35")
    p.add_argument("--option-slot", default="13:58")
    p.add_argument("--hedge-slots", default="10:00,11:00,13:30,14:00,14:55")
    p.add_argument("--flatten-slot", default="14:50")
    p.add_argument("--flatten-day", type=lambda value: datetime.strptime(value, "%Y%m%d").date(),
        help="可选策略主动清仓日；省略则停机保留策略持仓")
    p.add_argument("--min-remaining-days", type=int, default=5)
    p.add_argument("--close-remaining-days", type=int, default=2)
    p.add_argument("--rate", type=float, default=.02)
    p.add_argument("--min-time-value", type=float, default=5)
    p.add_argument("--max-market-age-seconds", type=int, default=300)
    p.add_argument("--max-iv-age-seconds", type=int, default=300)
    p.add_argument("--max-option-lots", type=int, default=1)
    p.add_argument("--max-total-option-lots", type=int, default=2)
    p.add_argument("--max-future-lots", type=int, default=1)
    p.add_argument("--max-notional", type=Decimal, default=Decimal("2000000"))
    p.add_argument("--limit-offset-ticks", type=int, default=1)
    p.add_argument("--max-session-orders", type=int, default=20)
    p.add_argument("--query-timeout", type=float, default=30)
    p.add_argument("--account-reconciliation", choices=("strict", "observe", "off"))
    p.add_argument("--strategy-position-source", choices=("fills", "model"))
    p.add_argument("--enable-orders", action="store_true")
    p.add_argument("--confirm-simnow", action="store_true")
    p.add_argument("--state-file", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--require-fills", action="store_true", help="必须有本次归属成交才通过验收")
    p.add_argument("--report-dir", type=Path, default=Path(__file__).with_name("results"))
    args = p.parse_args(argv)
    if not args.connect:
        p.error("须显式--connect")
    try:
        from bomber.framework.calendars import NativeCalendarProvider
        args.calendar_id = NativeCalendarProvider(args.calendar_id).calendar_id
        for name in ("expected_trading_day", "replay_md_trading_day", "valuation_day"):
            value = getattr(args, name)
            if value and datetime.strptime(value, "%Y%m%d").strftime("%Y%m%d") != value:
                raise ValueError("日期须为YYYYMMDD")
        if (args.simnow_environment == "replay") != bool(args.replay_md_trading_day):
            raise ValueError("replay须声明原始MD交易日；realtime不接受replay-md-trading-day")
        args.valuation_day = args.valuation_day or args.replay_md_trading_day or args.expected_trading_day
        if args.valuation_day != (args.replay_md_trading_day or args.expected_trading_day):
            raise ValueError("估值日须与选定行情环境日期一致，不能改写历史行情日期")
        if args.replay_test_source_time and not args.replay_test_start_time:
            raise ValueError("replay-test-source-time须同时声明replay-test-start-time")
        if args.replay_test_start_time:
            from .replay_test import parse_minute
            parse_minute(args.replay_test_start_time)
            if args.replay_test_source_time:
                parse_minute(args.replay_test_source_time)
            if args.simnow_environment != "replay" or args.market_source != "ctp" or args.index_source != "dolphindb-dfs":
                raise ValueError("08工程时间映射须使用CTP Replay和DFS历史指数")
            if args.resume:
                raise ValueError("08工程时间映射只支持新测试会话；恢复使用原始时间路径")
        for name in ("option_product", "future_product"):
            setattr(args, name, getattr(args, name).strip().upper())
            if not getattr(args, name).isalpha():
                raise ValueError("品种代码须为字母")
        if len(args.index_code) != 6 or not args.index_code.isdigit():
            raise ValueError("指数代码须为六位数字")
        if not args.index_dataset or args.index_schema_version < 1:
            raise ValueError("指数须绑定明确的数据集及正Schema版本")
        if not math.isfinite(args.index_poll_seconds) or not 0 < args.index_poll_seconds <= 30:
            raise ValueError("指数查询间隔须在(0,30]秒")
        if not 1 <= args.index_query_timeout <= 30 or any(not 0 <= value <= 9 for value in
                (args.index_price_precision, args.index_size_precision)):
            raise ValueError("指数查询超时须在[1,30]秒，精度须在[0,9]")
        args.config = OptionVegaConfig(nav=args.nav, vega_budget=args.vega_budget,
            delta_targets=tuple(float(value) for value in args.delta_targets.split(",")),
            option_slot=args.option_slot, hedge_slots=tuple(args.hedge_slots.split(",")),
            flatten_slot=args.flatten_slot, min_remaining_days=args.min_remaining_days,
            close_remaining_days=args.close_remaining_days, rate=args.rate, min_time_value=args.min_time_value,
            max_market_age_seconds=args.max_market_age_seconds, max_iv_age_seconds=args.max_iv_age_seconds,
            max_option_lots=args.max_option_lots, max_total_option_lots=args.max_total_option_lots,
            max_future_lots=args.max_future_lots)
        for value in (args.seconds, args.query_timeout, args.ready_timeout, args.instrument_max_age_seconds,
                      args.reference_max_observation_seconds, args.reference_timeout):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("时长和超时须为有限正数")
        if not math.isfinite(args.frame_wait_seconds) or not 0 <= args.frame_wait_seconds <= min(30, args.max_market_age_seconds):
            raise ValueError("frame-wait-seconds须在[0,30]")
        if not math.isfinite(args.reference_refresh_seconds) or args.reference_refresh_seconds < 0:
            raise ValueError("参考刷新间隔无效")
        if args.instrument_max_age_seconds > 120 or args.limit_offset_ticks < 0 or args.max_session_orders < 1:
            raise ValueError("逐合约时效最多120秒，跳价须非负，订单上限须为正")
        if not args.max_notional.is_finite() or args.max_notional <= 0:
            raise ValueError("名义金额上限须为有限正数")
        if args.reference_source == "file" and (not args.allow_file_reference_test or not args.opt_basic or not args.fut_basic):
            raise ValueError("文件资料须显式测试开关及opt-basic、fut-basic路径")
        if args.mode == "recording":
            if args.account_reconciliation not in (None, "off"):
                raise ValueError("08 Recording只接收行情，不关联物理账户")
            args.account_reconciliation = "off"
            if args.enable_orders or args.confirm_simnow or args.state_file or args.resume or args.require_fills:
                raise ValueError("Recording不接受报单、恢复或成交验收参数")
        else:
            if args.state_file is None or args.resume != args.state_file.is_file():
                raise ValueError("新会话须独立新state-file；恢复须--resume及已有原文件")
            if args.state_file.exists() and not args.state_file.is_file():
                raise ValueError("state-file不是普通文件")
            if args.mode == "simnow" and not (args.enable_orders and args.confirm_simnow):
                raise ValueError("SimNow报单须enable-orders和confirm-simnow")
            if args.mode == "model" and (args.enable_orders or args.confirm_simnow):
                raise ValueError("Model不接受物理账户报单授权")
        # 此模块不加载原生交易扩展，帮助与参数错误可独立检查。
        from bomber.framework.trader.execution.policy import execution_policy_from_args
        args.execution_policy = execution_policy_from_args(args)
    except (ValueError, TypeError) as error:
        p.error(str(error))
    return args


def _dolphin_feed(table, source_id, *, max_age_seconds):
    from bomber.framework.market.stream.dolphin import DolphinDbLiveDataFeed, DolphinDbConfig, DolphinDbStreamSpec
    from bomber.framework.market.stream.health import StreamHealthConfig
    from bomber.framework.trader.runtime.live.channels.ctp import required
    if not table:
        raise ValueError(f"须配置{source_id}的已收盘分钟流表，禁止猜测表名")
    return DolphinDbLiveDataFeed(DolphinDbConfig(host=required("DDB_HOST"),
        port=int(os.getenv("DDB_PORT", "8848")), username=required("DDB_USERNAME"), password=required("DDB_PASSWORD"),
        streaming_port=int(os.getenv("DDB_STREAMING_PORT", "0")),
        streams=(DolphinDbStreamSpec.bar(table, action_name=f"{source_id}-{os.getpid()}-{time.time_ns()}"),)),
        source_id=source_id, health_config=StreamHealthConfig(stale_after_seconds=max_age_seconds))


def prepare_inputs(args, context, lifecycle):
    from bomber.framework.calendars.trading import NativeCalendarProvider, RULES_END
    from bomber.framework.dataprep.live_options import LiveOptionReferences
    from bomber.framework.dataprep.sources import ReferenceSourceFactory, DolphinDbReferenceConfig
    from bomber.framework.dataprep.sources.base import ReferenceDataset
    from bomber.framework.market.basic.base import InstrumentId, InstrumentMeta
    from bomber.framework.market.stream.ctp import CtpLiveDataFeed, CtpMdConfig
    from bomber.framework.market.stream.aggregation import TradeTickBarFeed, QuoteMidBarFeed
    from bomber.framework.trader.runtime.live.channels.ctp import build_md_config
    from .live_runtime import OptionReferences
    from .vega_signal import remaining_trading_days
    valuation_day = datetime.strptime(args.valuation_day, "%Y%m%d").date()
    calendar_snapshot = NativeCalendarProvider(args.calendar_id).snapshot(valuation_day, RULES_END)
    calendar = calendar_snapshot.days
    if valuation_day not in calendar:
        raise ValueError("估值日不在交易日历内")
    config = (DolphinDbReferenceConfig.from_env(database=args.reference_database,
        read_timeout_seconds=args.reference_timeout) if args.reference_source == "dolphindb" else
        {ReferenceDataset.OPTIONS_BASIC: args.opt_basic, ReferenceDataset.FUTURES_BASIC: args.fut_basic})
    source = ReferenceSourceFactory.create(args.reference_source, config)
    context.resources.callback(source.close)
    source.open()
    service = LiveOptionReferences(source, option_product=args.option_product, future_product=args.future_product,
        index_code=args.index_code, valuation_day=valuation_day, refresh_seconds=args.reference_refresh_seconds,
        max_observation_seconds=args.reference_max_observation_seconds, max_refresh_seconds=args.reference_timeout)
    refs = context.references = OptionReferences(service, context.trading_day, calendar,
        calendar_manifest=calendar_snapshot.manifest)
    if args.flatten_day is not None and (args.flatten_day < valuation_day or args.flatten_day not in calendar):
        raise ValueError("清仓日须在估值日之后或当天，且属于交易日历")
    if args.mode == "simnow":
        execution_day = datetime.strptime(context.trading_day, "%Y%m%d").date()
        invalid = [key for key, info in refs.contracts.items() if not info.list_day <= execution_day <= info.last_day]
        invalid.extend(key for key, row in refs.futures.items()
            if not row["listDate"] <= execution_day <= row["lastTradeDate"])
        if invalid:
            raise ValueError(f"回放合约在实际TD日不可交易，不能将历史期权发往柜台: {invalid}")
    for info in refs.contracts.values():
        remaining_trading_days(calendar, valuation_day, info.last_day, args.min_remaining_days)
    index_id = InstrumentId.from_str(args.index_code + ".XSHG")
    index_query_feed = None
    if args.index_source == "dolphindb-dfs":
        from bomber.framework.dataprep.features import load_feature_bundle, FeatureProviderFactory, FeatureQueryCache
        from bomber.framework.dataprep.sources.policy import DataSourcePurpose
        from bomber.framework.market.stream.feature_query import FeatureQueryBarFeed
        bundle = load_feature_bundle(args.index_feature_config, DolphinDbReferenceConfig.from_env(
            read_timeout_seconds=args.index_query_timeout))
        binding = bundle.binding(args.index_dataset, args.index_schema_version)
        if binding.spec.frequency.period_ns != 60_000_000_000 or binding.bar_projection is None:
            raise ValueError("08策略指数输入须声明一分钟真实OHLCV投影")
        raw_config = binding.source_config
        overrides = {}
        for name in ("database", "table", "availability"):
            value = getattr(args, "index_"+name)
            if value is not None:
                overrides[name] = "source-time" if value == "update-time" else value
        if args.index_bar_timestamp is not None:
            overrides["event_time"] = replace(raw_config.event_time, label=args.index_bar_timestamp)
        raw_config = replace(raw_config, **overrides)
        provider = FeatureProviderFactory.create_for(DataSourcePurpose.LIVE_FEATURE,
            binding.backend, raw_config, bundle.registry)
        context.resources.callback(provider.close)
        start = datetime.combine(valuation_day, datetime.min.time(), ZoneInfo("Asia/Shanghai"))
        cache = FeatureQueryCache(provider, dataset_id=binding.spec.dataset_id,
            schema_version=binding.spec.schema_version, keys=(args.index_code,),
            start_ns=int(start.timestamp())*1_000_000_000,
            end_ns=int((start+timedelta(days=1)).timestamp())*1_000_000_000-1,
            poll_seconds=args.index_poll_seconds, source_id="vega-index-dfs")
        index_query_feed = index = FeatureQueryBarFeed(cache, key=args.index_code,
            projection=binding.bar_projection, replay=args.simnow_environment == "replay",
            max_market_age_seconds=args.max_market_age_seconds)
        index_identity = {"channel": "dolphindb-dfs", "source": raw_config.identity,
            "dataset_id": binding.spec.dataset_id, "schema_version": binding.spec.schema_version,
            "schema_fingerprint": binding.spec.fingerprint,
            "database": raw_config.database, "table": raw_config.table,
            "availability": raw_config.availability, "timestamp_label": raw_config.timestamp_label,
            "poll_seconds": args.index_poll_seconds, "query_timeout": args.index_query_timeout,
            "price_precision": args.index_price_precision, "size_precision": args.index_size_precision}
    else:
        index = _dolphin_feed(args.index_bar_table or os.getenv("DDB_INDEX_BAR_TABLE"), "vega-index",
            max_age_seconds=args.max_market_age_seconds)
        index_identity = {"channel": "dolphindb", "table": index.config.streams[0].table_name,
            "host": index.config.host, "port": index.config.port}
    index.register_instrument(InstrumentMeta(index_id, price_precision=args.index_price_precision,
        size_precision=args.index_size_precision if index_query_feed else 0,
        price_increment=Decimal(1).scaleb(-args.index_price_precision), exchange="XSHG"))
    ctp_upstream = ctp_bar_feed = None
    dolphin_feeds = [] if index_query_feed else [index]
    feeds = {"index-bars": index}
    labels = {"index-bars": "end" if index_query_feed else args.index_bar_timestamp or "end"}
    identity = {"index": index_identity}
    if args.market_source == "ctp":
        ctp_upstream = CtpLiveDataFeed(build_md_config(CtpMdConfig, lifecycle, flow_path=f"/tmp/bomber-vega-md-{os.getpid()}"))
        cls = TradeTickBarFeed if args.ctp_price_type == "last" else QuoteMidBarFeed
        if args.replay_test_start_time:
            from .replay_test import ReplayTestTimeMap, mapped_bar_feed
            time_map = ReplayTestTimeMap(valuation_day, args.replay_test_start_time, args.replay_test_source_time)
            ctp_bar_feed = market = mapped_bar_feed("VEGA_CTP_1M", ctp_upstream,
                price_type=args.ctp_price_type, time_map=time_map)
        else:
            ctp_bar_feed = market = cls("VEGA_CTP_1M", ctp_upstream)
        labels["market-bars"] = "inclusive-end"
        identity["market"] = {"channel": "ctp", "front": lifecycle.md_front, "price": args.ctp_price_type}
        if args.replay_test_start_time:
            identity["market"]["test_time_mapping"] = time_map.identity
    else:
        market = _dolphin_feed(args.market_bar_table or os.getenv("DDB_OPTION_BAR_TABLE"), "vega-market",
            max_age_seconds=args.max_market_age_seconds)
        dolphin_feeds.append(market)
        labels["market-bars"] = args.market_bar_timestamp
        identity["market"] = {"channel": "dolphindb", "table": market.config.streams[0].table_name,
            "host": market.config.host, "port": market.config.port}
    for key, instrument in refs.instrument_ids.items():
        info = refs.contracts.get(key)
        tick = info.tick if info else refs.futures[key]["minChgPriceNum"]
        multiplier = info.multiplier if info else refs.futures[key]["contMultNum"]
        market.register_instrument(InstrumentMeta(instrument, price_precision=max(0, -tick.normalize().as_tuple().exponent),
            price_increment=tick, multiplier=multiplier, exchange="CFFEX"))
    feeds["market-bars"] = market
    return SimpleNamespace(references=refs, feeds=feeds, labels=labels, index_id=index_id,
        ctp_upstream=ctp_upstream, ctp_bar_feed=ctp_bar_feed, dolphin_feeds=dolphin_feeds,
        feature_query_feeds=(index_query_feed,) if index_query_feed else (),
        sources={**{key: "market-bars" for key in refs.instrument_ids}, args.index_code: "index-bars"}, source_identity=identity)


def describe(session):
    if session is None:
        return {}
    from bomber.framework.trader.execution.policy import observed_account_positions
    s, p = session.strategy, session.runner.position_manager
    ids = session.references.instrument_ids
    time_map = getattr(getattr(session, "bar_feed", None), "time_map", None)
    return {"frames": s.frames, "entry_decisions": s.entry_decisions, "submissions": s.submissions,
        "recorded_requests": [{"revision": request.revision, "ts_event": request.ts_event,
            "signal_ns": request.metadata.get("signal_ts"),
            "targets": {str(key): str(value) for key, value in request.targets.items()}}
            for request in getattr(session.client, "requests", ())] if session.mode == "recording" else [],
        "entry_audit": s.entry_audit, "frame_audit": session.runner.frame_audit,
        "bar_time_audit": session.runner.bar_time_audit,
        "ignored_stale_candidate_bars": session.runner.ignored_stale_candidate_bars,
        "calendar_manifest": session.references.calendar_manifest,
        "market_sources": getattr(session, "source_identity", {}),
        "replay_test_time_mapping": None if time_map is None else time_map.snapshot,
        "validation_scope": "engineering_chain" if time_map is not None else "source_time",
        "order_submission_audit": getattr(getattr(session.execution, "backend", None), "snapshot", []),
        "option_slot": s.config.option_slot,
        "fills_received": s.fills_received, "order_updates_received": s.order_updates_received,
        "option_goals": s.option_goals, "future_goals": s.future_goals,
        "strategy_positions": {key: p.position(s.strategy_id, str(i)) for key, i in ids.items()},
        "account_positions": observed_account_positions(session.client, ids.values(), positions=p),
        "execution_positions": {key: p.account_position(session.client.client_id, i) for key, i in ids.items()},
        "late_bar_count": session.runner.frames.late_count, "signals": s.signals[-30:],
        "feature_queries": [feed.manifest for feed in session.runner.feature_query_feeds],
        "skipped_index_frames": session.runner.skipped_feature_frames,
        "account_dispatch": getattr(session.client, "dispatch_status", {}),
        "execution_origin": {"model": "model", "simnow": "ctp_native", "recording": "recording"}[session.mode]}


def build_runtime(args, *, account_routes=()):
    from bomber.framework.trader.runtime.live.runtime import ManagedLiveRuntime
    from bomber.framework.trader.runtime.reports import LiveRunReport
    from bomber.framework.trader.runtime.live.channels.ctp import required, build_simnow_transport
    from bomber.framework.trader.runtime.live.channels.ctp_market import CtpMarketCredentials
    from .recovery import OptionLocalLifecycle, OptionCtpLifecycle
    from .live_runtime import CLIENT_ID, assemble
    if args.mode == "simnow":
        from bomber.framework.trader.execution.ctp import CtpTdApiTransport
        transport = build_simnow_transport(CtpTdApiTransport, client_id=CLIENT_ID,
            flow_path=f"/tmp/bomber-vega-td-{os.getpid()}", timeout_seconds=args.query_timeout)
        lifecycle = OptionCtpLifecycle(transport, md_front=required("CTP_MD_ADDRESS") if args.market_source == "ctp" else None,
            orders=True, max_session_orders=args.max_session_orders, expected_trading_day=args.expected_trading_day,
            environment=args.simnow_environment if args.market_source == "ctp" else "realtime",
            replay_md_trading_day=args.replay_md_trading_day if args.market_source == "ctp" else None,
            scope_to_reference_instruments=True, account_reconciliation=args.execution_policy.account_reconciliation,
            resume=args.resume, ready_seconds=args.ready_timeout, valuation_day=args.valuation_day,
            state_file=args.state_file, require_fills=args.require_fills)
    else:
        lifecycle = OptionLocalLifecycle(args.expected_trading_day, mode=args.mode, valuation_day=args.valuation_day,
            resume=args.resume, state_file=args.state_file, ready_seconds=args.ready_timeout, require_fills=args.require_fills)
    if args.market_source == "ctp":
        lifecycle.md_front = required("CTP_MD_ADDRESS")
        lifecycle.md_credentials = CtpMarketCredentials(os.getenv("CTP_BROKER_ID", "9999"), required("CTP_ACCOUNT_ID"),
            required("CTP_PASSWORD"), os.getenv("CTP_PRODUCTION_MODE", "true").lower() in {"1", "true", "yes", "on"})
    report = LiveRunReport(args.report_dir, prefix=f"option-vega-{args.mode}", describe=describe,
        metadata={"mode": args.mode, "market_source": args.market_source, "index_source": args.index_source,
            "index_code": args.index_code, "calendar_id": args.calendar_id,
            "simnow_environment": args.simnow_environment, "state_file": str(args.state_file),
            "account_reconciliation": args.execution_policy.account_reconciliation.value,
            "strategy_position_source": args.execution_policy.position_source.value})
    return ManagedLiveRuntime(CLIENT_ID, controller=lifecycle,
        prepare_inputs=lambda context: prepare_inputs(args, context, lifecycle),
        assemble_session=lambda inputs, context: assemble(args, inputs, getattr(lifecycle, "driver", None),
            account_routes=account_routes), report=report, seconds=args.seconds)


def main(argv=None):
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    runtime = build_runtime(args)
    try:
        runtime.run()
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
