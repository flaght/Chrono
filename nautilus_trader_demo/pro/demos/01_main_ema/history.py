"""EMA历史组装与预热；使用公共Provider工厂，策略本身不读取数据库。"""
import json
import os
from contextlib import nullcontext
from pathlib import Path

from bomber.framework.dataprep.history import (
    HistoryService, DolphinDbHistoryConfig, HistoryError, HistoryBar, ReferenceFactorAdjuster)
from bomber.framework.dataprep.sources import DolphinDbReferenceConfig


def build_history(args):
    path = getattr(args, "history_config", None) or os.getenv("EMA_HISTORY_CONFIG")
    config = None
    adjuster = None
    if path:
        data = json.loads(Path(path).read_text())
        if args.history_db == "dolphindb":
            allowed = {"database", "table", "columns", "time_unit", "time_label", "instrument_format", "venue",
                "date_column", "time_column", "source_timezone", "factor_source_days", "factor_database"}
            if not data.keys() <= allowed or not {"database", "table", "columns"} <= data.keys():
                raise HistoryError("DolphinDB行情配置须声明database/table/columns及可选时间口径")
            source_days = data.pop("factor_source_days", None)
            factor_database = data.pop("factor_database", None)
            config = DolphinDbHistoryConfig(
                connection=DolphinDbReferenceConfig.from_env(read_timeout_seconds=args.reference_timeout), **data)
            if source_days is not None:
                from datetime import date
                adjuster = ReferenceFactorAdjuster(
                    DolphinDbReferenceConfig.from_env(database=factor_database, read_timeout_seconds=args.reference_timeout),
                    {date.fromisoformat(day): date.fromisoformat(source) for day, source in source_days.items()})
            elif factor_database is not None:
                raise HistoryError("factor_database须配合显式factor_source_days")
        else:
            config = data  # 新Provider自行校验配置，不更改策略/协调器。
    return HistoryService(database_backend=args.history_db, database_config=config,
        file_path=args.history_file, missing_policy=args.history_missing_policy, price_adjuster=adjuster)


def warm_history(runner, before_ns, *, phase):
    checkpoint = runner.ema_checkpoint
    strategy = checkpoint.strategy
    live_bars = buffered_bars(runner)
    if strategy.bars_used:
        if strategy.last_processed_ns >= before_ns:
            raise HistoryError("指标断点晚于历史恢复边界")
        stamps = tuple(runner.calendar.missing_minutes(strategy.last_processed_ns, before_ns))
    else:
        stamps = runner.calendar.last_minutes(before_ns, runner.history_minutes)
        if not stamps and live_bars:
            first = min(bar.ts_event for bar in live_bars)
            stamps = tuple(runner.calendar.missing_minutes(first - 60_000_000_000, before_ns))
    result = runner.history_service.load(checkpoint.instrument, stamps, phase=phase, live_bars=live_bars)
    prices = []
    # 全部转换成功才推进指标；没有历史复权口径时不拿当前因子补造。
    for bar in result.bars:
        if bar.adjusted_close is None:
            raise HistoryError("EMA历史需要adjusted_close历史复权价；请提供对应字段/因子适配，不能使用当前因子猜造")
        prices.append((bar.ts_event, bar.adjusted_close))
    client = next(iter(runner._clients.values()))
    with runner._submit_lock, getattr(client, "_submit_lock", nullcontext()):
        for stamp, price in prices:
            checkpoint.warm_closed(stamp, price)
    if result.missing:
        print(f"允许历史不足启动：{result.audit()}", flush=True)
    return result


def buffered_bars(runner):
    if not getattr(runner, "history_buffer", ()):
        return ()
    with runner._submit_lock:
        events = tuple(getattr(runner, "history_buffer", ()))
    result = []
    for bar in events:
        if bar.ts_event < runner.first_complete_bar_ns:
            continue
        if str(bar.bar_type.instrument_id) != runner.ema_checkpoint.instrument:
            raise HistoryError("实时缓冲合约变更")
        assignment = runner.references.snapshot(bar.ts_event)
        if assignment.instrument(runner.ema_checkpoint.strategy.config.product, "main").lower() != runner.ema_checkpoint.instrument.split(".")[0].lower():
            raise HistoryError("实时缓冲期间主力角色变更")
        factor = assignment.factor(runner.ema_checkpoint.strategy.config.product, "main")
        result.append(HistoryBar(runner.ema_checkpoint.instrument, bar.ts_event,
            close=bar.close.as_decimal(), adjusted_close=bar.close.as_decimal() * factor))
    return tuple(result)


def flush_buffer(runner):
    live_bars = buffered_bars(runner)
    if live_bars:
        warm_history(runner, max(bar.ts_event for bar in live_bars) + 1, phase="recovery")
    with runner._submit_lock:
        runner.history_buffer = [bar for bar in runner.history_buffer
            if bar.ts_event > runner.ema_checkpoint.strategy.last_processed_ns]
