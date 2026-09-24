"""第六类 CTP 期货正式历史回测：文件计划 + 独立时钟 + 原生模拟 Backend。"""

from __future__ import annotations

import argparse
from pathlib import Path

from bomber.backtest.config import BacktestEngineConfig
from bomber.model.identifiers import TraderId

from examples.cross_section.run_bar_backtest import CTP_DATA_DIR, create_instrument
from examples.scheduled_targets.local_input import load_target_csv
from examples.scheduled_targets.execution_audit import audit_schedule_execution, format_audit
from examples.scheduled_targets.scheduled_strategy import ScheduledTargetStrategy
from market.basic.base import DataType
from market.replay.parsers.bar import BarColumns, MappedBarParser
from trader import (
    ExecutionRoute, MarketStreamBinding, NautilusMarketFeedAdapter,
    NautilusSimExecutionBackend, NetTargetOrderPlanner, PositionManager,
    RuntimeMode, SimulationExecutionClient, UnifiedHistoricalRuntime,
    UnifiedStrategyRunner,
)
from trader.scheduling import TimedFileReplayFeed


DEFAULT_TARGETS = Path(__file__).with_name("positions_20260728.csv")
SYMBOLS = {"rb": ("rb2704", "rb2704_20260728.feather"),
           "sa": ("SA703", "SA703_20260728.feather")}


def run_case(*, bars_dir: Path = CTP_DATA_DIR, targets_csv: Path = DEFAULT_TARGETS) -> None:
    schedule = load_target_csv(targets_csv)
    feed = TimedFileReplayFeed(schedule.slots, "SCHEDULED_CTP_BAR_REPLAY")
    parser = MappedBarParser(
        columns=BarColumns(symbol="symbol", exchange="exchange", timestamp="datetime",
                           open="open", high="high", low="low", close="close",
                           volume="volume", value="value", open_interest="open_interest",
                           vwap="vwap"),
        bar_spec="1-MINUTE", timezone="Asia/Shanghai",
        exchange_aliases={"XSGE": "SHFE", "XZCE": "CZCE", "XDCE": "DCE"},
    )
    backend = NautilusSimExecutionBackend(
        "scheduled-ctp-sim",
        BacktestEngineConfig(trader_id=TraderId("SCHEDULED-CTP-001"), run_analysis=True),
    )
    instruments = {}
    venues = set()
    for key, (symbol, filename) in SYMBOLS.items():
        path = bars_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        profile, instrument, meta = create_instrument(symbol)
        if str(profile.venue) not in venues:
            backend.add_profile(profile)
            venues.add(str(profile.venue))
        backend.add_instrument(instrument)
        feed.register_instrument(meta)
        feed.add_bar_feather(path, parser)
        instruments[key] = instrument.id
    unknown = {key for plan in schedule.plans for key in plan.targets} - set(instruments)
    if unknown:
        raise ValueError(f"目标文件含未配置的真实合约目标: {sorted(unknown)}")

    positions = PositionManager()
    client = SimulationExecutionClient(
        backend.backend_id, NetTargetOrderPlanner(positions), backend, positions,
    )
    market_adapter = NautilusMarketFeedAdapter(
        "scheduled-ctp-market", feed, backend,
        tuple(MarketStreamBinding(item, DataType.BAR, "1-MINUTE")
              for item in instruments.values()), manage_lifecycle=False,
    )
    strategy = ScheduledTargetStrategy("scheduled-ctp-targets", schedule)
    runner = UnifiedStrategyRunner(RuntimeMode.HISTORICAL, position_manager=positions)
    runner.add_data_feed("scheduled-bars", feed)
    runner.add_execution_client(client)
    runner.add_strategy(
        strategy, data_bindings=(), time_feed_ids=("scheduled-bars",),
        execution_routes=tuple(ExecutionRoute(key, backend.backend_id, instrument_id)
                               for key, instrument_id in instruments.items()),
    )
    runtime = UnifiedHistoricalRuntime("scheduled-ctp-runtime", runner, market_adapter)
    try:
        result = runtime.run()
        orders = backend.engine.trader.generate_orders_report()
        fills = backend.engine.trader.generate_fills_report()
        print(f"S5正式基础回测: bars={result.replay_summary.bars} "
              f"slots={len(schedule.slots)} submitted={sum(x.status == 'SUBMITTED' for x in strategy.audit)} "
              f"orders={len(orders)} fills={len(fills)}")
        print(f"S5调度审计: {strategy.audit}")
        if any(item.status != "SUBMITTED" for item in strategy.audit):
            raise AssertionError("目标时点未全部触发；检查样本时间范围和时区")
        if orders.empty or fills.empty:
            raise AssertionError("已触发目标但没有正式模拟成交")
        if client.report_errors:
            raise AssertionError(f"执行回报异常: {client.report_errors}")
        account_positions = {
            str(instrument): positions.account_position(backend.backend_id, instrument)
            for instrument in instruments.values()
        }
        working_positions = {
            str(instrument): positions.working_quantity(backend.backend_id, instrument)
            for instrument in instruments.values()
        }
        native_positions = {
            str(instrument): backend.engine.portfolio.net_position(instrument)
            for instrument in instruments.values()
        }
        audit = audit_schedule_execution(
            schedule, backend.reports, instruments,
            actual_positions=account_positions,
            working_positions=working_positions,
            native_positions=native_positions,
        )
        if len(audit.fills) != len(fills):
            raise AssertionError(f"统一回报与原生成交数不一致: {len(audit.fills)} != {len(fills)}")
        expected_fee = sum((item.quantity for item in audit.fills), start=0)
        if audit.commission_by_currency != {"CNY": expected_fee}:
            raise AssertionError(
                f"手续费与示例Profile的1 CNY/手不一致: "
                f"actual={dict(audit.commission_by_currency)} expected={expected_fee} CNY",
            )
        for row in format_audit(audit):
            print(f"S6逐笔成交: {row}")
        print(f"S6执行审计通过: fills={len(audit.fills)} "
              f"fees={dict(audit.commission_by_currency)} "
              f"final_targets={dict(audit.final_targets)} "
              f"account_positions={account_positions} native_positions={native_positions}")
    finally:
        runtime.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bars-dir", type=Path, default=CTP_DATA_DIR)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    args = parser.parse_args()
    run_case(bars_dir=args.bars_dir, targets_csv=args.targets)


if __name__ == "__main__":
    main()
