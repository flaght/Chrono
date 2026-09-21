"""EMA正式引擎链路的分阶段验证。

N5只使用内存Bar，验证桥、正式Runtime、订单和成交；N6/N7再分别加入真实CTP
Bar/Tick文件。全部使用BacktestEngine模拟撮合，不连接任何真实交易端。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

from bomber.backtest.config import BacktestEngineConfig
from bomber.model import BarType, Money, Venue
from bomber.model.currencies import CNY
from bomber.model.enums import AccountType, OmsType
from bomber.model.identifiers import InstrumentId, TraderId

from examples.single_ema.ema_ctp_backtest import _future, run_case
from examples.single_ema.strategies import EmaCrossConfig, EmaCrossTargetStrategy
from market.basic.base import InstrumentMeta, make_bar
from strategy import BacktestRuntimePort
from strategy.bridge import NautilusStrategyBridge, NautilusStrategyBridgeConfig
from strategy.runtime import NautilusBacktestRuntime


SYNTHETIC_ID = InstrumentId.from_str("rb9999.SHFE")


def test5_bridge_and_backtest_runtime() -> None:
    """N5：用内存Bar隔离验证统一策略已真正进入原生撮合引擎。"""

    meta = InstrumentMeta(
        instrument_id=SYNTHETIC_ID,
        price_precision=0,
        size_precision=0,
        price_increment=Decimal(1),
        multiplier=Decimal(10),
        currency="CNY",
        exchange="SHFE",
    )
    prices = [100, 101, 103, 105, 102, 99, 97, 100, 104, 106]
    bars = [
        make_bar(
            instrument_id=SYNTHETIC_ID,
            open=price - 1,
            high=price + 1,
            low=price - 2,
            close=price,
            volume=100,
            ts_event=1_800_000_000_000_000_000 + index * 60_000_000_000,
            meta=meta,
            bar_type="1-MINUTE",
        )
        for index, price in enumerate(prices)
    ]
    runtime = NautilusBacktestRuntime(
        "n5-synthetic",
        BacktestEngineConfig(trader_id=TraderId("N5-TESTER"), run_analysis=False),
    )
    assert isinstance(runtime, BacktestRuntimePort)
    runtime.add_venue(
        venue=Venue("SHFE"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=CNY,
        starting_balances=[Money(1_000_000, CNY)],
    )
    runtime.add_instrument(_future(SYNTHETIC_ID, "2030-01-01"))
    runtime.add_data(bars)
    target_strategy = EmaCrossTargetStrategy(
        "n5-ema",
        EmaCrossConfig(fast_period=2, slow_period=3),
    )
    bridge = NautilusStrategyBridge(
        NautilusStrategyBridgeConfig(
            instrument_id=SYNTHETIC_ID,
            bar_type=BarType.from_str(f"{SYNTHETIC_ID}-1-MINUTE-LAST-EXTERNAL"),
        ),
        target_strategy,
    )
    runtime.add_strategy(bridge)
    try:
        result = runtime.run()
        orders = runtime.engine.trader.generate_orders_report()
        fills = runtime.engine.trader.generate_fills_report()
        assert bridge.target_events
        assert not orders.empty
        assert not fills.empty
        assert result.total_orders == len(orders)
        print(
            f"N5通过：targets={len(bridge.target_events)} "
            f"orders={len(orders)} fills={len(fills)}",
        )
    finally:
        runtime.stop()


def test6_ctp_bar_backtest() -> None:
    """N6：真实rb2704 Feather Bar进入正式撮合链路。"""

    run_case("bar")
    print("N6通过：CTP Feather Bar正式回测")


def test7_ctp_tick_backtest() -> None:
    """N7：真实rb2609 Tick经内部聚合Bar后进入正式撮合链路。"""

    run_case("tick")
    print("N7通过：CTP Tick内部聚合正式回测")


STAGES = {
    5: test5_bridge_and_backtest_runtime,
    6: test6_ctp_bar_backtest,
    7: test7_ctp_tick_backtest,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="正式策略运行时分阶段验证")
    parser.add_argument(
        "--stage",
        choices=(*(str(stage) for stage in STAGES), "all"),
        default="all",
    )
    args = parser.parse_args()
    if args.stage == "all":
        # Bomber/Nautilus的Rust日志器是进程级单例。一个测试函数释放
        # BacktestEngine后，在同一Python进程创建第二个引擎会因重复初始化
        # 全局logger而直接panic，无法由Python异常捕获。因此all模式只负责编排，
        # 每个阶段放入独立子进程；单阶段模式仍便于直接调试。
        script = str(Path(__file__).resolve())
        for stage in STAGES:
            print(f"\n=== 启动独立进程验证 N{stage} ===", flush=True)
            subprocess.run(
                [sys.executable, script, "--stage", str(stage)],
                check=True,
            )
        print("Formal strategy tests OK")
        return

    stage = int(args.stage)
    print(f"\n--- N{stage} ---")
    STAGES[stage]()
    print("Formal strategy tests OK")


if __name__ == "__main__":
    main()
