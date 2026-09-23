#!/usr/bin/env python3
"""CTP实时Tick→已收盘分钟Bar→同一EMA策略；当前仅Recording，不下单。"""

from __future__ import annotations

import argparse
import os
import time
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

from examples.single_ema.strategies import EmaCrossConfig, EmaCrossTargetStrategy
from market.basic.base import DataType, InstrumentId, InstrumentMeta
from market.stream import TradeTickBarFeed
from market.stream.ctp import CtpLiveDataFeed, CtpMdConfig
from strategy import (
    DataBinding, ExecutionRoute, RecordingExecutionClient, RuntimeMode,
    UnifiedStrategyRunner,
)


load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"缺少环境变量: {name}")
    return value


def build_runner(args):
    """只替换Feed/Client装配，不改变EMA策略类。"""
    instrument_id = InstrumentId.from_str(f"{args.symbol}.{args.exchange}")
    upstream = CtpLiveDataFeed(CtpMdConfig(
        front=_required("CTP_MD_ADDRESS"),
        broker_id=_required("CTP_BROKER_ID"),
        user_id=_required("CTP_ACCOUNT_ID"),
        password=_required("CTP_PASSWORD"),
        flow_path=os.getenv("CTP_MD_FLOW_PATH", "/tmp/bomber-ctp-ema"),
        production_mode=os.getenv("CTP_PRODUCTION_MODE", "").lower()
        in {"1", "true", "yes", "on"},
    ))
    feed = TradeTickBarFeed("CTP_EMA_1M", upstream)
    feed.register_instrument(InstrumentMeta(
        instrument_id=instrument_id,
        price_precision=args.price_precision,
        size_precision=0,
        price_increment=Decimal(args.price_increment),
        multiplier=Decimal(args.multiplier),
        currency="CNY",
        exchange=args.exchange,
    ))
    strategy = EmaCrossTargetStrategy(
        "ema-ctp-live", EmaCrossConfig(
            fast_period=args.fast, slow_period=args.slow,
            long_quantity=Decimal(args.quantity),
            short_quantity=-Decimal(args.quantity),
            skip_single_price=False,
        ),
    )
    client = RecordingExecutionClient("recording-only")
    runner = UnifiedStrategyRunner(RuntimeMode.LIVE)
    runner.add_data_feed("ctp-bars", feed)
    runner.add_execution_client(client)
    runner.add_strategy(
        strategy,
        data_bindings=(DataBinding(
            "primary_bar", "ctp-bars", instrument_id, DataType.BAR, "1-MINUTE",
        ),),
        execution_routes=(ExecutionRoute(
            "position", client.client_id, instrument_id,
        ),),
    )
    return runner, strategy, client


def main() -> None:
    parser = argparse.ArgumentParser(description="CTP在线EMA Recording示例（不下单）")
    parser.add_argument("--symbol", default=os.getenv("CTP_SYMBOL"))
    parser.add_argument("--exchange", default=os.getenv("CTP_EXCHANGE", "SHFE"))
    parser.add_argument("--fast", type=int, default=2)
    parser.add_argument("--slow", type=int, default=3)
    parser.add_argument("--quantity", default="1")
    parser.add_argument("--price-precision", type=int, default=0)
    parser.add_argument("--price-increment", default="1")
    parser.add_argument("--multiplier", default="10")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    if not args.symbol:
        parser.error("必须通过--symbol或CTP_SYMBOL指定合约")
    if args.timeout <= 0:
        parser.error("--timeout必须大于0")
    args.symbol = args.symbol.strip()
    args.exchange = args.exchange.strip().upper()
    runner, strategy, client = build_runner(args)
    print(f"启动CTP在线EMA：{args.symbol}.{args.exchange}，仅Recording，不下单")
    runner.start()
    reported = 0
    try:
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            requests = client.requests
            for request in requests[reported:]:
                print(f"EMA目标请求: {request}")
            reported = len(requests)
            time.sleep(min(0.5, max(0, deadline - time.monotonic())))
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop()
    print(f"已停止：bars_seen={strategy.bars_seen} targets={len(client.requests)}")


if __name__ == "__main__":
    main()
