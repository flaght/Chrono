"""第四类M3b：真实RB分钟Bar → DataHub研究价 → 逻辑目标记录探针。

只验证策略装配，不接模拟撮合和换月执行；必须显式指定测试交易日。
"""

from __future__ import annotations

import argparse
from datetime import date
from decimal import Decimal
from pathlib import Path

from datahub import MinimalDataHub
from examples.role_cross.local_input import load_rb_role_research
from examples.role_cross.role_strategy import RoleCrossTargetStrategy
from market.basic.base import DataType, InstrumentId, InstrumentMeta
from market.replay.base import FileReplayFeed
from market.replay.parsers.bar import FixedInstrumentBarParser
from strategy import (
    DataBinding, ExecutionRoute, RecordingExecutionClient,
    RuntimeMode, UnifiedStrategyRunner,
)


ROLE_ROOT = Path("/workspace/worker/pj/neutron/tests/temp/role")
BARS_ROOT = Path("/workspace/worker/kdwk/Chrono/nautilus_trader_demo/test_data/bars/mo")


def _meta_from_basic(path: Path, symbol: str, price_increment: Decimal) -> InstrumentMeta:
    """身份和乘数取自基础表；缺失的价格步长必须由调用方显式提供。"""
    import pandas as pd

    basic = pd.read_feather(path)
    selected = basic.loc[basic["symbol"].astype(str).str.lower() == symbol.lower()]
    if len(selected) != 1:
        raise ValueError(f"fut_basic需要唯一合约记录: {symbol}, 实际={len(selected)}")
    row = selected.iloc[0]
    exchange = str(row["exchangeCD"]).upper()
    venue = {"XSGE": "SHFE", "SHFE": "SHFE"}.get(exchange)
    if venue is None:
        raise ValueError(f"RB合约交易所未映射: {exchange}")
    increment = Decimal(str(price_increment))
    multiplier = Decimal(str(row["contMultNum"]))
    if not increment.is_finite() or not multiplier.is_finite() or increment <= 0 or multiplier <= 0:
        raise ValueError(f"合约基础表价格步长或乘数无效: {symbol}")
    return InstrumentMeta(
        InstrumentId.from_str(f"{symbol}.{venue}"),
        price_precision=max(0, -increment.normalize().as_tuple().exponent),
        size_precision=0,
        price_increment=increment,
        multiplier=multiplier,
        currency="CNY",
        exchange=venue,
    )


def _bar_timestamp_column(path: Path) -> str:
    import pandas as pd
    import pyarrow as pa

    for name in ("datetime", "timestamp"):
        try:
            pd.read_feather(path, columns=[name])
        except (KeyError, ValueError, pa.ArrowInvalid):
            continue
        return name
    raise ValueError(f"真实合约Bar缺少datetime/timestamp列: {path}")


def run_probe(
    trading_day: date,
    *,
    bars_dir: Path,
    contract_struct: Path,
    factors: Path,
    fut_basic: Path,
    price_increment: Decimal,
) -> None:
    loaded = load_rb_role_research(
        bars_dir=bars_dir,
        contract_struct_path=contract_struct,
        factors_path=factors,
        fut_basic_path=fut_basic,
    )
    ends = dict(loaded.day_end_ns)
    if trading_day not in ends:
        raise ValueError(f"没有可用于研究的交易日: {trading_day}")
    hub = MinimalDataHub(loaded.store)
    contracts = hub.snapshot(ends[trading_day]).contracts
    print(f"M3b角色合约: {dict(contracts)}")
    feed = FileReplayFeed("ROLE_CROSS_RB_BAR_REPLAY")
    instruments = []
    for symbol in dict.fromkeys(contracts.values()):
        files = tuple(bars_dir.rglob(f"{symbol}_{trading_day:%Y%m%d}.feather"))
        if len(files) != 1:
            raise FileNotFoundError(
                f"{trading_day} {symbol}需要唯一真实合约Bar文件，找到{len(files)}个",
            )
        meta = _meta_from_basic(fut_basic, symbol, price_increment)
        feed.register_instrument(meta)
        feed.add_bar_feather(
            files[0], FixedInstrumentBarParser(
                meta.instrument_id, timestamp=_bar_timestamp_column(files[0]),
            ),
        )
        instruments.append(meta.instrument_id)

    client = RecordingExecutionClient("recording-only")
    strategy = RoleCrossTargetStrategy("rb-role-cross", hub)
    runner = UnifiedStrategyRunner(RuntimeMode.HISTORICAL)
    runner.add_data_feed("rb-bars", feed)
    runner.add_execution_client(client)
    runner.add_strategy(
        strategy,
        data_bindings=tuple(
            DataBinding(str(item), "rb-bars", item, DataType.BAR, "1-MINUTE")
            for item in instruments
        ),
        # M3只验策略目标：这里是固定记录端，不代表主力换月已接入。
        execution_routes=(ExecutionRoute("rb_main", client.client_id, instruments[0]),),
    )
    runner.start()
    try:
        summary = runner.run_replay()
        print(
            f"M3b真实Bar探针: day={trading_day} bars={summary.bars} "
            f"complete_frames={strategy.complete_frames} "
            f"signals={len(strategy.signal_events)} requests={len(client.requests)} "
            f"unavailable_events={strategy.unavailable_events}",
        )
        signal = strategy.signal
        print(
            "M3b价差诊断: "
            f"first={signal.first_spread} last={signal.previous_spread} "
            f"min={signal.min_spread} max={signal.max_spread} "
            f"positive={signal.positive_frames} "
            f"negative={signal.negative_frames} zero={signal.zero_frames}",
        )
        if strategy.complete_frames != (
            signal.positive_frames + signal.negative_frames + signal.zero_frames
        ):
            raise AssertionError("完整帧数量与信号处理帧数量不一致")
        if signal.positive_frames and signal.negative_frames and not strategy.signal_events:
            raise AssertionError("价差已出现正负两侧但没有穿越信号")
        if strategy.complete_frames == 0:
            print("尚无四角色同分钟完整帧；请检查四合约Bar覆盖与时间戳，未验收信号。")
        for event in strategy.signal_events[:10]:
            print(
                f"  signal ts={event.ts_event} direction={event.direction} "
                f"main={event.main_instrument} spread={event.spread}",
            )
    finally:
        runner.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="第四类RB真实Bar记录型策略探针")
    parser.add_argument("--day", type=date.fromisoformat, required=True)
    parser.add_argument("--bars-dir", type=Path, default=BARS_ROOT)
    parser.add_argument("--contract-struct", "--fut-contract", type=Path,
                        default=ROLE_ROOT / "fut_contract_data.feather")
    parser.add_argument("--factors", type=Path, default=ROLE_ROOT / "fut_adjustment_factors.feather")
    parser.add_argument("--fut-basic", type=Path, default=ROLE_ROOT / "fut_basic.feather")
    parser.add_argument("--price-increment", type=Decimal, required=True,
                        help="fut_basic缺少最小变动价位；探针必须显式提供RB价格步长")
    args = parser.parse_args()
    run_probe(
        args.day, bars_dir=args.bars_dir, contract_struct=args.contract_struct,
        factors=args.factors, fut_basic=args.fut_basic,
        price_increment=args.price_increment,
    )


if __name__ == "__main__":
    main()
