"""第二类阶段3/4：两种离线存储格式接入同一正式模拟回测。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from examples.cross_section.run_backtest import (
    BINANCE_SYMBOLS,
    CONTRACT_SPECS,
    build_feed,
    create_instrument,
    run_case,
)
from market.basic.base import DataType


def _closes(minute: int) -> tuple[int, ...]:
    # 五个合约价格随分钟变化，用于触发可预测的强弱排名。
    return (100 + 2 * minute, 100 + minute, 100, 200 - 5 * minute, 100 - 2 * minute)


def _make_fixture(root: Path, source: str) -> None:
    start = datetime(2026, 7, 28, 9, 0)
    for index, spec in enumerate(CONTRACT_SPECS):
        folder = root / spec.root
        folder.mkdir()
        if source == "bar":
            try:
                import pyarrow as pa
                import pyarrow.feather as feather
            except ImportError as error:
                raise RuntimeError("Feather阶段需要pyarrow") from error
            rows = []
            for minute in range(27):
                close = _closes(minute)[index]
                rows.append({
                    "datetime": start + timedelta(minutes=minute),
                    "open": float(close), "high": float(close + spec.price_increment),
                    "low": float(close - spec.price_increment), "close": float(close),
                    "volume": 10,
                })
            feather.write_feather(pa.Table.from_pylist(rows), folder / "sample.feather")
        else:
            # 只含example03所需的一档报价列，没有Volume和LastPrice。
            with (folder / "sample.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=(
                    "TradingDay", "InstrumentID", "UpdateTime", "UpdateMillisec",
                    "BidPrice1", "BidVolume1", "AskPrice1", "AskVolume1",
                ))
                writer.writeheader()
                for minute in range(28):
                    close = _closes(minute)[index]
                    writer.writerow({
                        "TradingDay": "20260728", "InstrumentID": spec.symbol,
                        "UpdateTime": (start + timedelta(minutes=minute)).strftime("%H:%M:%S"),
                        "UpdateMillisec": 0,
                        "BidPrice1": close, "BidVolume1": 2,
                        "AskPrice1": close + spec.price_increment, "AskVolume1": 3,
                    })


def test_feed(source: str) -> None:
    """先验证Parser/Reader/Tick聚合，不启动Nautilus模拟引擎。"""
    with TemporaryDirectory(prefix=f"cross-section-{source}-") as directory:
        root = Path(directory)
        _make_fixture(root, source)
        feed = build_feed(source, root)
        bars = []
        feed.register_bar_handler(bars.append)
        for spec in CONTRACT_SPECS:
            feed.subscribe(spec.instrument_id, DataType.BAR, "1-MINUTE")
        feed.connect()
        try:
            summary = feed.replay()
        finally:
            feed.disconnect()
        assert len(bars) == 27 * 5
        assert len({bar.bar_type.instrument_id for bar in bars}) == 5
        if source == "tick":
            assert summary.quote_ticks == 28 * 5
            assert all("MID" in str(bar.bar_type) for bar in bars)
            assert all(bar.volume.as_decimal() == 0 for bar in bars)
            first = next(bar for bar in bars if bar.bar_type.instrument_id == CONTRACT_SPECS[0].instrument_id)
            assert first.close.as_decimal() == 100.5
        else:
            assert summary.bars == 27 * 5
        print(f"第二类{source}输入通过：五路共{len(bars)}根标准Bar")


def test_formal(source: str) -> None:
    """相同策略分别接Bar与Tick→MID Bar，至少产生一次组合调仓。"""
    with TemporaryDirectory(prefix=f"cross-section-{source}-") as directory:
        root = Path(directory)
        _make_fixture(root, source)
        run_case(source, root)
    print(f"第二类{source}正式离线回测通过")


def test_instrument_factory() -> None:
    """Feed和模拟Backend共用同一合约工厂，避免Bar精度与撮合合约漂移。"""
    names = [spec.symbol for spec in CONTRACT_SPECS] + list(BINANCE_SYMBOLS)
    for name in names:
        profile, instrument, meta = create_instrument(name)
        assert meta.instrument_id == instrument.id
        assert meta.price_precision == instrument.price_precision
        assert meta.price_increment == instrument.price_increment.as_decimal()
        assert str(profile.venue) == meta.exchange
    print("第二类合约工厂通过：CTP/BN十个合约的Profile、instrument和Meta保持一致")


STAGES = {
    "instrument-factory": test_instrument_factory,
    "bar-feed": lambda: test_feed("bar"),
    "tick-feed": lambda: test_feed("tick"),
    "bar-formal": lambda: test_formal("bar"),
    "tick-formal": lambda: test_formal("tick"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="第二类五品种两种离线来源测试")
    parser.add_argument("--stage", required=True, choices=tuple(STAGES))
    args = parser.parse_args()
    STAGES[args.stage]()


if __name__ == "__main__":
    main()
