"""回归测试：实际Feather交易所别名和BN逐标的Kline解析器。"""

from __future__ import annotations

from pathlib import Path

from examples.cross_section.run_bar_backtest import (
    BINANCE_SYMBOLS,
    CTP_DATA_DIR,
    BINANCE_DATA_DIR,
    build_feed,
)
from market.replay.parsers.base import ParserContext


def test_ctp_exchange_aliases() -> None:
    feed = build_feed(CTP_DATA_DIR, market="ctp")
    parser = feed._sources[0].parser
    for symbol, exchange, expected in (
        ("SA703", "XZCE", "SA703.CZCE"),
        ("rb2704", "XSGE", "rb2704.SHFE"),
        ("jm2608", "XDCE", "jm2608.DCE"),
        ("jm2608", "DCE", "jm2608.DCE"),
    ):
        row = {
            "symbol": symbol,
            "exchange": exchange,
            "datetime": "2026-07-28 09:31:00",
            "open": 3000, "high": 3001, "low": 2999, "close": 3000,
            "volume": 5, "value": 15000, "open_interest": 100,
            "vwap": 3000,
        }
        context = ParserContext(Path("sample.feather"), 1, feed.get_instrument_meta)
        events = parser.parse(row, context)
        assert str(events[0].instrument_id) == expected
    print("CTP Bar交易所映射通过：XZCE→CZCE、XSGE→SHFE、XDCE→DCE")


def test_binance_parser_per_symbol() -> None:
    feed = build_feed(BINANCE_DATA_DIR, market="binance")
    assert len(feed._sources) == len(BINANCE_SYMBOLS)
    for source, symbol in zip(feed._sources, BINANCE_SYMBOLS):
        assert str(source.parser.instrument_id) == f"{symbol}-PERP.BINANCE"
    print("BN Bar解析器通过：五份CSV分别绑定自己的合约代码")


if __name__ == "__main__":
    test_ctp_exchange_aliases()
    test_binance_parser_per_symbol()
