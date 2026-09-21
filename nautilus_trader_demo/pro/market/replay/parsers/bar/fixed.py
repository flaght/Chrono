"""文件路径已经确定标的时，为逐行Bar补充固定身份。"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from market.basic.base import InstrumentId
from market.replay.parsers.bar.mapped import BarColumns, MappedBarParser
from market.replay.parsers.base import ParsedEvent, ParserContext


class FixedInstrumentBarParser:
    """适配只有datetime/OHLCV列、没有symbol/exchange列的Feather文件。"""

    def __init__(
        self,
        instrument_id: InstrumentId,
        *,
        timestamp: str = "datetime",
        bar_spec: str = "1-MINUTE",
        timezone: str = "Asia/Shanghai",
    ) -> None:
        self.instrument_id = instrument_id
        self._mapped = MappedBarParser(
            columns=BarColumns(
                symbol="__fixed_symbol__",
                exchange="__fixed_exchange__",
                timestamp=timestamp,
                open="open",
                high="high",
                low="low",
                close="close",
                volume="volume",
            ),
            bar_spec=bar_spec,
            timezone=timezone,
        )

    def reset(self) -> None:
        self._mapped.reset()

    def parse(self, row: Mapping[str, Any], context: ParserContext) -> Iterable[ParsedEvent]:
        enriched = {
            **row,
            "__fixed_symbol__": self.instrument_id.symbol.value,
            "__fixed_exchange__": str(self.instrument_id.venue),
        }
        return self._mapped.parse(enriched, context)
