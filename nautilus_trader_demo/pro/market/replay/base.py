from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from market.basic.base import (
    Bar,
    CustomBar,
    InstrumentMeta,
    MarketDataFeed,
    QuoteTick,
    DataType,
    SubscriptionRequest,
    TradeTick,
)
from market.replay.parsers.base import (
    DataLoadError,
    MarketEvent,
    ParsedEvent,
    ParserContext,
    RowParser,
)
from market.replay.readers import CsvReader, FeatherReader, ReaderError, RowReader

@dataclass(frozen=True)
class FileSource:
    path: str | Path
    reader: RowReader
    parser: RowParser
    kind: Literal["tick", "bar"]

@dataclass(frozen=True)
class ReplaySummary:
    trade_ticks: int = 0
    quote_ticks: int = 0
    bars: int = 0
    custom_bars: int = 0

    @property
    def total(self) -> int:
        return self.trade_ticks + self.quote_ticks + self.bars + self.custom_bars



class FileReplayFeed(MarketDataFeed):
    """Read CSV/Feather sources, sort events and dispatch subscriptions."""

    def __init__(self, source_id: str = "FILE_REPLAY_FEED") -> None:
        super().__init__(source_id=source_id)
        self._sources: list[FileSource] = []
        self._events: tuple[MarketEvent, ...] = ()

    def add_tick_csv(self, path: str | Path, parser: RowParser) -> None:
        self._add_source(path, CsvReader(), parser, "tick")

    def add_bar_csv(self, path: str | Path, parser: RowParser) -> None:
        self._add_source(path, CsvReader(), parser, "bar")

    def add_bar_feather(self, path: str | Path, parser: RowParser) -> None:
        self._add_source(path, FeatherReader(), parser, "bar")

    def add_source(
        self,
        path: str | Path,
        reader: RowReader,
        parser: RowParser,
        kind: Literal["tick", "bar"],
    ) -> None:
        """Register an explicit reader/parser pair for an offline source."""
        self._add_source(path, reader, parser, kind)

    def _add_source(
        self,
        path: str | Path,
        reader: RowReader,
        parser: RowParser,
        kind: Literal["tick", "bar"],
    ) -> None:
        suffixes = getattr(reader, "suffixes", frozenset())
        if suffixes and Path(path).suffix.lower() not in suffixes:
            expected = ", ".join(sorted(suffixes))
            raise DataLoadError(f"{kind} file must use one of [{expected}]: {path}")
        self._sources.append(FileSource(path, reader, parser, kind))
        self._events = ()
        
    def register_instrument(self, meta: InstrumentMeta) -> None:
        super().register_instrument(meta)
        self._events = ()

    def connect(self) -> None:
        if not self._sources:
            raise DataLoadError("no file source configured")
        self._is_connected = True

    def disconnect(self) -> None:
        self._events = ()
        self._is_connected = False

    def _on_subscription_added(self, request: SubscriptionRequest) -> None:
        del request
        self._events = ()

    def _on_subscription_removed(self, request: SubscriptionRequest) -> None:
        del request
        self._events = ()

    def load_events(self, force_reload: bool = False) -> tuple[MarketEvent, ...]:
        if not self._is_connected:
            raise RuntimeError("feed must be connected before loading")
        if self._events and not force_reload:
            return self._events

        loaded: list[tuple[int, ParsedEvent]] = []
        sequence = 0
        for source in self._sources:
            path = Path(source.path).expanduser().resolve()
            if not path.is_file():
                raise DataLoadError(f"file does not exist: {path}")
            source.parser.reset()
            position = 0
            try:
                rows = source.reader.read(path)
                for position, row in rows:
                    context = ParserContext(path, position, self.get_instrument_meta)
                    parsed_events = source.parser.parse(row, context)
                    for parsed in parsed_events:
                        _validate_source_event(source.kind, parsed, context)
                        if self._subscribed(parsed):
                            loaded.append((sequence, parsed))
                            sequence += 1
            except DataLoadError:
                raise
            except ReaderError as exc:
                raise DataLoadError(str(exc)) from exc
            except Exception as exc:
                raise ParserContext(path, position, self.get_instrument_meta).error(str(exc)) from exc

        # Sort by historical clock while retaining file order for equal times.
        loaded.sort(key=lambda item: (item[1].payload.ts_init, item[0]))
        self._events = tuple(item[1].payload for item in loaded)
        return self._events


    def replay(self) -> ReplaySummary:
        trade_count = quote_count = bar_count = custom_bar_count = 0
        for event in self.load_events():
            if isinstance(event, TradeTick):
                self._emit_trade_tick(event)
                trade_count += 1
            elif isinstance(event, QuoteTick):
                self._emit_quote_tick(event)
                quote_count += 1
            elif isinstance(event, Bar):
                self._emit_bar(event)
                bar_count += 1
            elif isinstance(event, CustomBar):
                self._emit_custom_bar(event)
                custom_bar_count += 1
        return ReplaySummary(trade_count, quote_count, bar_count, custom_bar_count)

    def replay_step(self) -> Iterable[MarketEvent]:
        """Dispatch and yield one event at a time."""
        for event in self.load_events():
            if isinstance(event, TradeTick):
                self._emit_trade_tick(event)
            elif isinstance(event, QuoteTick):
                self._emit_quote_tick(event)
            elif isinstance(event, Bar):
                self._emit_bar(event)
            elif isinstance(event, CustomBar):
                self._emit_custom_bar(event)
            yield event

    def _subscribed(self, event: ParsedEvent) -> bool:
        for request in self._subscriptions.get(event.instrument_id, set()):
            if request.data_type is not event.data_type:
                continue
            if event.data_type not in (DataType.BAR, DataType.CUSTOM_BAR):
                return True
            if request.bar_spec is None or request.bar_spec == event.bar_spec:
                return True
        return False

def _validate_source_event(
    kind: Literal["tick", "bar"],
    event: ParsedEvent,
    context: ParserContext,
) -> None:
    if kind == "tick" and not isinstance(event.payload, (TradeTick, QuoteTick)):
        raise context.error("Tick parser returned a non-tick event")
    if kind == "bar" and not isinstance(event.payload, (Bar, CustomBar)):
        raise context.error("Bar parser returned a non-bar event")
