import csv,pdb
from dataclasses import dataclass
from pathlib import Path
from typing import Callable,Protocol,Mapping,Any,Iterable,Literal

from market.basic.base import (
    Bar,
    CustomBar,
    InstrumentId,
    InstrumentMeta,
    QuoteTick,
    TradeTick,
    DataType,
    make_bar,
    make_custom_bar,
    make_custom_bar_all_in_one,
    make_quote_tick,
    make_trade_tick,
    SubscriptionRequest
)

from market.basic.base import  MarketDataFeed


MarketEvent = TradeTick | QuoteTick | Bar | CustomBar


class DataLoadError(ValueError):
    """A file row cannot be converted into market data."""


@dataclass(frozen=True)
class ParsedEvent:
    data_type: DataType
    instrument_id: InstrumentId
    payload: MarketEvent
    bar_spec: str | None = None


@dataclass(frozen=True)
class ParserContext:
    path: Path
    line: int
    get_meta: Callable[[InstrumentId], InstrumentMeta | None]

    def require_meta(self, instrument_id: InstrumentId) -> InstrumentMeta:
        meta = self.get_meta(instrument_id)
        if meta is None:
            raise self.error(f"instrument is not registered: {instrument_id}")
        return meta

    def error(self, message: str) -> DataLoadError:
        return DataLoadError(f"{self.path}:{self.line}: {message}")


class RowParser(Protocol):
    """Source-specific row converter, for example CTP or Binance."""

    def reset(self) -> None: ...

    def parse(
        self,
        row: Mapping[str, Any],
        context: ParserContext,
    ) -> Iterable[ParsedEvent]: ...



@dataclass(frozen=True)
class FileSource:
    path: str | Path
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
        self._add_source(path, parser, "tick", ".csv")

    def add_bar_feather(self, path: str | Path, parser: RowParser) -> None:
        self._add_source(path, parser, "bar", ".feather")

    def _add_source(
        self,
        path: str | Path,
        parser: RowParser,
        kind: Literal["tick", "bar"],
        required_suffix: str,
    ) -> None:
        if Path(path).suffix.lower() != required_suffix:
            raise DataLoadError(f"{kind} file must use {required_suffix}: {path}")
        self._sources.append(FileSource(path, parser, kind))
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
            for position, row in _read_rows(path):
                context = ParserContext(path, position, self.get_instrument_meta)
                try:
                    parsed_events = source.parser.parse(row, context)
                    for parsed in parsed_events:
                        _validate_source_event(source.kind, parsed, context)
                        if self._subscribed(parsed):
                            loaded.append((sequence, parsed))
                            sequence += 1
                except DataLoadError:
                    raise
                except Exception as exc:
                    raise context.error(str(exc)) from exc

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

def _read_rows(path: Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise DataLoadError(f"CSV file has no header: {path}")
            for line, row in enumerate(reader, start=2):
                yield line, row
        return

    if suffix == ".feather":
        try:
            import pyarrow.feather as feather
        except ImportError as exc:
            raise DataLoadError("reading Bar Feather files requires pyarrow") from exc
        table = feather.read_table(path)
        for row_number, row in enumerate(table.to_pylist(), start=1):
            yield row_number, row
        return

    raise DataLoadError(f"unsupported file type: {path.suffix}")


def _validate_source_event(
    kind: Literal["tick", "bar"],
    event: ParsedEvent,
    context: ParserContext,
) -> None:
    if kind == "tick" and not isinstance(event.payload, (TradeTick, QuoteTick)):
        raise context.error("Tick CSV parser returned a non-tick event")
    if kind == "bar" and not isinstance(event.payload, (Bar, CustomBar)):
        raise context.error("Bar Feather parser returned a non-bar event")
