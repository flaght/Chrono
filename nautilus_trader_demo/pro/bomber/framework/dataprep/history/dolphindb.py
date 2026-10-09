"""DolphinDB历史分钟Provider；可配置字段映射，构造时不加载SDK或连接。"""
from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime, timedelta, timezone
import re
from threading import RLock
from .base import HistoryBar, HistoryProvider, HistoryError, HistoryUnavailable, select_bars, MINUTE_NS
from ..sources.dolphindb import DolphinDbReferenceConfig, _session_factory


@dataclass(frozen=True)
class DolphinDbHistoryConfig:
    connection: DolphinDbReferenceConfig
    database: str
    table: str
    columns: dict = field(default_factory=lambda: {"instrument_id": "instrument_id", "ts_event": "ts_event", "close": "close"})
    time_unit: str = "ns"
    time_label: str = "end_ns"
    instrument_format: str = "full"
    venue: str | None = None
    date_column: str | None = None
    time_column: str | None = None
    source_timezone: str = "UTC"

    def __post_init__(self):
        if re.fullmatch(r"dfs://[A-Za-z0-9_/-]+", self.database) is None:
            raise HistoryError("行情数据库须为安全的dfs路径")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.table) is None:
            raise HistoryError("行情表名无效；须提供实际表名")
        allowed = {"instrument_id", "ts_event", "close", "adjusted_close", "open", "high", "low", "volume", "cumulative_factor"}
        columns = dict(self.columns)
        composite = self.date_column is not None or self.time_column is not None
        required = {"instrument_id"} if composite else {"instrument_id", "ts_event"}
        if (not required <= columns.keys()
                or not {"close", "adjusted_close"} & columns.keys() or not columns.keys() <= allowed
                or any(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", v) is None for v in columns.values())):
            raise HistoryError("行情字段映射缺失或无效")
        if composite and (self.date_column is None or self.time_column is None or "ts_event" in columns
                or any(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None
                    for name in (self.date_column, self.time_column))):
            raise HistoryError("组合时间须同时提供安全date_column/time_column，不能同时映射ts_event")
        if self.source_timezone not in {"UTC", "Asia/Shanghai"}:
            raise HistoryError("历史源时区仅支持UTC或Asia/Shanghai，不能猜测偏移")
        if composite and (self.time_label != "start" or self.time_unit != "ns"):
            raise HistoryError("组合date/time当前仅支持分钟开始标签，统一转为纳秒")
        if self.time_unit not in {"ns", "ms", "s"} or self.time_label not in {"start", "end_ns"}:
            raise HistoryError("须声明时间单位及分钟起点/末尾口径")
        if self.instrument_format not in {"full", "symbol"} or (self.instrument_format == "symbol"
                and (self.venue is None or re.fullmatch(r"[A-Z0-9]+", self.venue) is None)):
            raise HistoryError("symbol格式须明确固定Venue，不能猜测交易所")
        object.__setattr__(self, "columns", columns)


class DolphinDbHistoryProvider(HistoryProvider):
    def __init__(self, config, *, session_factory=None):
        if not isinstance(config, DolphinDbHistoryConfig):
            raise TypeError("须提供DolphinDbHistoryConfig")
        self.config = config
        self._factory = session_factory or _session_factory
        self._session = None
        self._lock = RLock()

    def open(self):
        with self._lock:
            if self._session is not None:
                return
            session = self._factory(enableSSL=self.config.connection.enable_ssl)
            try:
                c = self.config.connection
                if session.connect(c.host, c.port, c.username, c.password, reconnect=False,
                        readTimeout=c.read_timeout_seconds, writeTimeout=c.read_timeout_seconds) is not True:
                    raise HistoryUnavailable("历史库未确认连接成功")
            except Exception:
                session.close()
                raise HistoryUnavailable("历史库连接失败") from None
            self._session = session

    def _script(self, request):
        if re.fullmatch(r"[A-Za-z0-9_.-]+", request.instrument_id) is None:
            raise HistoryError("合约标识无效")
        c = self.config
        symbol = request.instrument_id
        if c.instrument_format == "symbol":
            if not symbol.endswith("." + c.venue):
                raise HistoryError("请求交易所与历史库固定Venue不一致")
            symbol = symbol.rsplit(".", 1)[0]
        if c.date_column is not None:
            end = f"long(nanotimestamp(concatDateTime({c.date_column},{c.time_column})))"
            ordering = f"{c.date_column},{c.time_column}"
        else:
            scale = {"ns": 1, "ms": 1000000, "s": 1000000000}[c.time_unit]
            end = f"long({c.columns['ts_event']})*{scale}"
            ordering = c.columns['ts_event']
        if c.source_timezone == "Asia/Shanghai":
            end = f"({end}-{8 * 3600 * 1000000000}l)"
        if c.time_label == "start":
            end = f"({end}+{MINUTE_NS - 1}l)"
        fields = ",".join(f"{physical} as {logical}" for logical, physical in c.columns.items() if logical != "ts_event")
        partition_filter = ""
        if c.date_column is not None:
            zone = timezone(timedelta(hours=8 if c.source_timezone == "Asia/Shanghai" else 0))
            start = datetime.fromtimestamp(request.start_ns // 1000000000, zone).strftime("%Y.%m.%d")
            last = datetime.fromtimestamp((request.end_ns - 1) // 1000000000, zone).strftime("%Y.%m.%d")
            partition_filter = f"{c.date_column}>={start},{c.date_column}<={last},"
        # 只使用已验证名称、整数和合约字符串；读取多一行检测截断，禁止截断后当完整窗口。
        return (f'select top {request.max_rows + 1} {fields},{end} as ts_event '
            f'from loadTable("{c.database}","{c.table}") '
            f'where {partition_filter}{c.columns["instrument_id"]}="{symbol}",'
            f'{end}>={request.start_ns}l,{end}<{request.end_ns}l order by {ordering}')

    def read(self, request):
        with self._lock:
            if self._session is None:
                raise HistoryUnavailable("历史库尚未打开")
            try:
                frame = self._session.run(self._script(request))
            except HistoryError:
                raise
            except Exception:
                raise HistoryUnavailable("历史分钟查询失败") from None
            if not hasattr(frame, "to_dict"):
                raise HistoryError("历史库须返回标准表格")
            rows = frame.to_dict("records")
            if len(rows) > request.max_rows:
                raise HistoryError("历史查询被行数上限截断")
            bars = []
            for row in rows:
                stamp = row.pop("ts_event")
                if int(stamp) != stamp:
                    raise HistoryError("历史时间戳精度丢失")
                if self.config.instrument_format == "symbol":
                    row["instrument_id"] = str(row["instrument_id"]) + "." + self.config.venue
                bars.append(HistoryBar(ts_event=int(stamp), **row))
            return select_bars(bars, request)

    def close(self):
        with self._lock:
            session, self._session = self._session, None
            if session is not None:
                session.close()
