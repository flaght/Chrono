"""JSONL历史行情Provider；每行已完整结束的真实合约分钟。"""
from pathlib import Path
import json
import csv
from .base import HistoryBar, HistoryProvider, select_bars


class FileHistoryProvider(HistoryProvider):
    storage_kind = "file"
    def __init__(self, config):
        self.path = Path(config)

    def open(self):
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

    def read(self, request):
        rows = []
        with self.path.open() as handle:
            records = csv.DictReader(handle) if self.path.suffix.lower() == ".csv" else (json.loads(line) for line in handle)
            for row in records:
                if row.get("instrument_id") != request.instrument_id:
                    continue
                row["ts_event"] = int(row["ts_event"])
                if request.start_ns <= row["ts_event"] < request.end_ns:
                    rows.append(HistoryBar(**{k: row[k] for k in (
                        "instrument_id", "ts_event", "close", "adjusted_close", "open", "high", "low", "volume", "cumulative_factor") if k in row and row[k] != ""}))
        return select_bars(rows, request)

    def close(self):
        pass
