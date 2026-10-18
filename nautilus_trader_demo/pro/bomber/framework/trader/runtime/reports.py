"""在线运行报告；策略只提供自己的指标，不参与账户或生命周期处理。"""

from pathlib import Path
import json
import time


class LiveRunReport:
    def __init__(self, report_dir, *, prefix="live", metadata=None, describe=None, clock_ns=None):
        self.root = Path(report_dir)
        self.prefix = prefix
        self.metadata = dict(metadata or {})
        self.describe = describe or (lambda session: {})
        self.clock_ns = clock_ns or (lambda: time.time_ns())
        self.path = None

    def begin(self):
        path = self.root / f"{self.prefix}-{self.clock_ns()}"
        path.mkdir(parents=True, exist_ok=False)
        self.path = path

    def write(self, *, status, fields, session, events):
        result = {**self.metadata, **self.describe(session), **fields, "status": status}
        (self.path / "summary.json").write_text(json.dumps(result, ensure_ascii=False,
            indent=2, default=str) + "\n", encoding="utf-8")
        if session is not None:
            (self.path / "events.txt").write_text("\n".join(map(str, events)) + "\n", encoding="utf-8")
        print(f"停机结果: {result}\n记录目录: {self.path}", flush=True)
        return result
