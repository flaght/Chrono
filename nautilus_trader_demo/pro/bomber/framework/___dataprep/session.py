"""每次运行独立管理、生命周期有限的输入缓存与诊断收集。"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from functools import wraps
import json
from pathlib import Path
from collections.abc import Mapping

from .contracts import CoverageReport, InputError, InputIssue, SCHEMA_VERSION

_CURRENT = ContextVar("input_session", default=None)


def json_value(value):
    if is_dataclass(value):
        return {f.name: json_value(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [json_value(v) for v in value]
    if isinstance(value, (Path, Decimal, date, datetime)):
        return str(value)
    return value


class InputSession:
    def __init__(self):
        self.cache = {}
        self.manifest = {}
        self.issues = []
        self.coverage = CoverageReport()
        self.paths = []
        self.counters = {"directory_scans": 0, "feather_reads": 0}
        self._token = None

    def __enter__(self):
        if self._token is not None:
            raise RuntimeError("An InputSession cannot be re-entered")
        self._token = _CURRENT.set(self)
        return self

    def __exit__(self, *exc):
        _CURRENT.reset(self._token)
        self._token = None
        self.cache.clear()

    def fingerprint(self, file):
        path = Path(file).expanduser().resolve()
        stat = path.stat()
        version = (stat.st_size, stat.st_mtime_ns)
        previous = self.manifest.get(str(path))
        if previous and (previous["size"], previous["mtime_ns"]) != version:
            fail("SOURCE_CHANGED", "Input changed during this run", source=path)
        self.manifest[str(path)] = {"size": version[0], "mtime_ns": version[1]}
        return (str(path), *version)

    def write_reports(self, directory):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        payloads = {
            "input_manifest.json": {"schema_version": SCHEMA_VERSION,
                "files": self.manifest, "paths": self.paths, "counters": self.counters,
                "evidence": "file_size_and_mtime_not_content_hash"},
            "input_coverage.json": self.coverage,
            "input_issues.json": self.issues,
        }
        for name, payload in payloads.items():
            (root / name).write_text(json.dumps(json_value(payload), ensure_ascii=False,
                indent=2, allow_nan=False) + "\n", encoding="utf-8")


def current_session():
    return _CURRENT.get()


def record_issue(issue):
    session = current_session()
    if session is not None:
        session.issues.append(issue)


def fail(code, message, **details):
    if "source" in details and details["source"] is not None:
        details["source"] = str(details["source"])
    issue = InputIssue(code, message, **details)
    record_issue(issue)
    raise InputError(issue)


def input_session(function):
    """只包装数据准备流程，不在导入时调用被包装函数。"""
    @wraps(function)
    def wrapped(*args, **kwargs):
        if current_session() is not None:
            session = current_session()
            if "start_day" in kwargs:
                session.coverage.requested = (kwargs.get("start_day"), kwargs.get("end_day"))
            elif args and hasattr(args[0], "start_day"):
                session.coverage.requested = (args[0].start_day, args[0].end_day)
            try:
                return function(*args, **kwargs)
            except Exception as exc:
                _write_failure(session, args, kwargs, exc)
                raise
        with InputSession() as session:
            if "start_day" in kwargs:
                session.coverage.requested = (kwargs.get("start_day"), kwargs.get("end_day"))
            elif args and hasattr(args[0], "start_day"):
                session.coverage.requested = (args[0].start_day, args[0].end_day)
            try:
                return function(*args, **kwargs)
            except Exception as exc:
                _write_failure(session, args, kwargs, exc)
                raise
    return wrapped


def _write_failure(session, args, kwargs, exc):
    # 保留原始异常，报告写入失败不能覆盖原始错误。
    from uuid import uuid4
    root = kwargs.get("report_dir")
    if root is None and args:
        root = getattr(args[0], "report_dir", None)
    if root is None:
        return
    if not isinstance(exc, InputError):
        session.issues.append(InputIssue("RUN_FAILED", str(exc)))
    try:
        session.write_reports(Path(root) / ("input_failed_" + uuid4().hex[:12]))
    except (OSError, TypeError, ValueError):
        pass


def write_input_reports(directory):
    session = current_session()
    if session is not None:
        session.write_reports(directory)


def read_feather(path, columns=None):
    import pandas as pd
    session = current_session()
    if session is None:
        return pd.read_feather(path, columns=columns)
    fingerprint = session.fingerprint(path)
    full_key = ("feather", fingerprint, None)
    if full_key in session.cache:
        frame = session.cache[full_key]
        return (frame if columns is None else frame.loc[:, list(columns)]).copy(deep=True)
    key = ("feather", fingerprint, None if columns is None else tuple(columns))
    if key not in session.cache:
        session.cache[key] = pd.read_feather(path, columns=columns)
        session.counters["feather_reads"] += 1
    return session.cache[key].copy(deep=True)
