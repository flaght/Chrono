"""Lifecycle and stream-subscription wrapper for the official DolphinDB SDK."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


logger = logging.getLogger("DolphinDbDriver")
RowHandler = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class NativeDolphinSubscription:
    table_name: str
    action_name: str
    columns: tuple[str, ...]
    offset: int = -1
    resub: bool = True
    batch_size: int = 0
    throttle: float = 0.01
    filter: Any = field(default=None, compare=False, hash=False, repr=False)


class DolphinDbDriver(Protocol):
    def start(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        streaming_port: int = 0,
        keep_alive_seconds: int = 60,
    ) -> None: ...

    def subscribe(self, subscription: NativeDolphinSubscription, handler: RowHandler) -> None: ...

    def unsubscribe(self, subscription: NativeDolphinSubscription) -> None: ...

    def stop(self) -> None: ...


class OfficialDolphinDbDriver:
    """Own one SDK session while exposing no market-model dependencies."""

    def __init__(self, session_factory: Callable[..., Any] | None = None) -> None:
        self._session_factory = session_factory
        self._session: Any = None
        self._host = ""
        self._port = 0
        self._username = ""
        self._password = ""
        self._subscriptions: dict[
            tuple[str, str],
            tuple[NativeDolphinSubscription, Callable[[Any], None]],
        ] = {}

    def start(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        streaming_port: int = 0,
        keep_alive_seconds: int = 60,
    ) -> None:
        if self._session is not None:
            return
        factory = self._session_factory or _official_session_factory()
        session = factory(keepAliveTime=keep_alive_seconds)
        connected = session.connect(host, port, username, password)
        if connected is False:
            close = getattr(session, "close", None)
            if close is not None:
                close()
            raise ConnectionError(f"DolphinDB connection failed: {host}:{port}")
        if streaming_port:
            session.enableStreaming(streaming_port)
        else:
            session.enableStreaming()
        self._session = session
        self._host = host
        self._port = port
        self._username = username
        self._password = password

    def subscribe(
        self,
        subscription: NativeDolphinSubscription,
        handler: RowHandler,
    ) -> None:
        session = self._require_session()
        key = (subscription.table_name, subscription.action_name)
        if key in self._subscriptions:
            return
        callback = _make_callback(subscription.columns, handler)
        session.subscribe(
            host=self._host,
            port=self._port,
            handler=callback,
            tableName=subscription.table_name,
            actionName=subscription.action_name,
            offset=subscription.offset,
            resub=subscription.resub,
            filter=subscription.filter,
            msgAsTable=False,
            batchSize=subscription.batch_size,
            throttle=subscription.throttle,
            userName=self._username,
            password=self._password,
        )
        self._subscriptions[key] = (subscription, callback)
        logger.info(
            "DolphinDB 流表订阅成功: table=%s action=%s",
            subscription.table_name,
            subscription.action_name,
        )

    def unsubscribe(self, subscription: NativeDolphinSubscription) -> None:
        key = (subscription.table_name, subscription.action_name)
        if key not in self._subscriptions:
            return
        self._require_session().unsubscribe(
            self._host,
            self._port,
            subscription.table_name,
            subscription.action_name,
        )
        self._subscriptions.pop(key, None)

    def stop(self) -> None:
        session = self._session
        if session is None:
            return
        for subscription, _ in reversed(tuple(self._subscriptions.values())):
            try:
                session.unsubscribe(
                    self._host,
                    self._port,
                    subscription.table_name,
                    subscription.action_name,
                )
            except Exception:
                logger.exception(
                    "DolphinDB 取消订阅失败: table=%s action=%s",
                    subscription.table_name,
                    subscription.action_name,
                )
        self._subscriptions.clear()
        try:
            close = getattr(session, "close", None)
            if close is not None:
                close()
        finally:
            self._session = None

    def _require_session(self) -> Any:
        if self._session is None:
            raise RuntimeError("DolphinDB driver has not been started")
        return self._session


def create_native_driver() -> OfficialDolphinDbDriver:
    return OfficialDolphinDbDriver()


def _official_session_factory() -> Callable[..., Any]:
    try:
        import dolphindb as ddb
    except ImportError as exc:
        raise RuntimeError(
            "DolphinDB Python SDK is not installed; install the version "
            "matching the DolphinDB server",
        ) from exc
    return getattr(ddb, "Session", None) or getattr(ddb, "session")


def _make_callback(columns: tuple[str, ...], handler: RowHandler) -> Callable[[Any], None]:
    def callback(message: Any) -> None:
        for row in rows_from_message(message, columns):
            handler(row)

    return callback


def rows_from_message(message: Any, columns: tuple[str, ...]) -> list[dict[str, Any]]:
    """Normalize DolphinDB single-row, batch and DataFrame callback payloads."""
    if isinstance(message, Mapping):
        return [dict(message)]
    to_dict = getattr(message, "to_dict", None)
    if callable(to_dict):
        return [dict(row) for row in to_dict("records")]
    if not _is_sequence_like(message):
        raise TypeError(f"unsupported DolphinDB callback payload: {type(message)!r}")

    values = list(message)
    if len(values) == len(columns) and (not values or not _is_sequence_like(values[0])):
        return [dict(zip(columns, values))]

    rows: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        if isinstance(value, Mapping):
            rows.append(dict(value))
            continue
        if not _is_sequence_like(value):
            raise TypeError(f"DolphinDB batch row {index} is not a sequence")
        row_values = list(value)
        if len(row_values) != len(columns):
            raise ValueError(
                f"DolphinDB {len(columns)}-column schema expected, "
                f"batch row {index} has {len(row_values)} values",
            )
        rows.append(dict(zip(columns, row_values)))
    return rows


def _is_sequence_like(value: Any) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    return hasattr(value, "__len__") and hasattr(value, "__iter__")
