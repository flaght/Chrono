"""Transport boundary around the project-owned CTP MD extension."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Protocol


logger = logging.getLogger("CtpMdDriver")


class CtpMdCallbacks(Protocol):
    def on_front_connected(self) -> None: ...
    def on_front_disconnected(self, reason: int) -> None: ...
    def on_login_response(self, error: Mapping[str, Any] | None) -> None: ...
    def on_subscribe_response(self, data: Mapping[str, Any] | None, error: Mapping[str, Any] | None) -> None: ...
    def on_unsubscribe_response(self, data: Mapping[str, Any] | None, error: Mapping[str, Any] | None) -> None: ...
    def on_depth_market_data(self, data: Mapping[str, Any]) -> None: ...
    def on_api_error(self, error: Mapping[str, Any] | None) -> None: ...


class CtpMdDriver(Protocol):
    """Transport operations required by ``CtpLiveDataFeed``."""

    def start(self, front: str, flow_path: Path, production_mode: bool) -> None: ...
    def stop(self) -> None: ...
    def login(self, broker_id: str, user_id: str, password: str) -> None: ...
    def subscribe(self, symbol: str) -> None: ...
    def unsubscribe(self, symbol: str) -> None: ...


def create_native_driver(callbacks: CtpMdCallbacks) -> CtpMdDriver:
    """Create the production driver backed by our ``bomber_ctp_md`` module."""
    try:
        from bomber_ctp_md import MdApi
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "未找到项目自有 CTP 扩展 bomber_ctp_md；"
            "请先按 market/native/ctp/binding/README.md 编译安装",
        ) from exc

    class CallbackApi(MdApi):
        def onFrontConnected(self) -> None:
            _invoke(callbacks.on_front_connected)

        def onFrontDisconnected(self, reason: int) -> None:
            _invoke(callbacks.on_front_disconnected, reason)

        def onRspUserLogin(self, data: dict[str, Any], error: dict[str, Any], reqid: int, last: bool) -> None:
            del data, reqid, last
            _invoke(callbacks.on_login_response, error)

        def onRspSubMarketData(self, data: dict[str, Any], error: dict[str, Any], reqid: int, last: bool) -> None:
            del reqid, last
            _invoke(callbacks.on_subscribe_response, data, error)

        def onRspUnSubMarketData(self, data: dict[str, Any], error: dict[str, Any], reqid: int, last: bool) -> None:
            del reqid, last
            _invoke(callbacks.on_unsubscribe_response, data, error)

        def onRspError(self, error: dict[str, Any], reqid: int, last: bool) -> None:
            del reqid, last
            _invoke(callbacks.on_api_error, error)

        def onRtnDepthMarketData(self, data: dict[str, Any]) -> None:
            _invoke(callbacks.on_depth_market_data, data)

    return NativeCtpMdDriver(CallbackApi())


class NativeCtpMdDriver:
    """Lifecycle wrapper for the low-level pybind ``MdApi`` object."""

    def __init__(self, api: Any) -> None:
        self._api = api
        self._started = False
        self._request_id = 0

    def start(self, front: str, flow_path: Path, production_mode: bool) -> None:
        flow_path.mkdir(parents=True, exist_ok=True)
        created = False
        try:
            self._api.createFtdcMdApi(f"{flow_path}/", production_mode)
            created = True
            self._api.registerFront(front)
            self._api.init()
            self._started = True
        except Exception:
            if created:
                self._api.release()
            raise

    def stop(self) -> None:
        if self._started:
            self._api.exit()
            self._started = False

    def login(self, broker_id: str, user_id: str, password: str) -> None:
        self._request_id += 1
        result = self._api.reqUserLogin(
            {"BrokerID": broker_id, "UserID": user_id, "Password": password},
            self._request_id,
        )
        _raise_submit_error("登录", result)

    def subscribe(self, symbol: str) -> None:
        _raise_submit_error(f"订阅 {symbol}", self._api.subscribeMarketData(symbol))

    def unsubscribe(self, symbol: str) -> None:
        _raise_submit_error(f"退订 {symbol}", self._api.unSubscribeMarketData(symbol))


def _raise_submit_error(action: str, result: Any) -> None:
    code = int(result or 0)
    if code:
        raise RuntimeError(f"CTP {action}请求提交失败: code={code}")


def _invoke(callback: Any, *args: Any) -> None:
    """Keep Python exceptions from escaping into the C++ callback boundary."""
    try:
        callback(*args)
    except Exception:
        logger.exception("CTP 原生回调处理失败: %s", callback.__name__)
