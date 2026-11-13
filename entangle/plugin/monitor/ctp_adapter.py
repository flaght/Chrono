# -*- coding: utf-8 -*-
"""在当前进程内为 vnpy_ctp 增加完整查询响应事件。"""

import importlib
import types

from vnpy.event import Event


EVENT_CTP_QUERY_RESPONSE = "eCtpMonitorQueryResponse"

CALLBACKS = {
    "onRspQryTradingAccount": "funds",
    "onRspQryInvestorPosition": "positions",
    "onRspQryOrder": "orders",
    "onRspQryTrade": "trades",
    "onRspQryInvestorPositionDetail": "position_details",
}


def _get_safe_original(current):
    """返回可安全直调的 Python 原回调，并兼容清理旧版 wrapper。"""
    if current is None:
        return None
    if getattr(current, "_entangle_monitor_wrapper", False):
        return current

    # 热重载时类上可能仍残留旧版 wrapper。旧版 wrapper 没有标记，
    # 但其闭包中保存了 previous；只恢复真正的 Python function。
    if getattr(current, "__module__", "") == __name__:
        freevars = getattr(current, "__code__", None)
        closure = getattr(current, "__closure__", None)
        if freevars and closure:
            values = dict(zip(freevars.co_freevars, closure))
            previous_cell = values.get("previous")
            if previous_cell:
                try:
                    previous = previous_cell.cell_contents
                except ValueError:
                    previous = None
                if (
                    isinstance(previous, types.FunctionType)
                    and getattr(previous, "__module__", "") != __name__
                ):
                    return previous
        return None

    return current if isinstance(current, types.FunctionType) else None


def install_query_callbacks(ctp_module):
    """安装查询回调；保留 vnpy 原有资金和持仓处理。"""
    gateway_class = getattr(ctp_module, "CtpGateway")
    gateway_module = importlib.import_module(gateway_class.__module__)
    td_api_class = getattr(gateway_module, "CtpTdApi")

    for callback_name, query_type in CALLBACKS.items():
        # 只能保留 CtpTdApi 类自身定义的 Python 回调。
        # getattr() 可能取得 TdApi C++ 基类的虚函数 trampoline；调用它会再次
        # 动态分派到本 wrapper，造成 callback -> previous -> callback 无限递归。
        current = td_api_class.__dict__.get(callback_name)
        if getattr(current, "_entangle_monitor_wrapper", False):
            continue
        original = _get_safe_original(current)

        def make_callback(name, table_name, previous):
            def callback(self, data, error, reqid, last):
                payload = {
                    "gateway_name": self.gateway_name,
                    "query_type": table_name,
                    "data": dict(data) if data else {},
                    "error": dict(error) if error else {},
                    "reqid": reqid,
                    "last": bool(last),
                }
                self.gateway.event_engine.put(
                    Event(EVENT_CTP_QUERY_RESPONSE, payload)
                )
                if previous:
                    return previous(self, data, error, reqid, last)
                return None
            callback.__name__ = name
            callback._entangle_monitor_wrapper = True
            callback._entangle_monitor_original = previous
            return callback

        setattr(
            td_api_class,
            callback_name,
            make_callback(
                callback_name, query_type, original
            ),
        )

    td_api_class._entangle_monitor_patched = True

