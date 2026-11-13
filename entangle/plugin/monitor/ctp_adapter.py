# -*- coding: utf-8 -*-
"""在当前进程内为 vnpy_ctp 增加完整查询响应事件。"""

import importlib

from vnpy.event import Event


EVENT_CTP_QUERY_RESPONSE = "eCtpMonitorQueryResponse"

CALLBACKS = {
    "onRspQryTradingAccount": ("funds", True),
    "onRspQryInvestorPosition": ("positions", True),
    "onRspQryOrder": ("orders", True),
    "onRspQryTrade": ("trades", True),
    "onRspQryInvestorPositionDetail": ("position_details", True),
}


def install_query_callbacks(ctp_module):
    """安装查询回调；保留 vnpy 原有资金和持仓处理。"""
    gateway_class = getattr(ctp_module, "CtpGateway")
    gateway_module = importlib.import_module(gateway_class.__module__)
    td_api_class = getattr(gateway_module, "CtpTdApi")

    if getattr(td_api_class, "_entangle_monitor_patched", False):
        return

    for callback_name, callback_config in CALLBACKS.items():
        query_type, call_original = callback_config
        original = getattr(td_api_class, callback_name, None)

        def make_callback(name, table_name, previous, preserve):
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
                if preserve and previous:
                    return previous(self, data, error, reqid, last)
                return None
            callback.__name__ = name
            return callback

        setattr(
            td_api_class,
            callback_name,
            make_callback(
                callback_name, query_type, original, call_original
            ),
        )

    td_api_class._entangle_monitor_patched = True

