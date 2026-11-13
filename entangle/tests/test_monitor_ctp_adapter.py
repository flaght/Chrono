# -*- coding: utf-8 -*-

import importlib
import sys
import types
import unittest


class FakeEvent(object):

    def __init__(self, event_type, data=None):
        self.type = event_type
        self.data = data


def load_adapter():
    if "vnpy.event" not in sys.modules:
        vnpy_module = types.ModuleType("vnpy")
        event_module = types.ModuleType("vnpy.event")
        event_module.Event = FakeEvent
        sys.modules["vnpy"] = vnpy_module
        sys.modules["vnpy.event"] = event_module
    return importlib.import_module("plugin.monitor.ctp_adapter")


class FakeEventEngine(object):

    def __init__(self):
        self.events = []

    def put(self, event):
        self.events.append(event)


class TestCtpAdapter(unittest.TestCase):

    def test_inherited_trampoline_is_not_called(self):
        adapter = load_adapter()
        module_name = "tests.fake_ctp_gateway"
        gateway_module = types.ModuleType(module_name)

        class BaseTdApi(object):
            def onRspQryOrder(self, data, error, reqid, last):
                # 模拟 C++ trampoline 再次动态分派到当前覆盖方法。
                return type(self).onRspQryOrder(
                    self, data, error, reqid, last
                )

        class CtpTdApi(BaseTdApi):
            def onRspQryInvestorPosition(self, data, error, reqid, last):
                self.position_original_calls += 1

        class CtpGateway(object):
            pass

        CtpGateway.__module__ = module_name
        gateway_module.CtpGateway = CtpGateway
        gateway_module.CtpTdApi = CtpTdApi
        sys.modules[module_name] = gateway_module
        exported_module = types.SimpleNamespace(CtpGateway=CtpGateway)

        adapter.install_query_callbacks(exported_module)
        # 再安装一次也不应形成 wrapper 链。
        adapter.install_query_callbacks(exported_module)

        api = CtpTdApi()
        api.gateway_name = "CTP_01"
        api.position_original_calls = 0
        api.gateway = types.SimpleNamespace(event_engine=FakeEventEngine())

        api.onRspQryOrder({"InstrumentID": "rb"}, {}, 1, True)
        self.assertEqual(len(api.gateway.event_engine.events), 1)

        api.onRspQryInvestorPosition({}, {}, 2, True)
        self.assertEqual(api.position_original_calls, 1)
        self.assertEqual(len(api.gateway.event_engine.events), 2)

    def test_legacy_recursive_wrapper_is_replaced(self):
        adapter = load_adapter()
        module_name = "tests.fake_ctp_gateway_legacy"
        gateway_module = types.ModuleType(module_name)

        class Trampoline(object):
            def __call__(self, instance, data, error, reqid, last):
                return type(instance).onRspQryOrder(
                    instance, data, error, reqid, last
                )

        inherited = Trampoline()

        def make_legacy(previous):
            def callback(self, data, error, reqid, last):
                return previous(self, data, error, reqid, last)
            callback.__module__ = adapter.__name__
            return callback

        class CtpTdApi(object):
            onRspQryOrder = make_legacy(inherited)

        class CtpGateway(object):
            pass

        CtpGateway.__module__ = module_name
        gateway_module.CtpGateway = CtpGateway
        gateway_module.CtpTdApi = CtpTdApi
        sys.modules[module_name] = gateway_module

        adapter.install_query_callbacks(
            types.SimpleNamespace(CtpGateway=CtpGateway)
        )
        api = CtpTdApi()
        api.gateway_name = "CTP_01"
        api.gateway = types.SimpleNamespace(event_engine=FakeEventEngine())
        api.onRspQryOrder({"InstrumentID": "rb"}, {}, 1, True)
        self.assertEqual(len(api.gateway.event_engine.events), 1)


if __name__ == "__main__":
    unittest.main()

