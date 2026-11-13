# -*- coding: utf-8 -*-

import threading
import unittest

from plugin.monitor.scheduler import QueryScheduler


class FakeTdApi(object):

    def __init__(self):
        self.reqid = 0
        self.login_status = True
        self.contract_inited = True
        self.scheduler = None

    def _respond(self, query_type, reqid):
        def callback():
            self.scheduler.handle_response({
                "gateway_name": "CTP_01",
                "query_type": query_type,
                "reqid": reqid,
                "data": {"row": query_type},
                "error": {},
                "last": True,
            })
        threading.Timer(0.01, callback).start()
        return 0

    def reqQryTradingAccount(self, request, reqid):
        return self._respond("funds", reqid)

    def reqQryOrder(self, request, reqid):
        return self._respond("orders", reqid)


class FakeGateway(object):

    def __init__(self):
        self.td_api = FakeTdApi()


class FakeMainEngine(object):

    def __init__(self):
        self.gateway = FakeGateway()

    def get_gateway(self, gateway_name):
        return self.gateway


class TestQueryScheduler(unittest.TestCase):

    def test_serial_queries(self):
        completed = []
        main_engine = FakeMainEngine()
        scheduler = QueryScheduler(
            main_engine,
            {"CTP_01": {"account_id": "1", "broker_id": "9999"}},
            lambda task: completed.append(task.query_type),
            min_interval=0.0,
        )
        main_engine.gateway.td_api.scheduler = scheduler
        scheduler.start()
        try:
            funds = scheduler.request("CTP_01", "funds", timeout=1.0)
            orders = scheduler.request("CTP_01", "orders", timeout=1.0)
            self.assertTrue(funds.completed.wait(2.0))
            self.assertTrue(orders.completed.wait(2.0))
            self.assertIsNone(funds.error)
            self.assertIsNone(orders.error)
            self.assertEqual(completed, ["funds", "orders"])
        finally:
            scheduler.stop()


if __name__ == "__main__":
    unittest.main()

