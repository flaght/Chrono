# -*- coding: utf-8 -*-

import unittest

from plugin.monitor.schema import (
    normalize_order, normalize_position, normalize_position_detail,
    normalize_trade, row_key,
)


class TestMonitorSchema(unittest.TestCase):

    def setUp(self):
        self.config = {
            "product_id": "PRODUCT_01",
            "account_id": "80051230",
            "broker_id": "9999",
        }

    def test_order_mapping_and_key(self):
        row = normalize_order({
            "TradingDay": "20261113",
            "ExchangeID": "SHFE",
            "InstrumentID": "rb2701",
            "OrderSysID": "12345",
            "Direction": "0",
            "CombOffsetFlag": "3",
            "OrderStatus": "1",
        }, "CTP_01", self.config)
        self.assertEqual(row["DirectionName"], "买")
        self.assertEqual(row["OffsetName"], "平今")
        self.assertEqual(row["OrderStatusName"], "部分成交还在队列")
        self.assertEqual(row_key("orders", row), "CTP_01|SHFE|12345")

    def test_trade_mapping_and_key(self):
        row = normalize_trade({
            "ExchangeID": " DCE ",
            "InstrumentID": "m2701 ",
            "TradeID": "  T100\x00  ",
            "OrderSysID": "  946488  ",
            "OrderLocalID": " L100\x00 ",
            "Direction": "1",
            "OffsetFlag": "0",
        }, "CTP_01", self.config)
        self.assertEqual(row["DirectionName"], "卖")
        self.assertEqual(row["OffsetName"], "开仓")
        self.assertEqual(row["TradeID"], "T100")
        self.assertEqual(row["OrderSysID"], "946488")
        self.assertEqual(row["OrderLocalID"], "L100")
        self.assertEqual(row_key("trades", row), "CTP_01|DCE|T100")

    def test_position_detail_trade_id_is_trimmed(self):
        row = normalize_position_detail({
            "ExchangeID": " SHFE ",
            "InstrumentID": "rb2701 ",
            "TradeID": " 946488\x00 ",
            "OpenDate": " 20261113 ",
            "Direction": "2",
        }, "CTP_01", self.config)
        self.assertEqual(row["TradeID"], "946488")
        self.assertEqual(row["OpenDate"], "20261113")
        self.assertEqual(
            row_key("position_details", row),
            "CTP_01|SHFE|946488|20261113|2",
        )

    def test_position_query_and_event_share_key(self):
        queried = normalize_position({
            "ExchangeID": "SHFE",
            "InstrumentID": "rb2701",
            "PosiDirection": "2",
        }, "CTP_01", self.config)
        event_row = normalize_position({
            "ExchangeID": "SHFE",
            "InstrumentID": "rb2701",
            "PosiDirection": "多",
        }, "CTP_01", self.config)
        self.assertEqual(
            row_key("positions", queried),
            row_key("positions", event_row),
        )


if __name__ == "__main__":
    unittest.main()

