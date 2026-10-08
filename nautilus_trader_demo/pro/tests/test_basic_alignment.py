"""期货与期权条款的时间门控、身份和文件准备一致性。"""
from datetime import date
from decimal import Decimal
import unittest
from unittest.mock import patch

from bomber.framework.datahub import (AsOfQuery, DataHub, DataHubUnavailable, FutureBasic, FutureBasicProvider,
                     FutureDataError, OptionBasic, OptionBasicProvider, PublicationPolicy,
                     ReferenceRecord)
from bomber.framework.dataprep.basic import load_future_basic_provider, load_option_basic_provider
from bomber.framework.dataprep.metadata import future_spec


def cases():
    future = FutureBasic("rb2605", "XSGE", "CNY", "RB", date(2025, 1, 1),
                         date(2026, 5, 15), Decimal(10), Decimal(1))
    option = OptionBasic("MO2609-C-6400", "CCFX", "CNY", "000852", "CO", Decimal(6400),
                         Decimal(100), date(2026, 6, 1), date(2026, 9, 18),
                         date(2026, 9, 18), Decimal("0.2"))
    return ((future, FutureBasicProvider, load_future_basic_provider, "future_basic"),
            (option, OptionBasicProvider, load_option_basic_provider, "option_basic"))


class BasicAlignmentTests(unittest.TestCase):
    def test_day_end_gate_and_immutable_values_for_both_assets(self):
        for value, provider_type, _, dataset in cases():
            with self.subTest(dataset=dataset):
                provider = provider_type((ReferenceRecord(
                    dataset, value.symbol, value, 10, 10, 10,
                    date(2026, 1, 5), PublicationPolicy.DAY_END),))
                with self.assertRaises(DataHubUnavailable):
                    provider.basic_at(value.symbol, AsOfQuery(10, trading_day=date(2026, 1, 5)))
                self.assertEqual(provider.basic_at(value.symbol,
                    AsOfQuery(10, trading_day=date(2026, 1, 6))), value)
                with self.assertRaises(AttributeError):
                    value.multiplier = Decimal(200)

    def test_publication_source_age_and_duplicate_gate_for_both_assets(self):
        for value, provider_type, _, dataset in cases():
            with self.subTest(dataset=dataset):
                first = ReferenceRecord(dataset, value.symbol, value, 10, 10, 10)
                newer = ReferenceRecord(dataset, value.symbol, value, 20, 30, 20)
                provider = provider_type((first, newer))
                self.assertEqual(provider.basic_at(value.symbol, AsOfQuery(15)), value)
                with self.assertRaises(FutureDataError):
                    provider.basic_at(value.symbol, AsOfQuery(25))
                self.assertEqual(DataHub({dataset: provider}).get(
                    dataset, value.symbol, AsOfQuery(30))[-1].value, value)
                with self.assertRaises(DataHubUnavailable):
                    provider.basic_at(value.symbol, AsOfQuery(40, max_source_age_ns=5))
                with self.assertRaises(ValueError):
                    provider_type((first, first))
                with self.assertRaises(ValueError):
                    provider_type((ReferenceRecord(dataset, "wrong", value, 10, 10, 10),))

    def test_file_adapters_share_reader_and_preserve_source_fields(self):
        import pandas as pd
        for value, _, loader, dataset in cases():
            row = dict(vars(value), source_date="2026-01-05")
            def make_record(info, source):
                self.assertEqual(source["source_date"], "2026-01-05")
                return ReferenceRecord(dataset, info.symbol, info, 10, 20, 10)
            with patch("bomber.framework.dataprep.basic.read_feather", return_value=pd.DataFrame([row])) as read:
                provider = loader("unused", record_factory=make_record)
                read.assert_called_once_with("unused")
                self.assertEqual(provider.basic_at(value.symbol, AsOfQuery(20)), value)

    def test_mapping_tick_priority_and_invalid_tick(self):
        option = cases()[1][0]
        raw = dict(vars(option), tickNum="0.2", minChgPriceNum="0.4", price_increment="0.6")
        self.assertEqual(OptionBasic.from_mapping(raw).price_increment, Decimal("0.2"))
        raw["tickNum"] = None
        self.assertEqual(OptionBasic.from_mapping(raw).price_increment, Decimal("0.4"))
        raw["tickNum"] = 0
        with self.assertRaises(ValueError):
            OptionBasic.from_mapping(raw)

    def test_reference_identity_and_execution_requirements(self):
        self.assertEqual(cases()[0][0].exchange, "XSGE")
        self.assertEqual(cases()[1][0].exchange, "CCFX")
        raw = dict(symbol="rb2605", code="RB", exchangeCD="XSGE",
                   listDate="2025-01-01", lastTradeDate="2026-05-15")
        self.assertIsNone(future_spec(raw, require_execution=False).tick)
        raw.update(minChgPriceNum=0, contMultNum=10)
        with self.assertRaises(ValueError):
            future_spec(raw)


if __name__ == "__main__":
    unittest.main()
