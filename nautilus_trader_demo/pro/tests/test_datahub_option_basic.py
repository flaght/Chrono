"""期权基本信息的不可变条款、已有时间契约及角色重复价格回归。"""
from datetime import date
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

from bomber.framework.datahub import (AsOfQuery, DataHub, FutureDataError, OptionBasic, OptionBasicProvider,
                     ReferenceRecord, PublicationPolicy, ObservedClose, RoleAssignment, RolePriceStore)


def basic(**changes):
    values = dict(symbol="MO2609-C-6400", exchange="CCFX", currency="CNY", underlying="000852",
                  option_kind="CO", strike=6400, multiplier=100,
                  list_date=date(2026,6,1), last_trade_date=date(2026,9,18), expiry_date=date(2026,9,18))
    values.update(changes)
    return OptionBasic(**values)


class OptionBasicTests(unittest.TestCase):
    def test_mapping_immutability_and_no_guessed_tick(self):
        info = OptionBasic.from_mapping(dict(Code="MO2609-C-6400", exchangeCD="CCFX",
            currencyCD="CNY", varTicker="000852", contractType="CO", strikePrice=6400,
            contMultNum=100, listDate="2026-06-01", lastTradeDate="2026-09-18", expDate="2026-09-18"))
        self.assertEqual(info.option_kind, "CALL")
        self.assertEqual(info.strike, Decimal(6400))
        self.assertIsNone(info.price_increment)
        with self.assertRaises(AttributeError):
            info.multiplier = 200
        self.assertEqual(basic(option_kind="PO").option_kind, "PUT")

    def test_versions_use_existing_hub_gate(self):
        old, new = basic(), basic(multiplier=200)
        provider = OptionBasicProvider((ReferenceRecord("option_basic", old.symbol, old, 100, 100, 100),
            ReferenceRecord("option_basic", new.symbol, new, 200, 250, 200)))
        hub = DataHub({"option_basic": provider})
        self.assertEqual(provider.basic_at(old.symbol, AsOfQuery(150)).multiplier, 100)
        with self.assertRaises(FutureDataError):
            hub.get("option_basic", old.symbol, AsOfQuery(220))
        self.assertEqual(provider.basic_at(old.symbol, AsOfQuery(250)).multiplier, 200)

    def test_day_end_and_duplicate_versions(self):
        info = basic()
        record = ReferenceRecord("option_basic", info.symbol, info, 100, 100, 100,
                                 date(2026,8,13), PublicationPolicy.DAY_END)
        provider = OptionBasicProvider((record,))
        self.assertEqual(provider.basic_at(info.symbol, AsOfQuery(100, trading_day=date(2026,8,14))), info)
        from bomber.framework.datahub import DataHubUnavailable
        with self.assertRaises(DataHubUnavailable):
            provider.basic_at(info.symbol, AsOfQuery(100, trading_day=date(2026,8,13)))
        with self.assertRaises(ValueError):
            OptionBasicProvider((record, record))

    def test_invalid_metadata(self):
        for changes in (dict(multiplier=0), dict(strike="NaN"), dict(option_kind="other"),
                        dict(exchange=""), dict(last_trade_date="2026-05-01")):
            with self.assertRaises(ValueError):
                basic(**changes)
        info=basic()
        with self.assertRaises(ValueError):
            OptionBasicProvider((ReferenceRecord("wrong", info.symbol, info, 1,1,1),))

    def test_conflicting_closes_rejected_identical_deduplicated(self):
        assignment=RoleAssignment(date(2026,8,14),date(2026,8,13),100,100,{"main":"x"},roles=("main",))
        first=ObservedClose("x", date(2026,8,14),100,Decimal(10))
        conflict=ObservedClose("x", date(2026,8,14),100,Decimal(20))
        for rows in ((first,conflict),(conflict,first)):
            with self.assertRaises(ValueError):
                RolePriceStore((assignment,),rows)
        self.assertEqual(RolePriceStore((assignment,),(first,first)).snapshot(100)["main"].raw_close,Decimal(10))

    def test_same_event_cannot_belong_to_two_trading_days(self):
        day = date(2026,8,14)
        assignment = RoleAssignment(day,date(2026,8,13),100,100,{"main":"x"},roles=("main",))
        first = ObservedClose("x",day,100,Decimal(10))
        wrong_day = ObservedClose("x",date(2026,8,13),100,Decimal(10))
        later = ObservedClose("x",day,200,Decimal(11))
        for rows in ((first,later,wrong_day),(wrong_day,later,first)):
            with self.assertRaises(ValueError):
                RolePriceStore((assignment,),rows)

    def test_repeated_unsorted_closes_preserve_asof_prices_and_roll_factor(self):
        source_day, next_day = date(2026,8,13), date(2026,8,14)
        assignments = (
            RoleAssignment(source_day,date(2026,8,12),50,50,{"main":"x"},roles=("main",)),
            RoleAssignment(next_day,source_day,200,200,{"main":"y"},roles=("main",)),
        )
        early = ObservedClose("x",source_day,80,Decimal(9))
        old = ObservedClose("x",source_day,100,Decimal(10))
        new = ObservedClose("y",source_day,100,Decimal(20))
        live = ObservedClose("y",next_day,200,Decimal(22))
        baseline = RolePriceStore(assignments,(early,old,new,live))
        repeated = RolePriceStore(assignments,(live,old,early,new,old,live,new,early))
        for timestamp, expected in ((80,Decimal(9)),(100,Decimal(10)),(199,Decimal(10)),(200,Decimal(22))):
            with self.subTest(timestamp=timestamp):
                self.assertEqual(repeated.snapshot(timestamp),baseline.snapshot(timestamp))
                self.assertEqual(repeated.snapshot(timestamp)["main"].raw_close,expected)
        self.assertEqual(repeated.factor_at(200,"main"),(Decimal("0.5"),Decimal("0.5")))
        self.assertEqual(repeated.snapshot(200)["main"].adjusted_close,Decimal(11))
        conflict = ObservedClose("x",source_day,100,Decimal(12))
        for rows in ((old,live,new,conflict),(conflict,new,live,old)):
            with self.assertRaises(ValueError):
                RolePriceStore(assignments,rows)

    def test_feather_adapter_retains_source_date_without_guessing(self):
        import pandas as pd
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path=Path(folder)/"opt_basic.feather"
            pd.DataFrame([dict(Code="MO2609-C-6400", exchangeCD="CCFX", currencyCD="CNY",
                varTicker="000852", contractType="CO", strikePrice=6400, contMultNum=100,
                listDate="2026-06-01", lastTradeDate="2026-09-18", expDate="2026-09-18",
                date="2026-08-13")]).to_feather(path)
            def factory(info,row):
                self.assertEqual(row["date"],"2026-08-13")
                return ReferenceRecord("option_basic",info.symbol,info,100,150,100)
            provider=OptionBasicProvider.from_feather(path,record_factory=factory)
            with self.assertRaises(FutureDataError):
                provider.basic_at("MO2609-C-6400",AsOfQuery(120))
            self.assertEqual(provider.basic_at("MO2609-C-6400",AsOfQuery(150)).underlying,"000852")


if __name__ == "__main__":
    unittest.main()
