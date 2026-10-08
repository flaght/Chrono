"""参考数据源的无网络验证；数据库会话使用假SDK，保留时间与交易边界。"""
from datetime import date
from decimal import Decimal
from importlib import import_module
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from bomber.framework.datahub.option_basic import OptionBasic
from bomber.framework.datahub.sector_roles import SectorDataUnavailable
from bomber.framework.dataprep.contracts import InputError
from bomber.framework.dataprep.live_references import LiveFuturesReferences
from bomber.framework.dataprep.sources import (
    DolphinDbReferenceConfig, ReferenceDataset as Dataset, ReferenceQuery,
    ReferenceSourceError, ReferenceSourceFactory)
from bomber.framework.dataprep.sources.normalize import normalize_frame

DAY = date(2026, 9, 22)
BASE = int(pd.Timestamp("2026-09-22T09:00:00+08:00").value)


def tables():
    return {
        "fut_basic": pd.DataFrame([
            {"date": "2026-01-01", "Code": symbol, "contractObject": "RB", "exchangeCD": "XSGE",
             "minChgPriceNum": 1, "contMultNum": 10, "lastTradeDate": "2027-04-30"}
            for symbol in ("rb2703", "rb2704")]),
        "fut_adjustment_factors": pd.DataFrame([{"date": "2026-09-22", "Code": "RB2704",
            "contractObject": "RB", "pcr_factor": "1.01", "pcr_cumfactor": "2"}]),
        "fut_contract": pd.DataFrame([{"date": "2026-09-21", "Code": "RB", "main": "rb2703",
            "second": "rb2703", "recent": "rb2703", "far": "rb2704"}]),
        "opt_basic": pd.DataFrame([{"date": "2026-09-01", "Code": "MO2610-C-6500",
            "exchangeCD": "CCFX", "currencyCD": "CNY", "contractType": "CO", "strikePrice": 6500,
            "contMultNum": 100, "varTicker": "000852", "lastTradeDate": "2026-10-16",
            "expDate": "2026-10-16", "tickNum": "0.2"}]),
    }


class FakeSession:
    def __init__(self, data=None, **kwargs):
        self.data = tables() if data is None else data
        self.calls = []
        self.fail = False
        self.connected = True
        self.closed = False
        self.on_read = None

    def connect(self, *args, **kwargs):
        self.calls.append(("connect", kwargs))
        return self.connected

    def close(self):
        self.closed = True

    def run(self, script):
        self.calls.append(("run", script))
        if self.fail:
            raise RuntimeError("secret-password must not leak")
        table = re.search(r'loadTable\("[^"]+", "([^"]+)"\)', script)[1]
        frame = self.data[table].copy()
        if self.on_read:
            self.on_read(table)
        return frame


class ReferenceSourceTests(unittest.TestCase):
    def setUp(self):
        self.now = BASE
        self.sdk = FakeSession()
        self.config = DolphinDbReferenceConfig("fake-host", 8848, "fake-user", "secret-password")
        self.source = ReferenceSourceFactory.create("dolphindb", self.config,
            session_factory=lambda **kwargs: self.sdk)
        self.source.open()
        self.addCleanup(self.source.close)

    def references(self, **kwargs):
        return LiveFuturesReferences(self.source, products=("RB",), trading_day=DAY,
            started_ns=BASE, clock_ns=lambda: self.now, **kwargs)

    def test_factory_is_lazy_and_placeholders_fail_explicitly(self):
        seen = []
        source = ReferenceSourceFactory.create("dolphindb", self.config,
            session_factory=lambda **kwargs: seen.append(kwargs))
        self.assertEqual(seen, [])
        with self.assertRaises(ReferenceSourceError):
            source.futures_basic(ReferenceQuery(products=("RB",)))
        for backend in ("mysql", "mongodb"):
            with self.assertRaises(NotImplementedError):
                ReferenceSourceFactory.create(backend, {}).open()
        with self.assertRaises(ValueError):
            ReferenceSourceFactory.create("unknown", {})

    def test_basic_symbol_product_and_listing_date(self):
        batch = self.source.futures_basic(ReferenceQuery(products=("RB",), active_on=DAY))
        row = batch.rows[0]
        self.assertEqual((row["symbol"], row["code"], row["listDate"]), ("RB2703", "RB", date(2026, 1, 1)))
        self.assertNotIn("Code", row)
        self.assertNotIn("date", row)
        self.assertNotIn("secret-password", repr(self.config))
        self.assertEqual(self.sdk.calls[0][1]["readTimeout"], 15)

    def test_factor_and_contract_code_have_different_meanings(self):
        factor = self.source.adjustment_factors(ReferenceQuery(products=("RB",), start_date=DAY, end_date=DAY))
        role = self.source.contract_structure(ReferenceQuery(products=("RB",), end_date=date(2026, 9, 21)))
        self.assertEqual((factor.rows[0]["symbol"], factor.rows[0]["code"], factor.rows[0]["trade_date"]),
                         ("RB2704", "RB", DAY))
        self.assertEqual((role.rows[0]["code"], role.rows[0]["date"]), ("RB", date(2026, 9, 21)))
        self.assertNotIn("symbol", role.rows[0])

    def test_option_without_contract_object_keeps_identity(self):
        batch = self.source.options_basic(ReferenceQuery(products=("MO",), active_on=DAY))
        row = batch.rows[0]
        self.assertNotIn("code", row)
        info = OptionBasic.from_mapping(row)
        self.assertEqual((info.symbol, info.list_date, info.option_kind, info.price_increment),
                         ("MO2610-C-6500", date(2026, 9, 1), "CALL", Decimal("0.2")))
        self.assertEqual(info.underlying, "000852")

    def test_alias_conflicts_and_duplicate_columns_are_rejected(self):
        frame = self.sdk.data["fut_basic"].assign(symbol="RB9999")
        with self.assertRaisesRegex(ValueError, "冲突"):
            normalize_frame(frame, Dataset.FUTURES_BASIC, ReferenceQuery(products=("RB",)), source="fake")
        frame = self.sdk.data["fut_basic"].copy()
        frame.columns = ["Code"] * len(frame.columns)
        with self.assertRaises(ValueError):
            normalize_frame(frame, Dataset.FUTURES_BASIC, ReferenceQuery(products=("RB",)), source="fake")

    def test_date_and_symbol_filters_are_enforced_on_result(self):
        old = self.sdk.data["fut_adjustment_factors"].assign(date="2026-09-21")
        self.sdk.data["fut_adjustment_factors"] = old
        batch = self.source.adjustment_factors(ReferenceQuery(products=("RB",), start_date=DAY, end_date=DAY))
        self.assertFalse(batch.rows)
        self.assertFalse(self.source.futures_basic(ReferenceQuery(symbols=("rb9999",))).rows)

    def test_unsafe_names_and_queries_are_rejected(self):
        with self.assertRaises(ValueError):
            DolphinDbReferenceConfig("fake", 8848, "u", "p", factors_table='x");delete y')
        with self.assertRaises(ValueError):
            ReferenceQuery(products=('RB"',))
        with self.assertRaises(ValueError):
            ReferenceQuery(symbols=("rb2704;delete",))
        with self.assertRaises(ValueError):
            ReferenceQuery()

    def test_query_is_bounded_readonly_and_truncation_fails(self):
        self.source.adjustment_factors(ReferenceQuery(products=("RB",), start_date=DAY, end_date=DAY))
        script = self.sdk.calls[-1][1]
        self.assertIn('upper(contractObject) in ["RB"]', script)
        self.assertIn("date >= 2026.09.22", script)
        self.assertNotIn("delete", script.lower())
        source = ReferenceSourceFactory.create("dolphindb",
            DolphinDbReferenceConfig("fake", 8848, "u", "p", max_rows=1),
            session_factory=lambda **kwargs: self.sdk)
        source.open()
        self.addCleanup(source.close)
        with self.assertRaisesRegex(ReferenceSourceError, "行数上限"):
            source.futures_basic(ReferenceQuery(products=("RB",)))

    def test_connect_and_query_errors_do_not_leak_credentials(self):
        self.sdk.fail = True
        with self.assertRaises(ReferenceSourceError) as caught:
            self.source.adjustment_factors(ReferenceQuery(products=("RB",)))
        self.assertNotIn("secret-password", str(caught.exception))
        sdk = FakeSession()
        sdk.connected = False
        source = ReferenceSourceFactory.create("dolphindb", self.config, session_factory=lambda **kwargs: sdk)
        with self.assertRaises(ReferenceSourceError):
            source.open()
        self.assertTrue(sdk.closed)

    def test_main_factor_override_does_not_change_execution_secondary(self):
        references = self.references(roles=("main", "secondary"))
        snapshot = references.snapshot(BASE)
        self.assertEqual(snapshot.instrument("RB", "main"), "rb2704")
        self.assertEqual(snapshot.instrument("RB", "secondary"), "rb2703")
        self.assertEqual(snapshot.factor("RB", "main"), Decimal(2))
        with self.assertRaises(SectorDataUnavailable):
            snapshot.factor("RB", "secondary")

    def test_multiple_products_use_one_complete_source_day(self):
        self.sdk.data["fut_contract"] = pd.concat([self.sdk.data["fut_contract"],
            pd.DataFrame([{"date": "2026-09-21", "Code": "JM", "main": "jm2701"}])])
        self.sdk.data["fut_adjustment_factors"] = pd.concat([self.sdk.data["fut_adjustment_factors"],
            pd.DataFrame([{"date": "2026-09-22", "Code": "JM2701", "contractObject": "JM", "pcr_cumfactor": "0.8"}])])
        self.sdk.data["fut_basic"] = pd.concat([self.sdk.data["fut_basic"],
            pd.DataFrame([{"date": "2026-01-01", "Code": "jm2701", "contractObject": "JM",
                "exchangeCD": "XDCE", "minChgPriceNum": "0.5", "contMultNum": 60,
                "lastTradeDate": "2027-01-31"}])])
        references = LiveFuturesReferences(self.source, products=("RB", "JM"), trading_day=DAY,
            started_ns=BASE, clock_ns=lambda: self.now)
        snapshot = references.snapshot(BASE)
        self.assertEqual(snapshot.instrument("JM", "main"), "jm2701")
        self.assertEqual(snapshot.factor("JM", "main"), Decimal("0.8"))
        self.assertEqual(references.instrument_specs["JM"]["main"].venue, "DCE")
        self.sdk.data["fut_contract"].loc[self.sdk.data["fut_contract"].Code.eq("JM"), "date"] = "2026-09-18"
        with self.assertRaisesRegex(SectorDataUnavailable, "来源日不一致"):
            references.refresh(force=True)

    def test_invalid_execution_metadata_is_rejected(self):
        self.sdk.data["fut_basic"]["minChgPriceNum"] = 0
        with self.assertRaises(ValueError):
            self.references()

    def test_missing_current_factor_cannot_use_previous_day(self):
        self.sdk.data["fut_adjustment_factors"]["date"] = "2026-09-21"
        with self.assertRaisesRegex(SectorDataUnavailable, "缺少本次交易日"):
            self.references()

    def test_same_day_role_is_not_used_as_previous_source(self):
        self.sdk.data["fut_contract"]["date"] = "2026-09-22"
        with self.assertRaises(InputError):
            self.references()

    def test_future_publication_blocks_until_visible(self):
        self.sdk.data["fut_adjustment_factors"]["available_ns"] = BASE + 10
        references = self.references()
        with self.assertRaises(SectorDataUnavailable):
            references.snapshot(BASE)
        self.now += 10
        self.assertEqual(references.snapshot(self.now).factor("RB", "main"), Decimal(2))

    def test_invalid_or_conflicting_factor_is_rejected(self):
        self.sdk.data["fut_adjustment_factors"]["pcr_cumfactor"] = "0"
        with self.assertRaises(InputError):
            self.references()
        self.sdk.data = tables()
        frame = self.sdk.data["fut_adjustment_factors"]
        self.sdk.data["fut_adjustment_factors"] = pd.concat([frame, frame.assign(pcr_cumfactor="3")])
        with self.assertRaises(InputError):
            self.references()

    def test_explicit_policy_does_not_guess_publication(self):
        with self.assertRaises(InputError):
            self.references(factor_availability="explicit")

    def test_read_failure_blocks_cached_snapshot_until_successful_refresh(self):
        references = self.references()
        self.now += 5_000_000_000
        self.sdk.fail = True
        with self.assertRaises(ReferenceSourceError):
            references.snapshot(self.now)
        count = len(self.sdk.calls)
        with self.assertRaises(SectorDataUnavailable):
            references.snapshot(self.now)
        self.assertEqual(len(self.sdk.calls), count)
        self.assertFalse(references.manifest["ready"])
        self.sdk.fail = False
        self.now += 5_000_000_000
        self.assertEqual(references.snapshot(self.now).factor("RB", "main"), Decimal(2))

    def test_update_cannot_be_visible_to_older_event(self):
        references = self.references()
        self.now += 5_000_000_000
        self.sdk.data["fut_adjustment_factors"]["pcr_cumfactor"] = "3"
        with self.assertRaises(SectorDataUnavailable):
            references.snapshot(BASE)
        self.assertEqual(references.snapshot(self.now).factor("RB", "main"), Decimal(3))

    def test_mid_read_update_is_rejected(self):
        seen = []
        def mutate(table):
            if table == "fut_adjustment_factors" and not seen:
                seen.append(table)
                self.sdk.data[table]["pcr_cumfactor"] = "3"
        self.sdk.on_read = mutate
        with self.assertRaisesRegex(SectorDataUnavailable, "读取期间"):
            self.references()

    def test_database_and_existing_file_entry_have_same_snapshot_and_metadata(self):
        refs = import_module("bomber.framework.dataprep.live_role")
        with TemporaryDirectory() as directory:
            paths = {"roles": Path(directory) / "roles.feather",
                     "factors": Path(directory) / "factors.feather", "basic": Path(directory) / "basic.feather"}
            self.sdk.data["fut_contract"].rename(columns={"date": "trade_date", "Code": "code"}).to_feather(paths["roles"])
            self.sdk.data["fut_adjustment_factors"].rename(columns={"date": "trade_date", "Code": "symbol",
                                                                 "contractObject": "code"}).to_feather(paths["factors"])
            self.sdk.data["fut_basic"].rename(columns={"date": "listDate", "Code": "symbol",
                                                      "contractObject": "code"}).to_feather(paths["basic"])
            old = refs.FileRoleReferences(product="RB", trading_day="20260922", started_ns=BASE,
                contract_struct=paths["roles"], factors=paths["factors"], fut_basic=paths["basic"])
            new = refs.SourceRoleReferences(self.source, product="RB", trading_day="20260922",
                started_ns=BASE, factor_date_basis="trading", clock_ns=lambda: self.now)
            self.assertEqual(old.snapshot(BASE), new.snapshot(BASE))
            self.assertEqual(old.spec, new.spec)
            self.assertEqual(old.instrument_id, new.instrument_id)
            self.assertEqual(old.instrument_meta(), new.instrument_meta())
            module = import_module("demos.01_main_ema.strategy")
            from bomber.framework.market.basic.base import make_bar
            traces = []
            for provider in (old, new):
                strategy = module.MainEmaStrategy("same-ema", provider,
                    module.MainEmaConfig("RB", "SHFE", fast_period=2, slow_period=3))
                with patch.object(strategy, "set_target") as submit:
                    for minute, close in enumerate((3000, 3010, 3020, 2980, 2970, 3030), 1):
                        self.now = BASE + minute * 60_000_000_000
                        bar = make_bar(new.instrument_id, close, close, close, close, 1,
                            self.now, meta=new.instrument_meta())
                        strategy.on_bar("main", bar)
                    traces.append(submit.call_args_list)
            self.assertGreaterEqual(len(traces[0]), 2)
            self.assertEqual(traces[0], traces[1])

    def test_database_adapter_defaults_to_role_source_day(self):
        refs = import_module("bomber.framework.dataprep.live_role")
        prior = self.sdk.data["fut_adjustment_factors"].assign(date="2026-09-21")
        current = self.sdk.data["fut_adjustment_factors"].assign(pcr_cumfactor="99", Code="RB2703")
        self.sdk.data["fut_adjustment_factors"] = pd.concat([prior, current])
        references = refs.SourceRoleReferences(self.source, product="RB", trading_day="20260922",
            started_ns=BASE, clock_ns=lambda: self.now)
        snapshot = references.snapshot(BASE)
        self.assertEqual(snapshot.trading_day, DAY)
        self.assertEqual(snapshot.source_day, date(2026, 9, 21))
        self.assertEqual(snapshot.instrument("RB", "main"), "rb2704")
        self.assertEqual(snapshot.factor("RB", "main"), Decimal(2))
        self.assertEqual(references.factor_date, date(2026, 9, 21))
        self.assertEqual(references.manifest["factor_date_basis"], "source")

    def test_public_role_adapters_support_secondary_file_and_source(self):
        refs = import_module("bomber.framework.dataprep.live_role")
        self.sdk.data["fut_adjustment_factors"] = self.sdk.data["fut_adjustment_factors"].assign(
            date="2026-09-21", role="secondary")
        source = refs.SourceRoleReferences(self.source, product="RB", trading_day="20260922",
            role="secondary", started_ns=BASE, clock_ns=lambda: self.now)
        with TemporaryDirectory() as directory:
            paths = [Path(directory) / f"{name}.feather" for name in ("roles", "factors", "basic")]
            pd.DataFrame([{"trade_date": "2026-09-21", "code": "RB", "second": "rb2703"}]).to_feather(paths[0])
            pd.DataFrame([{"trade_date": "2026-09-21", "code": "RB", "symbol": "rb2704",
                "role": "secondary", "pcr_cumfactor": "2"}]).to_feather(paths[1])
            pd.DataFrame([{"symbol": "rb2704", "code": "RB", "exchangeCD": "XSGE",
                "contMultNum": 10, "minChgPriceNum": 1, "listDate": "2026-01-01",
                "lastTradeDate": "2027-04-30"}]).to_feather(paths[2])
            file = refs.FileRoleReferences(product="RB", trading_day="20260922", role="secondary",
                contract_struct=paths[0], factors=paths[1], fut_basic=paths[2], started_ns=BASE,
                factor_date_basis="source")
            for references in (file, source):
                snapshot = references.snapshot(BASE)
                self.assertEqual(snapshot.instrument("RB", "secondary"), "rb2704")
                self.assertEqual(snapshot.factor("RB", "secondary"), Decimal(2))
                self.assertEqual(str(references.instrument_id), "rb2704.SHFE")
                with self.assertRaises(SectorDataUnavailable):
                    snapshot.factor("RB", "main")
            self.assertEqual(file.spec, source.spec)
        with self.assertRaisesRegex(ValueError, "交易所"):
            refs.SourceRoleReferences(self.source, product="RB", trading_day="20260922",
                role="secondary", started_ns=BASE, clock_ns=lambda: self.now,
                allowed_venues={"INE"})

    def test_source_day_cross_date_gap_uses_role_day_not_calendar_yesterday(self):
        self.sdk.data["fut_contract"]["date"] = "2026-09-30"
        self.sdk.data["fut_adjustment_factors"]["date"] = "2026-09-30"
        now = int(pd.Timestamp("2026-10-08T09:00:00+08:00").value)
        references = LiveFuturesReferences(self.source, products=("RB",), trading_day="20261008",
            started_ns=now, factor_date_basis="source", clock_ns=lambda: now)
        snapshot = references.snapshot(now)
        self.assertEqual(snapshot.trading_day, date(2026, 10, 8))
        self.assertEqual(snapshot.source_day, date(2026, 9, 30))
        self.assertEqual(snapshot.factor("RB", "main"), Decimal(2))
        self.assertEqual(references.manifest["factor_date"], "2026-09-30")

    def test_source_day_missing_cannot_fall_back_to_older_factor(self):
        self.sdk.data["fut_adjustment_factors"]["date"] = "2026-09-18"
        with self.assertRaisesRegex(SectorDataUnavailable, "缺少角色来源日"):
            self.references(factor_date_basis="source")

    def test_source_day_cannot_use_current_day_factor(self):
        with self.assertRaisesRegex(SectorDataUnavailable, "缺少角色来源日"):
            self.references(factor_date_basis="source")

    def test_source_day_explicit_publication_overrides_day_end_assumption(self):
        self.sdk.data["fut_adjustment_factors"]["date"] = "2026-09-21"
        self.sdk.data["fut_adjustment_factors"]["available_ns"] = BASE + 10
        references = self.references(factor_date_basis="source")
        with self.assertRaises(SectorDataUnavailable):
            references.snapshot(BASE)
        self.now += 10
        self.assertEqual(references.snapshot(self.now).factor("RB", "main"), Decimal(2))
        self.assertEqual(references.manifest["factor_publication_policy"], "explicit")

    def test_observed_source_day_is_available_during_night_session_without_backdating(self):
        self.sdk.data["fut_contract"]["date"] = "2026-10-08"
        self.sdk.data["fut_adjustment_factors"]["date"] = "2026-10-08"
        self.now = int(pd.Timestamp("2026-10-08T22:00:00+08:00").value)
        references = LiveFuturesReferences(self.source, products=("RB",), trading_day="20261009",
            started_ns=self.now - 1, clock_ns=lambda: self.now,
            factor_date_basis="source", factor_availability="observed-on-read",
            refresh_seconds=0)
        first = references.snapshot(self.now)
        self.assertEqual(first.source_day, date(2026, 10, 8))
        self.assertEqual(references.manifest["first_observed_ns"], self.now)
        with self.assertRaises(SectorDataUnavailable):
            references.snapshot(self.now - 1)
        self.now += 1
        self.assertEqual(references.snapshot(self.now).available_ns, first.available_ns)
        self.assertEqual(references.manifest["factor_publication_policy"], "observed-on-read")
        self.sdk.data["fut_adjustment_factors"]["pcr_cumfactor"] = "3"
        self.now += 1
        references.refresh(force=True)
        with self.assertRaises(SectorDataUnavailable):
            references.snapshot(self.now - 1)
        self.assertEqual(references.snapshot(self.now).factor("RB", "main"), Decimal(3))
        self.assertEqual(references.manifest["first_observed_ns"], self.now)

    def test_observed_source_day_still_respects_future_explicit_publication(self):
        self.sdk.data["fut_contract"]["date"] = "2026-10-08"
        self.sdk.data["fut_adjustment_factors"]["date"] = "2026-10-08"
        self.now = int(pd.Timestamp("2026-10-08T22:00:00+08:00").value)
        self.sdk.data["fut_adjustment_factors"]["available_ns"] = self.now + 10
        references = LiveFuturesReferences(self.source, products=("RB",), trading_day="20261009",
            started_ns=self.now, clock_ns=lambda: self.now,
            factor_date_basis="source", factor_availability="observed-on-read")
        with self.assertRaises(SectorDataUnavailable):
            references.snapshot(self.now)
        self.now += 10
        self.assertEqual(references.snapshot(self.now).factor("RB", "main"), Decimal(2))

    def test_source_day_file_and_database_share_explicit_source_policy(self):
        refs = import_module("bomber.framework.dataprep.live_role")
        self.sdk.data["fut_adjustment_factors"]["date"] = "2026-09-21"
        with TemporaryDirectory() as directory:
            paths = [Path(directory) / f"{name}.feather" for name in ("roles", "factors", "basic")]
            self.sdk.data["fut_contract"].rename(columns={"date": "trade_date", "Code": "code"}).to_feather(paths[0])
            self.sdk.data["fut_adjustment_factors"].rename(columns={"date": "trade_date", "Code": "symbol",
                "contractObject": "code"}).to_feather(paths[1])
            self.sdk.data["fut_basic"].rename(columns={"date": "listDate", "Code": "symbol",
                "contractObject": "code"}).to_feather(paths[2])
            file = refs.FileRoleReferences(product="RB", trading_day="20260922", started_ns=BASE,
                contract_struct=paths[0], factors=paths[1], fut_basic=paths[2], factor_date_basis="source")
            database = refs.SourceRoleReferences(self.source, product="RB", trading_day="20260922",
                started_ns=BASE, clock_ns=lambda: self.now)
            self.assertEqual(file.snapshot(BASE), database.snapshot(BASE))
            self.assertEqual(file.spec, database.spec)
            self.assertEqual(file.factor_date, database.factor_date)

    def test_future_basic_publication_also_blocks_snapshot(self):
        self.sdk.data["fut_basic"]["available_ns"] = BASE + 100
        self.sdk.data["fut_basic"]["source_version"] = "basic-v2"
        references = self.references()
        with self.assertRaises(SectorDataUnavailable):
            references.snapshot(BASE)
        self.now += 100
        self.assertEqual(references.snapshot(self.now).instrument("RB", "main"), "rb2704")

    def test_file_source_and_database_share_normalized_fields(self):
        paths = {Dataset.FUTURES_BASIC: "fut_basic.feather", Dataset.CONTRACT_STRUCTURE: "fut_contract.feather"}
        source = ReferenceSourceFactory.create("file", paths)
        with source, patch("bomber.framework.dataprep.sources.file.read_feather",
            side_effect=lambda path: self.sdk.data[path.stem].copy()):
            for dataset in paths:
                query = ReferenceQuery(products=("RB",), end_date=DAY)
                self.assertEqual(source.read(dataset, query).fingerprint, self.source.read(dataset, query).fingerprint)


if __name__ == "__main__":
    unittest.main(verbosity=2)
