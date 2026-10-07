"""使用真实 DataHub 的角色场景验收夹具；本轮未执行。"""
from datetime import date
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from dataprep import BarFileKey, BarReadSpec, InputError, InputSession
from dataprep.references import load_roll_anchor_closes
from dataprep.scenarios.role_futures import prepare_role_research


class RoleSceneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bars = self.root / "bars"
        self.bars.mkdir()
        self.roles = self.root / "roles.feather"

    def test_main_only_ignores_unused_recent_second_and_far(self):
        pd.DataFrame([dict(trade_date="2026-09-10", code="RB", main="rb2610")]).to_feather(self.roles)
        pd.DataFrame([dict(datetime="2026-09-11 09:30", close=100)]).to_feather(
            self.bars / "rb2610_20260911.feather")
        with InputSession():
            loaded = prepare_role_research(bars_dir=self.bars, contract_struct_path=self.roles,
                products=("RB",), signal_role="main", execution_product="RB", execution_role="main",
                end_day=date(2026, 9, 11))
        snapshot = loaded.store.snapshot(pd.Timestamp("2026-09-11 09:30", tz="Asia/Shanghai").value)
        self.assertEqual(snapshot.instrument("RB", "main"), "rb2610")
        self.assertEqual(set(snapshot.contracts["RB"]), {"main"})

    def test_delayed_roll_anchor_cannot_enter_precomputed_role_factors(self):
        day = date(2026, 9, 10)
        path = self.bars / "rb2610_20260910.feather"
        pd.DataFrame([dict(datetime="2026-09-10 15:00", close=100,
            published="2026-09-11 09:31")]).to_feather(path)
        spec = BarReadSpec(required_fields=("close",), value_policy="close_strict", available_column="published")
        with InputSession(), self.assertRaisesRegex(InputError, "UNSUPPORTED_CAPABILITY"):
            load_roll_anchor_closes({BarFileKey("future", "RB2610", day): path}, spec)

    def test_external_factors_do_not_require_internal_roll_anchors(self):
        # 真实换约场景，缺少新旧合约同日锚定价格。
        pd.DataFrame([
            dict(trade_date="2026-09-09", code="RB", main="rb2609"),
            dict(trade_date="2026-09-10", code="RB", main="rb2610"),
        ]).to_feather(self.roles)
        for label, contract in (("20260910", "rb2609"), ("20260911", "rb2610")):
            day_text = f"{label[:4]}-{label[4:6]}-{label[6:]}"
            pd.DataFrame([dict(datetime=day_text + " 09:30", close=100)]).to_feather(
                self.bars / f"{contract}_{label}.feather")
        factors = self.root / "factors.feather"
        pd.DataFrame([
            dict(trade_date="2026-09-09", code="RB", symbol="rb2609", pcr_cumfactor="0.8", available_ns=1),
            dict(trade_date="2026-09-10", code="RB", symbol="rb2610", pcr_cumfactor="0.9", available_ns=1),
        ]).to_feather(factors)
        with InputSession():
            loaded = prepare_role_research(bars_dir=self.bars, contract_struct_path=self.roles,
                products=("RB",), end_day=date(2026, 9, 11), factors_path=factors,
                factor_date_basis="source")
        snapshot = loaded.store.snapshot(pd.Timestamp("2026-09-11 09:30", tz="Asia/Shanghai").value)
        from decimal import Decimal
        self.assertEqual(snapshot.factor("RB", "main"), Decimal("0.9"))

    def test_generic_sector_main_signal_secondary_execution_needs_only_main_factors(self):
        from decimal import Decimal
        products = ("TA", "MA", "PP")
        pd.DataFrame([dict(trade_date="2026-09-10", code=product,
            main=product.lower() + "2609", second=product.lower() + "2701")
            for product in products]).to_feather(self.roles)
        # 主力因子合约覆盖旧主力映射，执行次主力保持原映射。
        for product in products:
            for month in ("2610", "2701"):
                pd.DataFrame([dict(datetime="2026-09-11 09:30", close=100)]).to_feather(
                    self.bars / f"{product.lower()}{month}_20260911.feather")
        factors = self.root / "factors.feather"
        pd.DataFrame([dict(trade_date="2026-09-11", code=product,
            symbol=product + "2610", pcr_cumfactor="1.2") for product in products]).to_feather(factors)
        with InputSession():
            loaded = prepare_role_research(bars_dir=self.bars, contract_struct_path=self.roles,
                products=products, signal_role="main", execution_product="PP", execution_role="secondary",
                end_day=date(2026, 9, 11), factors_path=factors, factor_date_basis="trading",
                factor_availability="aligned")
        snapshot = loaded.store.snapshot(pd.Timestamp("2026-09-11 09:30", tz="Asia/Shanghai").value)
        for product in products:
            self.assertEqual(snapshot.instrument(product, "main"), product.lower() + "2610")
            self.assertEqual(snapshot.factor(product, "main"), Decimal("1.2"))
        self.assertEqual(snapshot.instrument("PP", "secondary"), "pp2701")
        self.assertEqual(set(snapshot.cumulative_factors["PP"]), {"main"})


if __name__ == "__main__":
    unittest.main()
