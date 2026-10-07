"""外部因子回归夹具；仅生成代码，请在运行环境执行。"""
from datetime import date
from decimal import Decimal
from unittest import TestCase, main
from unittest.mock import patch

import pandas as pd

from datahub.role_prices import RoleAssignment
from dataprep import InputError, InputSession
from dataprep.factors import load_cumulative_factors, build_external_sector_store


class ExternalFactorTests(TestCase):
    def records(self, **extra):
        row = dict(trade_date="2026-01-02", code="RB", symbol="rb2605",
                   pcr_factor="0.99", pcr_cumfactor="0.85", available_ns=1)
        row.update(extra)
        with patch("dataprep.factors.read_feather", return_value=pd.DataFrame([row])):
            return load_cumulative_factors("factors.feather", date_basis="source")

    def build(self, factors, role="main"):
        day = date(2026, 1, 5)
        assignment = RoleAssignment(day, date(2026, 1, 2), 10, 10,
                                    {role: "rb2605"}, roles=(role,))
        return build_external_sector_store({"RB": {day: assignment}}, {day: 10},
            signal_role=role, factors=factors, date_basis="source").snapshot(10)

    def test_uses_supplied_cumulative_not_single_or_initial_one(self):
        self.assertEqual(self.build(self.records()).factor("RB", "main"), Decimal("0.85"))

    def test_explicit_secondary_is_supported(self):
        self.assertEqual(self.build(self.records(role="secondary"), "secondary").factor(
            "RB", "secondary"), Decimal("0.85"))

    def test_main_factor_cannot_be_applied_to_secondary(self):
        with self.assertRaisesRegex(InputError, "MISSING_FACTOR"):
            self.build(self.records(), "secondary")

    def test_factor_symbol_overrides_role_and_records_conflict(self):
        with InputSession() as session:
            snapshot = self.build(self.records(symbol="rb2610"))
            self.assertEqual(snapshot.instrument("RB", "main"), "rb2610")
            self.assertEqual(snapshot.factor("RB", "main"), Decimal("0.85"))
            self.assertEqual(session.issues[0].code, "FACTOR_CONTRACT_OVERRIDE")
            self.assertEqual(session.issues[0].severity, "WARNING")

    def test_execution_main_is_overridden_without_changing_secondary(self):
        day, source = date(2026, 4, 3), date(2026, 4, 2)
        row = RoleAssignment(day, source, 10, 10,
            {"main": "rb2605", "secondary": "rb2610"}, roles=("main", "secondary"))
        factors = {
            (day, "RB", "main"): ("rb2610", Decimal("1.201347"), 0),
            (day, "RB", "secondary"): ("rb2610", Decimal("0.9"), 0),
        }
        snapshot = build_external_sector_store({"RB": {day: row}}, {day: 10},
            signal_role="secondary", factors=factors, date_basis="trading").snapshot(10)
        self.assertEqual(snapshot.instrument("RB", "main"), "rb2610")
        self.assertEqual(snapshot.instrument("RB", "secondary"), "rb2610")
        self.assertEqual(snapshot.factor("RB", "secondary"), Decimal("0.9"))

    def test_override_does_not_allow_future_factor_publication(self):
        with self.assertRaisesRegex(InputError, "REFERENCE_NOT_AVAILABLE"):
            self.build(self.records(symbol="rb2610", available_ns=11))

    def test_future_publication_is_rejected(self):
        with self.assertRaisesRegex(InputError, "REFERENCE_NOT_AVAILABLE"):
            self.build(self.records(available_ns=11))

    def test_invalid_factor_is_rejected(self):
        with self.assertRaisesRegex(InputError, "INVALID_FACTOR"):
            self.records(pcr_cumfactor="0")

    def test_missing_publication_requires_explicit_assumption(self):
        frame = pd.DataFrame([dict(trade_date="2026-01-02", code="RB", symbol="rb2605",
                                   pcr_cumfactor="0.85")])
        with patch("dataprep.factors.read_feather", return_value=frame):
            with self.assertRaisesRegex(InputError, "MISSING_FIELD"):
                load_cumulative_factors("factors.feather", date_basis="source")
            records = load_cumulative_factors("factors.feather", date_basis="source",
                                              availability="source-day-end")
        self.assertGreater(next(iter(records.values()))[2], 0)

    def test_aligned_uses_current_day_factor_without_publication_column(self):
        day = date(2026, 1, 5)
        frame = pd.DataFrame([dict(trade_date=str(day), code="RB", symbol="rb2605",
                                   pcr_cumfactor="0.85")])
        with patch("dataprep.factors.read_feather", return_value=frame):
            factors = load_cumulative_factors("factors.feather", date_basis="trading",
                                              availability="aligned")
        assignment = RoleAssignment(day, date(2026, 1, 2), 10, 10,
                                    {"main": "rb2605"}, roles=("main",))
        snapshot = build_external_sector_store({"RB": {day: assignment}}, {day: 10},
            signal_role="main", factors=factors, date_basis="trading").snapshot(10)
        self.assertEqual(snapshot.factor("RB", "main"), Decimal("0.85"))
        self.assertEqual(snapshot.available_ns, 10)

    def test_aligned_still_honors_explicit_publication(self):
        frame = pd.DataFrame([dict(trade_date="2026-01-02", code="RB", symbol="rb2605",
                                   pcr_cumfactor="0.85", available_ns=11)])
        with patch("dataprep.factors.read_feather", return_value=frame):
            factors = load_cumulative_factors("factors.feather", date_basis="trading",
                                              availability="aligned")
        with self.assertRaisesRegex(InputError, "REFERENCE_NOT_AVAILABLE"):
            self.build(factors)

    def test_requested_scope_ignores_unrelated_bad_factors(self):
        day = date(2026, 1, 5)
        frame = pd.DataFrame([
            dict(trade_date=str(day), code="RU", symbol="ru2605", pcr_cumfactor=float("nan")),
            dict(trade_date="2025-12-15", code="RB", symbol="rb2605", pcr_cumfactor=0),
            dict(trade_date=str(day), code="RB", symbol="rb2605", role="secondary", pcr_cumfactor=-1),
            dict(trade_date=str(day), code="RB", symbol="rb2605", role="main", pcr_cumfactor="0.85"),
        ])
        # 长表包含 role 列，因此每行都必须显式声明角色。
        frame["role"] = frame.role.fillna("main")
        with patch("dataprep.factors.read_feather", return_value=frame):
            factors = load_cumulative_factors("factors.feather", date_basis="trading",
                availability="aligned", required_keys={(day, "RB", "main")})
        self.assertEqual(set(factors), {(day, "RB", "main")})
        self.assertEqual(factors[(day, "RB", "main")][1], Decimal("0.85"))

    def test_requested_bad_factor_reports_identity_and_value(self):
        day = date(2026, 1, 5)
        frame = pd.DataFrame([dict(trade_date=str(day), code="RB", symbol="rb2605", pcr_cumfactor=0)])
        with patch("dataprep.factors.read_feather", return_value=frame):
            with self.assertRaises(InputError) as caught:
                load_cumulative_factors("factors.feather", date_basis="trading",
                    availability="aligned", required_keys={(day, "RB", "main")})
        self.assertIn("rb2605", str(caught.exception))
        self.assertIn("pcr_cumfactor=0", str(caught.exception))
        self.assertEqual(caught.exception.issue.row, 1)

    def test_missing_requested_factor_is_not_replaced(self):
        day = date(2026, 1, 5)
        frame = pd.DataFrame([dict(trade_date=str(day), code="RU", symbol="ru2605", pcr_cumfactor=1)])
        with patch("dataprep.factors.read_feather", return_value=frame):
            factors = load_cumulative_factors("factors.feather", date_basis="trading",
                availability="aligned", required_keys={(day, "RB", "main")})
        assignment = RoleAssignment(day, date(2026, 1, 2), 10, 10,
                                    {"main": "rb2605"}, roles=("main",))
        with self.assertRaisesRegex(InputError, "MISSING_FACTOR"):
            build_external_sector_store({"RB": {day: assignment}}, {day: 10},
                signal_role="main", factors=factors, date_basis="trading")


if __name__ == "__main__":
    main()
