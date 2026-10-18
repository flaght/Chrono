#!/usr/bin/env python3

from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path


SCRIPT = Path(__file__).with_name("run_evaluation.py")
SPEC = importlib.util.spec_from_file_location("run_evaluation", SCRIPT)
assert SPEC and SPEC.loader
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class EvaluationTests(unittest.TestCase):
    def write_csv(self, path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["trade_time", "code", "factor", "forward_return"]
            )
            writer.writeheader()
            writer.writerows(rows)

    def test_time_series_metrics(self):
        rows = [
            {"trade_time": f"2026-01-{day:02d}", "code": "A", "factor": value,
             "future_return": value * 0.1}
            for day, value in enumerate([1, 2, 3, 4, 5, 6], 1)
        ]
        metrics = evaluation.evaluate_time_series(rows, roll_win=3, min_periods=2)
        self.assertAlmostEqual(metrics["total_ic"], 1.0)
        self.assertAlmostEqual(metrics["ic_mean"], 1.0)
        self.assertEqual(metrics["observations"], 6)
        self.assertEqual(metrics["period_count"], 5)

    def test_time_series_rejects_multiple_codes(self):
        rows = [
            {"trade_time": "1", "code": "A", "factor": 1.0, "future_return": 1.0},
            {"trade_time": "2", "code": "B", "factor": 2.0, "future_return": 2.0},
        ]
        with self.assertRaisesRegex(evaluation.EvaluationError, "exactly one code"):
            evaluation.evaluate_time_series(rows, roll_win=2, min_periods=2)

    def test_cross_section_rank_ic(self):
        rows = []
        for time in ("1", "2", "3"):
            for code, value in (("A", 1.0), ("B", 2.0), ("C", 3.0)):
                rows.append({
                    "trade_time": time,
                    "code": code,
                    "factor": value,
                    "future_return": value * 2,
                })
        metrics = evaluation.evaluate_cross_section(rows, min_cross_section_size=2)
        self.assertAlmostEqual(metrics["ic_mean"], 1.0)
        self.assertIsNone(metrics["total_ic"])
        self.assertEqual(metrics["observations"], 9)
        self.assertEqual(metrics["period_count"], 3)

    def test_cross_section_rejects_single_code(self):
        rows = [
            {"trade_time": "1", "code": "A", "factor": 1.0, "future_return": 1.0},
            {"trade_time": "2", "code": "A", "factor": 2.0, "future_return": 2.0},
        ]
        with self.assertRaisesRegex(evaluation.EvaluationError, "at least two codes"):
            evaluation.evaluate_cross_section(rows, min_cross_section_size=2)

    def test_average_ranks_handles_ties(self):
        self.assertEqual(evaluation.average_ranks([10, 10, 20]), [1.5, 1.5, 3.0])

    def test_main_writes_cross_section_artifacts_and_pass_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "factor.csv"
            output = root / "result"
            pass_rule = root / "pass-rule.json"
            rows = []
            for time in ("1", "2", "3"):
                for code, value in (("A", 1), ("B", 2), ("C", 3)):
                    rows.append({
                        "trade_time": time,
                        "code": code,
                        "factor": value,
                        "forward_return": value,
                    })
            self.write_csv(data, rows)
            pass_rule.write_text(json.dumps({
                "metric": "abs_ic_mean", "operator": ">=", "value": 0.5
            }), encoding="utf-8")
            result = evaluation.main([
                "--input", str(data),
                "--factor-column", "factor",
                "--return-column", "forward_return",
                "--evaluation-type", "cross_section",
                "--min-observations", "2",
                "--pass-rule", str(pass_rule),
                "--output-dir", str(output),
            ])
            self.assertEqual(result, 0)
            saved = json.loads((output / "evaluation-result.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "passed")
            self.assertEqual(saved["evaluation_type"], "cross_section")
            self.assertTrue((output / "manifest.json").is_file())

    def test_factor_artifact_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            path.write_text(json.dumps({
                "status": "awaiting_human_review", "ready_for_evaluation": False
            }), encoding="utf-8")
            with self.assertRaisesRegex(evaluation.EvaluationError, "ready_for_evaluation"):
                evaluation.validate_factor_artifact(path)

    def test_resolve_factor_artifact_checks_code_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            factor = root / "factor.py"
            factor.write_text("def compute(lf):\n    return lf\n", encoding="utf-8")
            artifact = root / "validated-factor-artifact.json"
            artifact.write_text(json.dumps({
                "status": "validated",
                "ready_for_evaluation": True,
                "factor_name": "demo_factor",
                "factor_file": "factor.py",
                "factor_sha256": evaluation.sha256_file(factor),
                "required_input_fields": ["trade_time", "code", "close"],
            }), encoding="utf-8")
            loaded, resolved = evaluation.resolve_factor_artifact(artifact)
            self.assertEqual(loaded["factor_name"], "demo_factor")
            self.assertEqual(resolved, factor.resolve())
            factor.write_text("def compute(lf):\n    return None\n", encoding="utf-8")
            with self.assertRaisesRegex(evaluation.EvaluationError, "SHA256"):
                evaluation.resolve_factor_artifact(artifact)

    def test_compute_mode_requires_both_data_inputs(self):
        with self.assertRaises(SystemExit):
            evaluation.parse_args([
                "--factor-artifact", "artifact.json",
                "--market-data", "market.parquet",
                "--return-column", "forward_return",
                "--evaluation-type", "time_series",
                "--output-dir", "result",
            ])

    def test_complete_time_series_evaluator_calls_save_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module = root / "cux001.py"
            module.write_text('''
from pathlib import Path
class Series:
    def notna(self): return self
    def sum(self): return 2
class Data:
    def __getitem__(self, key): return Series()
class FactorEvaluate1:
    def __init__(self, **kwargs):
        self.name = kwargs["name"]
        self.resample_data = Data()
    def run(self, is_check=False):
        return {"total_ic": 0.1, "ic_mean": 0.2, "ic_std": 0.4, "ic_ir": 0.5}
    def plot_results(self): pass
    def save_results(self, output_dir):
        target = Path(output_dir) / self.name
        target.mkdir(parents=True)
        for name in ("performance_summary.txt", "nav.csv", "ic.csv", "turnover.csv", "evaluation_plot.png", "evaluation.xml"):
            (target / name).write_text("ok")
''', encoding="utf-8")
            rows = [
                {"trade_time": "2026-01-01", "code": "A", "factor": 1.0, "future_return": 0.1},
                {"trade_time": "2026-01-02", "code": "A", "factor": 2.0, "future_return": 0.2},
            ]
            fake_pandas = types.SimpleNamespace(DataFrame=lambda value: value)
            with mock.patch.dict("sys.modules", {"pandas": fake_pandas}):
                metrics, evaluator = evaluation.run_complete_time_series_evaluation(
                    rows, factor_name="factor", return_name="ret",
                    evaluator_path=module, roll_win=2, resampling_win=1,
                    fee=0.0003, scale_method="raw", annualization_factor=252,
                )
            self.assertEqual(metrics["ic_sharpe"], 0.5)
            self.assertEqual(metrics["period_count"], 2)
            output = root / "out"
            output.mkdir()
            evaluator.plot_results()
            evaluator.save_results(str(output))
            self.assertTrue((output / "factor" / "evaluation.xml").is_file())

    def test_period_count_supports_polars_result_storage(self):
        class Column:
            def drop_nulls(self): return self
            def len(self): return 7
        class Frame:
            def get_column(self, name):
                self.name = name
                return Column()
        evaluator = types.SimpleNamespace(
            resample_data_pl=Frame(), resample_data=None
        )
        self.assertEqual(evaluation.evaluator_period_count(evaluator), 7)


if __name__ == "__main__":
    unittest.main()
