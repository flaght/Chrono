#!/usr/bin/env python3

from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("run_orchestrator.py")
SPEC = importlib.util.spec_from_file_location("run_orchestrator", SCRIPT)
assert SPEC and SPEC.loader
orchestrator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(orchestrator)


class OrchestratorTests(unittest.TestCase):
    def write_spec(self, path: Path) -> None:
        path.write_text(json.dumps({
            "factor_name": "demo_factor",
            "formula_description": "demo",
            "required_features": [{"name": "close"}],
        }), encoding="utf-8")

    def write_artifact(self, path: Path) -> None:
        path.write_text(json.dumps({
            "status": "validated",
            "ready_for_evaluation": True,
        }), encoding="utf-8")

    def test_codex_research_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = root / "request.json"
            request.write_text(json.dumps({
                "research": {"execution": "codex"},
                "development": {"execution": "codex"},
                "evaluation": {"evaluation_type": "time_series", "return_column": "ret"},
            }), encoding="utf-8")
            manifest = orchestrator.run_pipeline(request, root / "run")
            self.assertEqual(manifest["status"], "awaiting_codex_research")

    def test_existing_artifacts_complete_prepared_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = root / "factor-specification.json"
            artifact = root / "validated-factor-artifact.json"
            data = root / "prepared.csv"
            request = root / "request.json"
            self.write_spec(spec)
            self.write_artifact(artifact)
            with data.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["trade_time", "code", "factor", "ret"])
                writer.writeheader()
                for time in ("1", "2", "3"):
                    for code, value in (("A", 1), ("B", 2), ("C", 3)):
                        writer.writerow({"trade_time": time, "code": code, "factor": value, "ret": value})
            request.write_text(json.dumps({
                "research": {"execution": "codex", "specification": str(spec)},
                "development": {"execution": "codex", "validated_factor_artifact": str(artifact)},
                "evaluation": {
                    "evaluation_type": "cross_section",
                    "input": str(data),
                    "factor_column": "factor",
                    "return_column": "ret",
                    "min_observations": 2,
                },
            }), encoding="utf-8")
            run_dir = root / "run"
            manifest = orchestrator.run_pipeline(request, run_dir)
            self.assertEqual(manifest["status"], "completed")
            self.assertTrue((run_dir / "evaluation" / "evaluation-result.json").is_file())
            resumed = orchestrator.run_pipeline(request, run_dir)
            self.assertEqual(resumed["status"], "completed")

    def test_request_rejects_secret_fields(self):
        with self.assertRaisesRegex(orchestrator.OrchestrationError, "secret-like"):
            orchestrator.reject_secrets({"OPENAI_API_KEY": "not-allowed"})

    def test_read_markdown_idea_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "idea.md"
            path.write_text("# 因子想法\n\n持仓量与动量背离。\n", encoding="utf-8")
            self.assertEqual(
                orchestrator.read_idea_file(path),
                "# 因子想法\n\n持仓量与动量背离。",
            )

    def test_read_idea_file_rejects_unsupported_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "idea.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(orchestrator.OrchestrationError, "must be a"):
                orchestrator.read_idea_file(path)

    def test_nonempty_run_dir_with_idea_file_can_be_initialized(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "factor.txt").write_text("因子想法", encoding="utf-8")
            orchestrator.ensure_initializable_run_dir(
                run_dir, run_dir / "pipeline-manifest.json"
            )

    def test_unmanaged_run_dir_with_reserved_stage_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            (run_dir / "development").mkdir(parents=True)
            with self.assertRaisesRegex(orchestrator.OrchestrationError, "reserved"):
                orchestrator.ensure_initializable_run_dir(
                    run_dir, run_dir / "pipeline-manifest.json"
                )


if __name__ == "__main__":
    unittest.main()
