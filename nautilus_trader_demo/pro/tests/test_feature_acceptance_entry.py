"""防止部署预检静默失败；仅临时文件/入口调用，不加载行情或交易组件。"""
from contextlib import redirect_stdout
import io
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tests import run_feature_datahub as entry
from tests import run_cross_section_acceptance as cross_entry


class AcceptanceEntryTests(unittest.TestCase):
    def test_native_feature_preflight_works_without_document_directory(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            file = root / "feature.json"
            file.write_text('{}\n', encoding="utf-8")
            expected = {"feature.json": hashlib.sha256(file.read_bytes()).hexdigest()}
            report = cross_entry.inspect_feature_deployment(root, expected)
            self.assertEqual(report["mismatches"], [])
            self.assertFalse((root / "doc").exists())

    def test_native_feature_preflight_identifies_old_demo_before_import(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            file = root / "demos/08_option_vega/run_live.py"
            file.parent.mkdir(parents=True)
            file.write_text('# old CLI\n', encoding="utf-8")
            report = cross_entry.inspect_feature_deployment(root, {"demos/08_option_vega/run_live.py": "0" * 64})
            self.assertEqual(report["mismatches"][0]["path"], "demos/08_option_vega/run_live.py")
            self.assertNotEqual(report["mismatches"][0]["actual"], "0" * 64)

    def test_native_feature_preflight_lists_missing_config_and_retired_module(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            retired = root / "old_index.py"
            retired.write_text('# retired\n', encoding="utf-8")
            report = cross_entry.inspect_feature_deployment(root, {"feature.json": "0" * 64}, ("old_index.py",))
            self.assertEqual(report["mismatches"][0]["actual"], None)
            self.assertEqual(report["retired_files"], ["old_index.py"])

    def test_retired_specialized_provider_is_identified_before_import(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = root / entry.RETIRED_SOURCES[0]
            path.parent.mkdir(parents=True)
            path.write_text("# retired\n", encoding="utf-8")
            output = io.StringIO()
            with patch.object(entry, "ROOT", root), patch.object(entry, "import_module") as load, redirect_stdout(output):
                code = entry.main(["--only", "definitions", "--report-dir", str(root / "results")])
            self.assertEqual(code, 2)
            self.assertIn(entry.RETIRED_SOURCES[0], output.getvalue())
            load.assert_not_called()

    def test_default_code_only_deployment_runs_without_document_directory(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "source.py"
            source.write_text("# fixture\n", encoding="utf-8")
            expected = {"source.py": hashlib.sha256(source.read_bytes()).hexdigest()}
            output = io.StringIO()
            with patch.object(entry, "ROOT", root), patch.object(entry, "SOURCES", ("source.py",)), \
                    patch.object(entry, "EXPECTED_SOURCE_HASHES", expected), \
                    patch.object(entry, "import_module", return_value=SimpleNamespace(__file__=str(source))), \
                    patch.object(entry.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="fixture\n")) as run, \
                    redirect_stdout(output):
                code = entry.main(["--only", "entry", "--report-dir", str(root / "results")])
            report_file, = (root / "results").glob("*/summary.json")
            report = json.loads(report_file.read_text(encoding="utf-8"))
            self.assertFalse((root / "doc").exists())
            self.assertEqual(code, 0)
            self.assertEqual(report["manifest"], f"embedded:{entry.ACCEPTANCE_VERSION}")
            self.assertEqual(report["status"], "passed")
            run.assert_called_once()

    def test_embedded_hashes_still_reject_mixed_code_before_import(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "source.py"
            source.write_text("# unexpected version\n", encoding="utf-8")
            output = io.StringIO()
            with patch.object(entry, "ROOT", root), patch.object(entry, "SOURCES", ("source.py",)), \
                    patch.object(entry, "EXPECTED_SOURCE_HASHES", {"source.py": "0" * 64}), \
                    patch.object(entry, "import_module") as load, redirect_stdout(output):
                code = entry.main(["--report-dir", str(root / "results")])
            self.assertEqual(code, 2)
            self.assertIn("版本不一致: source.py", output.getvalue())
            load.assert_not_called()

    def run_failure(self, manifest_text):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest = root / "manifest.txt"
            if manifest_text is not None:
                manifest.write_text(manifest_text, encoding="utf-8")
            output = io.StringIO()
            with patch.object(entry, "ROOT", root), patch.object(entry, "SOURCES", ()), \
                    redirect_stdout(output):
                code = entry.main(["--manifest", str(manifest), "--report-dir", str(root / "results")])
            reports = list((root / "results").glob("*/summary.json"))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text(encoding="utf-8"))
            self.assertEqual(code, 2)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["modules"], [])
            self.assertIn("部署/加载预检失败", output.getvalue())
            self.assertIn(str(reports[0]), output.getvalue())
            return report, output.getvalue()

    def test_missing_manifest_prints_actual_error_and_saves_report(self):
        report, output = self.run_failure(None)
        self.assertEqual(report["deployment_error_type"], "FileNotFoundError")
        self.assertIn("manifest.txt", output)

    def test_malformed_manifest_is_not_silent(self):
        report, output = self.run_failure("invalid  some.py\n")
        self.assertEqual(report["deployment_error_type"], "RuntimeError")
        self.assertIn("部署摘要格式无效", output)

    def test_missing_expected_source_identifies_file_before_tests(self):
        relative = "bomber/framework/datahub/features/missing.py"
        report, output = self.run_failure("0" * 64 + "  " + relative + "\n")
        self.assertEqual(report["deployment_mismatches"], [relative])
        self.assertIn(relative, output)
        self.assertIn("actual=None", output)


if __name__ == "__main__":
    unittest.main()
