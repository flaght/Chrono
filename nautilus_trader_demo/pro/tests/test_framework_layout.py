"""Regression checks for copying framework without rewriting caller imports."""
from __future__ import annotations

import ast
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "bomber" / "framework"
SCRIPT = ROOT / "scripts" / "sync-bomber-framework.py"
SYNC = runpy.run_path(str(SCRIPT))


class FrameworkReleaseTests(unittest.TestCase):
    def run_sync(self, *args):
        return subprocess.run([sys.executable, "-B", str(SCRIPT), *map(str, args)],
                              capture_output=True, text=True)

    def test_stage_includes_dataprep_and_complete_native_build_inputs(self):
        files = SYNC["staged_files"](SOURCE)
        for name in ("dataprep/basic.py", "__init__.py", "fixes.py", "market/basic/setup.py",
                     "market/basic/custom_bar.pyx", "market/basic/custom_bar.pxd",
                     "market/native/ctp/binding/meson.build",
                     "market/native/ctp/binding/pyproject.toml",
                     "market/native/ctp/binding/src/bomber_ctp_md.cpp",
                     "market/native/ctp/binding/include/ctp/ThostFtdcMdApi.h",
                     "market/native/ctp/binding/libthostmduserapi_se.so",
                     "market/native/ctp/binding/libthosttraderapi_se.so"):
            with self.subTest(name=name):
                self.assertEqual(files[Path(name)], (SOURCE / name).read_bytes())
        self.assertFalse(any("__pycache__" in p.parts for p in files))
        self.assertNotIn(Path("market/basic/fast_factory.cpython-312-linux.so"), files)
        self.assertTrue(SYNC["excluded"](Path("market/basic/fast_factory.cpython-312-linux.so")))

    def test_apply_is_idempotent_removes_stale_files_and_preserves_engine(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)
            (target / "bomber").mkdir()
            engine = target / "bomber" / "__init__.py"
            engine.write_text("# original engine initializer\n")
            result = self.run_sync("--dist-root", target)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertFalse((target / "bomber/framework").exists())
            result = self.run_sync("--dist-root", target, "--apply")
            self.assertEqual(result.returncode, 0, result.stderr)
            framework = target / "bomber/framework"
            self.assertEqual((framework / "dataprep/basic.py").read_bytes(),
                             (SOURCE / "dataprep/basic.py").read_bytes())
            self.assertEqual(self.run_sync("--dist-root", target).returncode, 0)
            (framework / "obsolete.py").write_text("# stale\n")
            self.assertIn("STALE obsolete.py", self.run_sync("--dist-root", target).stdout)
            self.assertEqual(self.run_sync("--dist-root", target, "--apply").returncode, 0)
            self.assertFalse((framework / "obsolete.py").exists())
            self.assertEqual(engine.read_text(), "# original engine initializer\n")

    def test_source_overlap_is_rejected_without_writing(self):
        result = self.run_sync("--dist-root", ROOT, "--apply", "--adopt-existing")
        self.assertEqual(result.returncode, 2)
        self.assertIn("non-overlapping", result.stderr)
        self.assertFalse((SOURCE / SYNC["MARKER"]).exists())

    def test_existing_copy_requires_explicit_adoption(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "bomber/framework"
            destination.mkdir(parents=True)
            (destination / "old.py").write_text("# existing\n")
            result = self.run_sync("--dist-root", temp, "--apply")
            self.assertEqual(result.returncode, 2)
            self.assertTrue((destination / "old.py").exists())
            result = self.run_sync("--dist-root", temp, "--apply", "--adopt-existing")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((destination / "old.py").exists())

    def test_legacy_imports_are_rejected_and_source_is_not_rewritten(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            for name in SYNC["PACKAGE_NAMES"]:
                (source / name).mkdir()
            (source / "__init__.py").write_text("")
            (source / "fixes.py").write_text("from market.basic import base\n")
            with self.assertRaisesRegex(ValueError, "legacy imports"):
                SYNC["staged_files"](source)
            self.assertEqual((source / "fixes.py").read_text(), "from market.basic import base\n")

    def test_callers_and_framework_use_only_canonical_imports(self):
        legacy = SYNC["LEGACY_IMPORT_ROOTS"]
        for scope in ("bomber/framework", "demos", "examples", "tests", "scripts/integration"):
            for path in (ROOT / scope).rglob("*.py"):
                tree = ast.parse(path.read_bytes(), filename=str(path))
                self.assertEqual(SYNC["old_imports"](tree), [], str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        name = (node.func.id if isinstance(node.func, ast.Name)
                                else node.func.attr if isinstance(node.func, ast.Attribute) else "")
                        if name in {"import_module", "patch", "ModuleType"} and node.args:
                            arg = node.args[0]
                            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                                self.assertNotIn(arg.value.split(".")[0], legacy, str(path))


class DevelopmentOverlayTests(unittest.TestCase):
    def test_direct_entry_finds_local_framework_without_exporting_project_path(self):
        for demo in ("01_main_ema", "02_sector_chain"):
            with self.subTest(demo=demo), tempfile.TemporaryDirectory() as temp:
                project = Path(temp) / "pro1"
                entry = project / "demos" / demo / "run_backtest.py"
                entry.parent.mkdir(parents=True)
                shutil.copy2(ROOT / "demos" / demo / "run_backtest.py", entry)
                framework = project / "bomber" / "framework"
                scene = framework / "dataprep" / "scenarios"
                scene.mkdir(parents=True)
                shutil.copy2(ROOT / "bomber" / "__init__.py", project / "bomber" / "__init__.py")
                for package in (framework, scene.parent, scene):
                    (package / "__init__.py").write_text("")
                # Stop at the first framework import; exercise the real entry
                # without loading a trading engine or running a backtest.
                (scene / "role_futures.py").write_text(
                    "import bomber, bomber.framework\n"
                    "from pathlib import Path\n"
                    "assert bomber.__version__ == 'test-engine'\n"
                    f"assert Path(bomber.framework.__file__).resolve() == Path({str(framework / '__init__.py')!r}).resolve()\n"
                    "raise SystemExit(0)\n")
                (scene.parent / "bars.py").write_text(
                    "from .scenarios.role_futures import prepare_role_research\n")
                installed = Path(temp) / "site-packages"
                (installed / "bomber" / "framework").mkdir(parents=True)
                (installed / "bomber" / "__init__.py").write_text("__version__ = 'test-engine'\n")
                (installed / "bomber" / "framework" / "__init__.py").write_text(
                    "raise RuntimeError('installed framework selected')\n")
                (installed / "pandas.py").write_text("")
                (installed / "dotenv.py").write_text("def load_dotenv(): pass\n")
                env = os.environ.copy()
                env["PYTHONPATH"] = str(installed)
                result = subprocess.run([sys.executable, "-B", str(entry)], cwd=project,
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_local_framework_wins_and_engine_metadata_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            engine = Path(temp) / "bomber"
            engine.mkdir()
            (engine / "__init__.py").write_text(
                "from pathlib import Path\nfrom .model import TOKEN\n"
                "__version__ = 'test-engine'\nPACKAGE_ROOT = Path(__file__).parent.parent\n")
            (engine / "model.py").write_text("TOKEN = 'engine-model'\n")
            (engine / "framework").mkdir()
            (engine / "framework/__init__.py").write_text("raise RuntimeError('old framework selected')\n")
            code = f"""
import sys
sys.path[:0] = [{str(ROOT)!r}, {temp!r}]
import bomber, bomber.framework, bomber.model
assert bomber.__version__ == 'test-engine'
assert bomber.model.TOKEN == 'engine-model'
assert str(bomber.PACKAGE_ROOT) == {temp!r}
assert bomber.__spec__.origin == {str(engine / '__init__.py')!r}
assert bomber.framework.__file__ == {str(SOURCE / '__init__.py')!r}
from bomber.framework.datahub import RolePriceStore
assert RolePriceStore.__module__.startswith('bomber.framework.')
assert 'datahub' not in sys.modules
"""
            result = subprocess.run([sys.executable, "-B", "-S", "-c", code],
                                    cwd=temp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_relative_paths_follow_runtime_cwd_or_explicit_project_root(self):
        from bomber.framework.dataprep.paths import resolve_paths, resolve_targets_path
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temp:
            try:
                os.chdir(temp)
                root = Path(temp).resolve()
                data = resolve_paths({"data_root": "data"}, env={}, validate=False)
                self.assertEqual(data.fut, root / "data/kline/fut")
                explicit = resolve_paths({"data_root": "data"}, env={}, validate=False,
                                         project_root=root / "other-project")
                self.assertEqual(explicit.fut, root / "other-project/data/kline/fut")
                (root / "targets.csv").touch()
                self.assertEqual(resolve_targets_path("targets.csv"), root / "targets.csv")
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
