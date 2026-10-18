"""验收入口的标准库回归；不导入SDK或连接外部服务。"""

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tests import run_calendar_spread_acceptance as entry


class SourceChecksTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.relative = "strategy.py"
        self.source = self.root / self.relative
        self.source.write_bytes(b"fixture source\n")
        self.expected = {self.relative: hashlib.sha256(self.source.read_bytes()).hexdigest()}

    def check(self):
        with patch.object(entry, "ROOT", self.root), patch.object(entry, "EXPECTED_SHA256", self.expected):
            entry.check_sources()

    def test_preflight_requires_no_sidecar(self):
        self.assertFalse((self.root / "calendar_spread_live.sha256").exists())
        with patch.object(Path, "read_text", side_effect=AssertionError("读取了外置摘要")):
            self.check()

    def test_missing_source_has_structured_diagnostic(self):
        self.source.unlink()
        with self.assertRaisesRegex(RuntimeError, '"actual": "missing"') as caught:
            self.check()
        self.assertIn(self.relative, str(caught.exception))

    def test_mixed_source_version_is_rejected(self):
        self.source.write_bytes(b"different version\n")
        with self.assertRaisesRegex(RuntimeError, "版本不符") as caught:
            self.check()
        self.assertIn(self.expected[self.relative], str(caught.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
