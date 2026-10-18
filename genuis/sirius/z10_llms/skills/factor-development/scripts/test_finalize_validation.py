#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("finalize_validation.py")
SPEC = importlib.util.spec_from_file_location("finalize_validation", SCRIPT)
assert SPEC and SPEC.loader
finalization = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finalization)


class FinalizationTests(unittest.TestCase):
    def test_cli_allows_runtime_only_mode(self):
        args = finalization.parse_args([
            "--run-dir", "run",
            "--test-data", "market.feather",
        ])
        self.assertIsNone(args.approval)

    def test_approval_accepts_matching_hash(self):
        finalization.validate_approval({
            "approved": True,
            "reviewer": "reviewer-a",
            "factor_sha256": "abc",
        }, "abc")

    def test_approval_rejects_hash_mismatch(self):
        with self.assertRaisesRegex(finalization.FinalizationError, "does not match"):
            finalization.validate_approval({
                "approved": True,
                "reviewer": "reviewer-a",
                "factor_sha256": "abc",
            }, "def")

    def test_approval_requires_human_reviewer(self):
        with self.assertRaisesRegex(finalization.FinalizationError, "reviewer"):
            finalization.validate_approval({
                "approved": True,
                "reviewer": "",
                "factor_sha256": "abc",
            }, "abc")


if __name__ == "__main__":
    unittest.main()
