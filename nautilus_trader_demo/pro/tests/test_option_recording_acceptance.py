from copy import deepcopy
from importlib import import_module
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

review = import_module("demos.08_option_vega.acceptance").review_recording


def report():
    stamp = int(datetime(2026, 10, 12, 13, 58, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp())*1_000_000_000
    obs = {"000852": {"source_id": "index-bars", "source_ns": stamp, "received_ns": stamp+2},
        "IM2610": {"source_id": "market-bars", "source_ns": stamp-1, "received_ns": stamp+1},
        "MO2610-C-7000": {"source_id": "market-bars", "source_ns": stamp-1, "received_ns": stamp+1}}
    return {"status": "passed", "mode": "recording", "execution_origin": "recording",
        "simnow_environment": "realtime", "market_source": "ctp", "index_source": "dolphindb-dfs",
        "trading_day": "20261012", "valuation_day": "20261012", "observed_md_trading_day": "20261012",
        "orders_submitted": 0, "fills_received": 0, "account_positions": None, "observed_td_trading_day": None,
        "calendar_manifest": {"backend": "compiled"}, "feature_queries": [{"connected": False,
            "failure": None, "cached_records": 40, "last_query_ns": stamp+2, "emitted_versions": {str(stamp): 1}}],
        "cleanup_errors": [], "submissions": 1, "frames": 40, "skipped_index_frames": 0,
        "recorded_requests": [{"signal_ns": stamp, "targets": {"MO2610-C-7000.CFFEX": "-1", "IM2610.CFFEX": "0"}}],
        "entry_audit": [{"date": "2026-10-12", "slot": "13:58", "signal_ns": stamp,
            "future_bindings": {"MO2610-C-7000": "IM2610"},
            "outcome": "TARGET", "targets": {"MO2610-C-7000": "-1"}}],
        "frame_audit": [{"valuation_ns": stamp, "request_ns": stamp+5_000_000_000,
            "deadline_ns": stamp+5_000_000_000, "status": "CONSUMED", "observations": obs}]}


class RecordingAcceptanceTests(unittest.TestCase):
    def review(self, data, **kwargs):
        return review(data, trading_day="20261012", **kwargs)

    def test_closed_connection_still_has_successful_query_evidence(self):
        self.assertEqual(self.review(report())["status"], "passed")

    def test_general_passed_without_selection_is_not_acceptance(self):
        data = report(); data["entry_audit"] = []
        self.assertEqual(self.review(data)["status"], "failed")

    def test_online_missing_index_is_reported_as_observed_timeout(self):
        data = report(); data["frame_audit"][0]["status"] = "SKIPPED_INDEX"
        result = self.review(data)
        self.assertTrue(result["timeout_seen_online"])
        self.assertIn("已观测", result["timeout_note"])

    def test_failed_startup_does_not_claim_trades_from_missing_evidence(self):
        data = report()
        data.pop("orders_submitted")
        data["feature_queries"][0]["emitted_versions"] = {}
        data["failure"] = "行情未就绪"
        result = self.review(data)
        self.assertIn("报告缺少柜台报单计数证据", result["errors"])
        self.assertNotIn("Recording出现柜台报单", result["errors"])
        self.assertTrue(result["index_query_verified"])
        self.assertFalse(result["index_bar_delivery_verified"])

    def test_generated_goal_without_recorded_request_is_not_target_acceptance(self):
        data = report(); data["recorded_requests"] = []
        self.assertEqual(self.review(data)["status"], "failed")

    def test_no_candidates_does_not_claim_target_chain(self):
        data = report(); data["entry_audit"][0].update(outcome="NO_VALID_ENTRY", targets={})
        self.assertEqual(self.review(data)["status"], "failed")
        limited = self.review(data, require_target=False)
        self.assertEqual(limited["status"], "passed")
        self.assertFalse(limited["target_chain_verified"])

    def test_future_or_late_index_and_wrong_day_are_rejected(self):
        for change in ("late", "future", "day", "skipped", "fill"):
            data = deepcopy(report())
            frame = data["frame_audit"][0]
            if change == "late": frame["observations"]["000852"]["received_ns"] = frame["deadline_ns"]+1
            if change == "future": frame["observations"]["000852"]["source_ns"] += 60_000_000_000
            if change == "day": data["observed_md_trading_day"] = "20261009"
            if change == "skipped": frame["status"] = "SKIPPED_INDEX"
            if change == "fill": data["fills_received"] = 1
            with self.subTest(change=change):
                self.assertEqual(self.review(data)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
