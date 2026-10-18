"""在线阶段控制器的标准库回归；不导入SDK，不连接或报单。"""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.integration.calendar_spread import acceptance as entry


class StageTests(unittest.TestCase):
    context = {"trading_day": "20261013", "source_day": "20261012",
        "instruments": {"secondary": "cu2612.SHFE", "far": "cu2701.SHFE"}}

    def summary(self, mode):
        return {"status": "passed", "mode": mode, "cleanup_errors": [], "fixed_instruments": self.context["instruments"],
            "complete_frames": 4, "submissions": 1, "fills_received": 2, "final_active_orders": 0,
            "last_targets": {"cu2612.SHFE": "1", "cu2701.SHFE": "-1"},
            "working_quantities": {"cu2612.SHFE": "0", "cu2701.SHFE": "0"},
            "unassigned_positions": {"cu2612.SHFE": "0", "cu2701.SHFE": "0"},
            "strategy_positions": {"cu2612.SHFE": "1", "cu2701.SHFE": "-1"}}

    def test_order_authorization_required_only_for_simnow(self):
        for extra in (["--through", "simnow"], ["--enable-orders"], ["--confirm-simnow"]):
            with self.subTest(extra=extra), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                entry.parse_args(["--connect", *extra])
        args = entry.parse_args(["--connect", "--through", "simnow", "--enable-orders", "--confirm-simnow"])
        for mode in ("recording", "model"):
            worker = entry.command(args, self.context, mode, Path("fixture"))
            self.assertNotIn("--enable-orders", worker)
            self.assertNotIn("--confirm-simnow", worker)
        self.assertIn("--enable-orders", entry.command(args, self.context, "simnow", Path("fixture")))

    def test_warmed_recording_without_targets_cannot_advance(self):
        summary = self.summary("recording")
        summary["submissions"] = 0
        self.assertEqual(entry.review(summary, "recording", expected=self.context["instruments"], lookback=3)["status"], "failed")

    def test_model_rejects_incomplete_pair_even_with_fill_callbacks(self):
        summary = self.summary("model")
        summary["strategy_positions"]["cu2701.SHFE"] = "0"
        self.assertEqual(entry.review(summary, "model", expected=self.context["instruments"], lookback=3)["status"], "failed")

    def test_simnow_requires_shutdown_confirmation(self):
        summary = self.summary("simnow")
        summary["final_active_orders"] = None
        self.assertEqual(entry.review(summary, "simnow", expected=self.context["instruments"], lookback=3)["status"], "failed")

    def test_stage_failure_never_launches_later_mode(self):
        with TemporaryDirectory() as temp, patch.object(entry, "prepare", return_value=self.context), \
                patch.object(entry, "run_stage", return_value={"status": "failed", "mode": "recording"}) as run, \
                redirect_stdout(StringIO()):
            code = entry.main(["--connect", "--through", "simnow", "--enable-orders", "--confirm-simnow", "--report-dir", temp])
        self.assertEqual(code, 1)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[2], "recording")

    def test_stages_advance_in_order(self):
        modes = []
        def run(args, context, mode, folder):
            modes.append(mode)
            return {"status": "passed", "mode": mode}
        with TemporaryDirectory() as temp, patch.object(entry, "prepare", return_value=self.context), \
                patch.object(entry, "run_stage", side_effect=run), redirect_stdout(StringIO()):
            code = entry.main(["--connect", "--through", "simnow", "--enable-orders", "--confirm-simnow", "--report-dir", temp])
        self.assertEqual(code, 0)
        self.assertEqual(modes, ["recording", "model", "simnow"])

    def test_explicit_instrument_age_is_forwarded_to_every_stage(self):
        args = entry.parse_args(["--connect", "--instrument-max-age-seconds", "30"])
        self.assertEqual(entry.parse_args(["--connect"]).instrument_max_age_seconds, 10)
        replay = entry.parse_args(["--connect", "--environment", "replay", "--replay-md-trading-day", "20261012"])
        self.assertEqual(replay.instrument_max_age_seconds, 30)
        for mode in ("recording", "model", "simnow"):
            worker = entry.command(args, self.context, mode, Path("fixture"))
            index = worker.index("--instrument-max-age-seconds")
            self.assertEqual(float(worker[index + 1]), 30)
            self.assertEqual(worker.count("--instrument-max-age-seconds"), 1)

    def test_invalid_instrument_age_is_rejected_before_connect(self):
        for value in ("0", "-1", "121", "nan", "inf"):
            with self.subTest(value=value), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                entry.parse_args(["--connect", "--instrument-max-age-seconds", value])

    def test_failed_review_keeps_exact_market_diagnostic(self):
        summary = self.summary("recording")
        summary.update(status="failed", complete_frames=0, submissions=0,
            failure="双腿行情、参考资料或执行渠道失效",
            session_readiness={"ready": False, "issues": [{"code": "INSTRUMENT_UNHEALTHY",
                "detail": "cu2701.SHFE", "context": {"event_age_seconds": 10.033407117,
                    "receive_age_seconds": 9.004604016, "max_age_seconds": 10}}]})
        result = entry.review(summary, "recording", expected=self.context["instruments"], lookback=3)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["session_readiness"], summary["session_readiness"])
        self.assertEqual(result["runtime_failure"], summary["failure"])

    def test_completed_positions_with_working_or_unassigned_quantity_fail_review(self):
        for field in ("working_quantities", "unassigned_positions"):
            summary = self.summary("model")
            summary[field]["cu2612.SHFE"] = "1"
            with self.subTest(field=field):
                self.assertEqual(entry.review(summary, "model", expected=self.context["instruments"], lookback=3)["status"], "failed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
