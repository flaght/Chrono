"""固定合约因子处理、240分钟指标预热及首根新Bar衔接的无网络验证。"""
from datetime import date
from decimal import Decimal
from importlib import import_module
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

from bomber.framework.dataprep.history import (
    HistoryBar, HistoryService, HistoryProviderFactory, HistoryProvider, HistoryError,
    ReferenceFactorAdjuster, ImDayWindow)
from bomber.framework.dataprep.sources import ReferenceBatch, ReferenceDataset, ReferenceSourceError

DAY, SOURCE = date(2026, 9, 22), date(2026, 9, 21)
INSTRUMENT = "IM2610.CFFEX"
FACTOR = Decimal("1.456591")
ROOT = Path(__file__).resolve().parents[1]


class MemoryBars(HistoryProvider):
    def __init__(self, config): self.config = config
    def open(self): pass
    def read(self, request): return tuple(self.config["bars"])
    def close(self): pass


class HistoryAdjustmentTests(unittest.TestCase):
    def setUp(self):
        self.window = ImDayWindow(DAY)
        self.source, self.factory = Mock(), Mock()
        self.factory.create_for.return_value = self.source
        self.source.adjustment_factors.return_value = self.batch()
        self.adjuster = ReferenceFactorAdjuster("fake-config", {DAY: SOURCE}, factory=self.factory)
        self.bars = tuple(HistoryBar(INSTRUMENT, stamp, close="100") for stamp in self.window.stamps)
        class Factory(HistoryProviderFactory):
            _providers = dict(HistoryProviderFactory._providers, dolphindb=MemoryBars)
        self.bar_factory = Factory

    def batch(self, **changes):
        row = dict(symbol="IM2610", code="IM", trade_date=SOURCE, pcr_cumfactor=str(FACTOR))
        row.update(changes)
        return ReferenceBatch(ReferenceDataset.ADJUSTMENT_FACTORS, (row,), tuple(row), "fake-factor-source")

    def service(self, **kwargs):
        return HistoryService(database_config={"bars": self.bars}, factory=self.bar_factory,
            price_adjuster=self.adjuster, **kwargs)

    def runner(self, minutes=240):
        ema = import_module("demos.01_main_ema.strategy")
        cp = import_module("demos.01_main_ema.checkpoint")
        strategy = ema.MainEmaStrategy("probe", None, ema.MainEmaConfig("IM", "CFFEX"))
        return SimpleNamespace(ema_checkpoint=cp.EmaCheckpoint(strategy, INSTRUMENT), calendar=self.window,
            history_service=self.service(), history_minutes=minutes, _submit_lock=RLock(),
            _clients={"readonly": SimpleNamespace(_submit_lock=RLock())})

    def test_construction_is_lazy(self):
        self.factory.create_for.assert_not_called()

    def test_one_factor_query_adjusts_240_raw_bars_without_mutating_input(self):
        adjusted = self.adjuster(self.bars)
        self.assertEqual(len(adjusted), 240)
        self.assertEqual(adjusted[0].adjusted_close, Decimal("145.659100"))
        self.assertEqual(adjusted[-1].cumulative_factor, FACTOR)
        self.assertIsNone(self.bars[0].adjusted_close)
        self.source.adjustment_factors.assert_called_once()
        self.source.close.assert_called_once()
        query = self.source.adjustment_factors.call_args.args[0]
        self.assertEqual(query.symbols, ("IM2610",))
        self.assertEqual(query.start_date, SOURCE)
        self.assertEqual(self.adjuster.last_audit[0]["source_day"], "2026-09-21")

    def test_unknown_day_fails_before_any_query(self):
        stamp = ImDayWindow(date(2026, 9, 23)).stamps[0]
        with self.assertRaises(HistoryError): self.adjuster((HistoryBar(INSTRUMENT, stamp, close=1),))
        self.factory.create_for.assert_not_called()

    def test_different_event_days_use_their_declared_factors(self):
        other = date(2026, 9, 23)
        adjuster = ReferenceFactorAdjuster("config", {DAY: SOURCE, other: DAY}, factory=self.factory)
        self.source.adjustment_factors.side_effect = [self.batch(), self.batch(trade_date=DAY, pcr_cumfactor="2")]
        bars = (self.bars[0], HistoryBar(INSTRUMENT, ImDayWindow(other).stamps[0], close="100"))
        result = adjuster(bars)
        self.assertEqual([bar.adjusted_close for bar in result], [Decimal("145.659100"), Decimal("200")])
        self.assertEqual(self.source.adjustment_factors.call_count, 2)

    def test_pcr_factor_is_not_repeatedly_multiplied(self):
        self.source.adjustment_factors.return_value = self.batch(pcr_factor="2")
        adjusted = self.adjuster(self.bars[:2])
        self.assertEqual([bar.adjusted_close for bar in adjusted], [Decimal("145.659100")] * 2)

    def test_source_mapping_cannot_use_same_or_future_day(self):
        for source in (DAY, date(2026, 9, 23)):
            with self.assertRaises(HistoryError):
                ReferenceFactorAdjuster("config", {DAY: source}, factory=self.factory)

    def test_wrong_contract_product_or_date_fails_and_releases(self):
        for changes in ({"symbol": "IM2612"}, {"code": "IF"}, {"trade_date": DAY}):
            self.source.reset_mock()
            self.source.adjustment_factors.return_value = self.batch(**changes)
            with self.assertRaises(HistoryError): self.adjuster(self.bars[:1])
            self.source.close.assert_called_once()
        self.assertIsNone(self.bars[0].adjusted_close)

    def test_missing_and_duplicate_factors_never_default_to_one(self):
        valid = self.batch()
        for rows in ((), valid.rows + valid.rows):
            self.source.adjustment_factors.return_value = ReferenceBatch(valid.dataset, rows, valid.columns, valid.source)
            with self.assertRaises(HistoryError): self.adjuster(self.bars[:1])

    def test_nonpositive_and_nonfinite_factor_rejected(self):
        for factor in ("0", "-1", "nan", "Infinity"):
            self.source.adjustment_factors.return_value = self.batch(pcr_cumfactor=factor)
            with self.assertRaises(HistoryError): self.adjuster(self.bars[:1])

    def test_future_publication_cannot_adjust_earlier_minute(self):
        self.source.adjustment_factors.return_value = self.batch(available_ns=self.window.end_ns)
        with self.assertRaisesRegex(HistoryError, "尚不可用"): self.adjuster(self.bars[:1])
        self.assertIsNone(self.bars[0].adjusted_close)

    def test_already_adjusted_rows_do_not_read_or_reapply_factor(self):
        bar = HistoryBar(INSTRUMENT, self.window.stamps[0], adjusted_close=99)
        self.assertEqual(self.adjuster((bar,)), (bar,))
        self.factory.create_for.assert_not_called()

    def test_query_failure_releases_and_allow_missing_cannot_hide_factor_failure(self):
        self.source.adjustment_factors.side_effect = ReferenceSourceError("injected")
        with self.assertRaises(ReferenceSourceError):
            self.service(missing_policy="allow").load(INSTRUMENT, self.window.stamps, phase="recovery")
        self.source.close.assert_called_once()

    def test_service_adjustment_cannot_change_minute_identity(self):
        service = HistoryService(database_config={"bars": self.bars[:1]}, factory=self.bar_factory,
            price_adjuster=lambda rows: (HistoryBar(INSTRUMENT, rows[0].ts_event + 60_000_000_000, adjusted_close=1),))
        with self.assertRaisesRegex(HistoryError, "身份"):
            service.load(INSTRUMENT, self.window.stamps[:1], phase="recovery")

    def test_file_raw_prices_and_database_gap_use_same_factor_processing(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bars.jsonl"
            path.write_text(json.dumps(dict(instrument_id=INSTRUMENT, ts_event=self.window.stamps[0], close="100")))
            result = self.service(file_path=path).load(INSTRUMENT, self.window.stamps[:2], phase="preopen")
        self.assertEqual(result.sources, ("file", "dolphindb"))
        self.assertEqual([bar.adjusted_close for bar in result.bars], [Decimal("145.659100")] * 2)
        self.source.adjustment_factors.assert_called_once()

    def test_window_is_240_minutes_and_excludes_lunch(self):
        self.assertEqual(len(self.window.stamps), 240)
        self.assertEqual(self.window.stamps[120] - self.window.stamps[119], 91 * 60_000_000_000)
        self.assertEqual(self.window.last_minutes(self.window.end_ns, 240), self.window.stamps)
        self.assertEqual(self.window.missing_minutes(self.window.stamps[119], self.window.stamps[120] + 1),
            (self.window.stamps[120],))
        with self.assertRaises(HistoryError): self.window.last_minutes(self.window.end_ns, 241)

    def test_240_warmup_updates_indicators_and_publishes_no_targets(self):
        from scripts.integration.history.warmup import probe
        result = probe(self.service(), INSTRUMENT, DAY)
        self.assertEqual(result["ema"]["bars_used"], 240)
        self.assertEqual(result["ema"]["revision"], 0)
        self.assertIsNone(result["ema"]["last_target"])
        self.assertEqual(result["history"]["missing"], 0)
        self.assertEqual(result["targets_published"], 0)
        self.assertFalse(result["ctp_connected"])
        self.assertEqual(result["history"]["price_adjustment"][0]["factor"], "1.456591")

    def test_invalid_factor_does_not_partially_mutate_ema(self):
        runner = self.runner()
        self.source.adjustment_factors.return_value = self.batch(symbol="IM2612")
        warm_history = import_module("demos.01_main_ema.history").warm_history
        with self.assertRaises(HistoryError): warm_history(runner, self.window.end_ns, phase="recovery")
        self.assertEqual(runner.ema_checkpoint.strategy.bars_used, 0)

    def test_explicit_allow_warms_239_real_rows_and_reports_the_missing_minute(self):
        from scripts.integration.history.warmup import probe
        self.bars = self.bars[:120] + self.bars[121:]
        result = probe(self.service(missing_policy="allow"), INSTRUMENT, DAY)
        self.assertEqual(result["status"], "allowed_missing")
        self.assertFalse(result["coverage_complete"])
        self.assertEqual(result["history"]["required"], 240)
        self.assertEqual(result["history"]["loaded"], 239)
        self.assertEqual(result["history"]["missing"], 1)
        self.assertEqual(result["missing_minutes"], ["2026-09-22T13:00:59+08:00"])
        self.assertEqual(result["ema"]["bars_used"], 239)
        self.assertEqual(result["targets_published"], 0)
        self.assertEqual(result["orders_submitted"], 0)

    def test_same_missing_minute_still_fails_by_default_before_factor_query(self):
        from scripts.integration.history.warmup import probe
        from bomber.framework.dataprep.history import HistoryUnavailable
        self.bars = self.bars[:120] + self.bars[121:]
        with self.assertRaises(HistoryUnavailable): probe(self.service(), INSTRUMENT, DAY)
        self.source.adjustment_factors.assert_not_called()

    def test_allow_missing_does_not_claim_initialized_ema_with_too_few_rows(self):
        from scripts.integration.history.warmup import probe
        self.bars = self.bars[:4]
        with self.assertRaises(RuntimeError):
            probe(self.service(missing_policy="allow"), INSTRUMENT, DAY)

    def test_warmup_cli_missing_policy_is_explicit_and_defaults_to_fail(self):
        from scripts.integration.history.warmup import parse_args
        base = ["--connect", "--config", "unused.json", "--instrument", INSTRUMENT,
            "--trading-day", "2026-09-22"]
        self.assertEqual(parse_args(base).history_missing_policy, "fail")
        self.assertEqual(parse_args([*base, "--history-missing-policy", "allow"]).history_missing_policy, "allow")

    def test_restart_then_first_new_bar_publishes_one_target_without_historical_replay(self):
        from bomber.framework.market.basic.base import make_bar, InstrumentId
        runner = self.runner(239)
        history = import_module("demos.01_main_ema.history")
        history.warm_history(runner, self.window.end_ns - 60_000_000_000, phase="recovery")
        saved = runner.ema_checkpoint.snapshot_state()
        restored = self.runner(239)
        restored.ema_checkpoint.restore_state(saved)
        history.warm_history(restored, self.window.end_ns - 60_000_000_000, phase="recovery")
        strategy = restored.ema_checkpoint.strategy
        self.assertEqual(strategy.bars_used, 239)
        self.assertEqual(strategy._revision, 0)
        intents = []
        strategy._bind(SimpleNamespace(submit=intents.append))
        assignment = SimpleNamespace(instrument=lambda product, role: "IM2610", factor=lambda product, role: FACTOR)
        strategy.data_hub = SimpleNamespace(snapshot=lambda stamp: assignment)
        event = make_bar(InstrumentId.from_str(INSTRUMENT), 100, 100, 100, 100, 1, self.window.stamps[-1])
        strategy.on_bar("fixed-test", event)
        strategy.on_bar("fixed-test", event)
        self.assertEqual(strategy.bars_used, 240)
        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].revision, 1)
        self.assertEqual(intents[0].ts_event, self.window.stamps[-1])
        self.assertEqual(strategy.fills_received, 0)

    def test_build_history_accepts_explicit_factor_map_without_connecting(self):
        from bomber.framework.dataprep.sources import DolphinDbReferenceConfig
        module = import_module("demos.01_main_ema.history")
        args = SimpleNamespace(history_config=ROOT / "demos/01_main_ema/history_im_20260922.json",
            history_db="dolphindb", history_file=None, history_missing_policy="fail", reference_timeout=15)
        fake = DolphinDbReferenceConfig("fake", 8848, "fake", "fake")
        with patch.object(module.DolphinDbReferenceConfig, "from_env", return_value=fake):
            service = module.build_history(args)
        self.assertEqual(service.price_adjuster.source_days, {DAY: SOURCE})
        self.assertEqual(service.config.date_column, "date")

    def resume_bars(self):
        self.bars = tuple(HistoryBar(INSTRUMENT, stamp, open="100", high="150",
            low="99", close=str(100 + index / 10))
            for index, stamp in enumerate(self.window.stamps))

    def test_indicator_file_resume_matches_continuous_and_ignores_duplicate_events(self):
        from scripts.integration.history.resume import probe
        self.resume_bars()
        with TemporaryDirectory() as directory:
            result = probe(self.service(), INSTRUMENT, DAY, Path(directory) / "indicator.json")
        self.assertEqual(result["saved_ema"]["bars_used"], 239)
        self.assertEqual(result["after_event"]["bars_used"], 240)
        self.assertTrue(result["continuous_ema_matches"])
        self.assertTrue(result["duplicate_new_event_ignored"])
        self.assertEqual(result["targets_captured"], 1)
        self.assertEqual(result["historical_targets_replayed"], 0)
        self.assertEqual(result["orders_submitted"], 0)
        self.assertEqual(result["after_event"]["revision"], 1)
        self.assertTrue(result["coverage_complete"])

    def test_allow_gap_file_resume_uses_actual_prefix_without_filling_hole(self):
        from scripts.integration.history.resume import probe
        self.resume_bars()
        self.bars = self.bars[:120] + self.bars[121:]
        with TemporaryDirectory() as directory:
            result = probe(self.service(missing_policy="allow"), INSTRUMENT, DAY,
                Path(directory) / "indicator.json")
        self.assertEqual(result["saved_ema"]["bars_used"], 238)
        self.assertEqual(result["after_event"]["bars_used"], 239)
        self.assertFalse(result["coverage_complete"])
        self.assertEqual(result["history"]["missing"], 1)
        self.assertEqual(result["targets_captured"], 1)

    def test_resume_requires_real_terminal_bar_even_if_missing_allowed(self):
        from scripts.integration.history.resume import probe
        self.resume_bars()
        self.bars = self.bars[:-1]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "indicator.json"
            with self.assertRaises(HistoryError):
                probe(self.service(missing_policy="allow"), INSTRUMENT, DAY, path)
            self.assertFalse(path.exists())

    def test_resume_preserves_existing_indicator_before_database_query(self):
        from scripts.integration.history.resume import probe
        with TemporaryDirectory() as directory:
            path = Path(directory) / "indicator.json"
            path.write_text("existing evidence")
            with self.assertRaises(FileExistsError): probe(self.service(), INSTRUMENT, DAY, path)
            self.assertEqual(path.read_text(), "existing evidence")
        self.source.adjustment_factors.assert_not_called()

    def test_indicator_file_digest_rejects_changed_state(self):
        from scripts.integration.history.resume import digest, read_indicator
        state = {"bars_used": 238}
        artifact = {"version": 1, "purpose": "readonly_indicator_probe",
            "ema": dict(state), "sha256": digest(state)}
        artifact["ema"]["bars_used"] = 239
        with TemporaryDirectory() as directory:
            path = Path(directory) / "indicator.json"
            path.write_text(json.dumps(artifact))
            with self.assertRaises(HistoryError): read_indicator(path)

    def test_staged_restart_backfills_without_target_then_captures_one_new_event(self):
        from scripts.integration.history.restart import run_stage
        self.resume_bars()
        self.bars = self.bars[:120] + self.bars[121:]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "restart.json"
            with patch("scripts.integration.history.restart.PROCESS_RUN_ID", "first"):
                prepared = run_stage(self.service(missing_policy="allow"), INSTRUMENT, DAY, path, stage="prepare")
            with patch("scripts.integration.history.restart.PROCESS_RUN_ID", "second"):
                resumed = run_stage(self.service(missing_policy="allow"), INSTRUMENT, DAY, path, stage="resume")
        self.assertEqual(prepared["saved_ema"]["bars_used"], 237)
        self.assertEqual(resumed["restored_ema"], prepared["saved_ema"])
        self.assertEqual(resumed["after_backfill"]["bars_used"], 238)
        self.assertEqual(resumed["after_backfill"]["revision"], 0)
        self.assertEqual(resumed["after_event"]["bars_used"], 239)
        self.assertTrue(resumed["separate_process_observed"])
        self.assertEqual(resumed["targets_captured"], 1)
        self.assertEqual(resumed["backfill_history"]["required"], 1)
        self.assertEqual(resumed["orders_submitted"], 0)

    def test_staged_restart_missing_backfill_cannot_be_hidden_by_allow(self):
        from scripts.integration.history.restart import run_stage
        from bomber.framework.dataprep.history import HistoryUnavailable
        self.resume_bars()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "restart.json"
            run_stage(self.service(), INSTRUMENT, DAY, path, stage="prepare")
            self.bars = self.bars[:-2] + self.bars[-1:]
            with self.assertRaises(HistoryUnavailable):
                run_stage(self.service(missing_policy="allow"), INSTRUMENT, DAY, path, stage="resume")

    def test_staged_restart_wrong_period_rejected_before_database_read(self):
        from scripts.integration.history.restart import run_stage
        self.resume_bars()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "restart.json"
            run_stage(self.service(), INSTRUMENT, DAY, path, stage="prepare")
            self.factory.reset_mock()
            with self.assertRaises(ValueError):
                run_stage(self.service(), INSTRUMENT, DAY, path, stage="resume", fast=2)
            self.factory.create_for.assert_not_called()

    def test_staged_restart_detects_changed_historical_prefix(self):
        from dataclasses import replace
        from scripts.integration.history.restart import run_stage
        self.resume_bars()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "restart.json"
            run_stage(self.service(), INSTRUMENT, DAY, path, stage="prepare")
            self.bars = self.bars[:-3] + (replace(self.bars[-3], close=Decimal("140")),) + self.bars[-2:]
            with self.assertRaisesRegex(HistoryError, "修订"):
                run_stage(self.service(), INSTRUMENT, DAY, path, stage="resume")


if __name__ == "__main__":
    unittest.main(verbosity=2)
