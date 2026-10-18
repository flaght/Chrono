"""执行 Bar 时钟门控夹具；仅生成代码，请在运行环境执行。"""
from decimal import Decimal
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

# 当前示例通过直接脚本方式导入同目录模块。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demos" / "02_sector_chain"))
strategy_module = importlib.import_module("demos.02_sector_chain.strategy")
signal_module = importlib.import_module("sector_signal")


class Snapshot:
    def instrument(self, product, role):
        return "rb2610" if role == "secondary" else product.lower() + "2605"

    def factor(self, product, role):
        return Decimal(1)


class ControlledSignal:
    def __init__(self):
        self.last_timestamp = -1
        self.direction = 1

    def update(self, timestamp, closes):
        self.last_timestamp = timestamp
        return signal_module.SectorSignal(timestamp, self.direction,
            Decimal("0.01"), Decimal("0.005"), (Decimal("0.005"), Decimal("0.005")))


class ExecutionClockTests(unittest.TestCase):
    def make_strategy(self, delay=0):
        hub = SimpleNamespace(snapshot=lambda timestamp: Snapshot())
        strategy = strategy_module.SectorChainTargetStrategy("test-sector", hub,
            strategy_module.SectorChainConfig(submission_delay_bars=delay))
        strategy.signal = ControlledSignal()
        self.submissions = []
        strategy.set_target = lambda key, quantity, timestamp, **kwargs: self.submissions.append(
            (key, quantity, timestamp, kwargs))
        return strategy

    def bar(self, strategy, symbol, timestamp):
        strategy.on_bar("bars", SimpleNamespace(ts_event=timestamp, close=Decimal(100),
            bar_type=SimpleNamespace(instrument_id=SimpleNamespace(symbol=symbol))))

    def frame(self, strategy, timestamp):
        for symbol in ("jm2605", "i2605", "rb2605"):
            self.bar(strategy, symbol, timestamp)

    def test_stale_execution_quote_waits_for_same_timestamp_execution_bar(self):
        strategy = self.make_strategy()
        self.bar(strategy, "rb2610", 1)
        self.frame(strategy, 100)
        self.assertEqual(self.submissions, [])
        self.assertEqual(strategy.execution_events[-1]["reason"], "waiting_for_execution_bar")
        self.bar(strategy, "rb2610", 100)
        self.assertEqual(self.submissions[0][:3], ("rb_secondary", Decimal(1), 100))
        self.assertIsNone(strategy._pending)

    def test_execution_bar_arriving_first_allows_submission_after_frame(self):
        strategy = self.make_strategy()
        self.bar(strategy, "rb2610", 100)
        self.frame(strategy, 100)
        self.assertEqual(len(self.submissions), 1)
        self.assertEqual(self.submissions[0][2], 100)

    def test_positive_delay_counts_only_later_execution_bars(self):
        strategy = self.make_strategy(delay=2)
        self.frame(strategy, 100)
        self.bar(strategy, "rb2610", 100)
        self.assertEqual(strategy._pending.remaining_bars, 2)
        self.bar(strategy, "rb2610", 101)
        self.assertEqual(self.submissions, [])
        self.bar(strategy, "rb2610", 102)
        self.assertEqual(self.submissions[0][2], 102)

    def test_new_opposite_signal_replaces_waiting_target(self):
        strategy = self.make_strategy()
        self.frame(strategy, 100)
        strategy.signal.direction = -1
        self.frame(strategy, 101)
        self.bar(strategy, "rb2610", 101)
        self.assertEqual(len(self.submissions), 1)
        self.assertEqual(self.submissions[0][1], Decimal(-1))

    def test_no_future_execution_bar_expires_without_submission(self):
        strategy = self.make_strategy()
        self.frame(strategy, 100)
        strategy.on_stop()
        self.assertEqual(self.submissions, [])
        self.assertIsNone(strategy._pending)
        self.assertEqual(strategy.execution_events[-1]["kind"], "target_expired_no_future_bar")


if __name__ == "__main__":
    unittest.main()
