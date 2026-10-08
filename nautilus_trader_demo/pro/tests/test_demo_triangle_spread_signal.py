"""在部署环境验证三腿归一化、历史窗口和方向规则。"""

from decimal import Decimal
from importlib import import_module
from math import log
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
module = import_module("demos.06_triangle_spread.spread_signal")


class SignalTests(unittest.TestCase):
    def make_signal(self):
        return module.TriangleSpreadSignal("RB", ("HC", "I"),
                                           (Decimal("0.5"), Decimal("0.5")), 2, 2.0, 0.5)

    def test_normalization_and_weights(self):
        spread = module.normalized_log_spread(
            {"RB": 110, "HC": 120, "I": 80}, {"RB": 100, "HC": 100, "I": 100},
            "RB", ("HC", "I"), (0.25, 0.75),
        )
        self.assertAlmostEqual(spread, log(1.1) - 0.25 * log(1.2) - 0.75 * log(0.8))

    def test_warmup_and_current_excluded_from_statistics(self):
        signal = self.make_signal()
        self.assertIsNone(signal.update({"RB": 100, "HC": 100, "I": 100}, 0))
        self.assertIsNone(signal.update({"RB": 110, "HC": 100, "I": 100}, 0))
        result = signal.update({"RB": 121, "HC": 100, "I": 100}, 0)
        # 此前窗口为零和一个对数涨幅，当前为两倍涨幅，因此标准分数为3。
        self.assertAlmostEqual(result.z_score, 3.0)
        self.assertEqual(result.direction, -1)

    def test_entry_exit_reverse_and_hold(self):
        self.assertEqual(module.direction_for(-2.0, -1, 2.0, 0.5), 1)
        self.assertEqual(module.direction_for(2.0, 1, 2.0, 0.5), -1)
        self.assertEqual(module.direction_for(0.5, 1, 2.0, 0.5), 0)
        self.assertEqual(module.direction_for(1.0, 1, 2.0, 0.5), 1)

    def test_zero_variance_and_baseline_remains_fixed(self):
        signal = self.make_signal()
        for _ in range(2):
            self.assertIsNone(signal.update({"RB": 100, "HC": 100, "I": 100}, 0))
        result = signal.update({"RB": 100, "HC": 100, "I": 100}, 1)
        self.assertEqual(result.z_score, 0.0)
        self.assertEqual(result.direction, 0)
        signal.update({"RB": 110, "HC": 100, "I": 100}, 0)
        self.assertEqual(signal.baseline, {"RB": 100.0, "HC": 100.0, "I": 100.0})

    def test_invalid_frame_does_not_initialize_or_advance(self):
        signal = self.make_signal()
        for values in ({"RB": 100, "HC": 100}, {"RB": 0, "HC": 100, "I": 100}):
            with self.assertRaises(ValueError):
                signal.update(values, 0)
        self.assertFalse(signal.baseline)
        self.assertFalse(signal.history)
        self.assertIsNone(signal.update({"RB": 100, "HC": 100, "I": 100}, 0))


if __name__ == "__main__":
    unittest.main()
