"""在运行环境验证横截面信号、复权换约连续性和真实价格手数。"""

from decimal import Decimal
from importlib import import_module
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
module = import_module("demos.04_cross_section.cross_section_signal")


class SignalTests(unittest.TestCase):
    def test_warmup_return_and_ranking(self):
        signal = module.CrossSectionMomentumSignal(("RB", "HC", "I"), 2)
        self.assertIsNone(signal.update({"RB": 100, "HC": 100, "I": 100}))
        self.assertIsNone(signal.update({"RB": 105, "HC": 95, "I": 100}))
        result = signal.update({"RB": 110, "HC": 90, "I": 100})
        self.assertEqual(result.ranking, ("RB", "I", "HC"))
        self.assertEqual(result.scores["RB"], Decimal("0.1"))
        self.assertEqual(result.scores["HC"], Decimal("-0.1"))

    def test_ties_use_product_code(self):
        signal = module.CrossSectionMomentumSignal(("RB", "I", "HC"), 1)
        signal.update({"RB": 100, "I": 100, "HC": 100})
        result = signal.update({"I": 100, "HC": 100, "RB": 100})
        self.assertEqual(result.ranking, ("HC", "I", "RB"))

    def test_roll_uses_adjusted_price_without_reset(self):
        signal = module.CrossSectionMomentumSignal(("RB", "HC"), 1)
        signal.update({"RB": Decimal(100) * Decimal(2), "HC": 100})
        # 主力真实价格从100变成200，但累计因子抵消换约跳变。
        result = signal.update({"RB": Decimal(200) * Decimal(1), "HC": 90})
        self.assertEqual(result.scores["RB"], Decimal(0))
        self.assertEqual(result.long_products, ("RB",))
        self.assertEqual(result.short_products, ("HC",))

    def test_invalid_frame_does_not_advance_window(self):
        signal = module.CrossSectionMomentumSignal(("RB", "HC"), 1)
        with self.assertRaises(ValueError):
            signal.update({"RB": 100})
        with self.assertRaises(ValueError):
            signal.update({"RB": 100, "HC": Decimal("NaN")})
        self.assertIsNone(signal.update({"RB": 100, "HC": 100}))

    def test_integer_quantity_does_not_exceed_budget(self):
        self.assertEqual(module.target_quantity(Decimal(100000), Decimal(3000), Decimal(10)), Decimal(3))
        self.assertEqual(module.target_quantity(Decimal(100), Decimal(3000), Decimal(10)), Decimal(0))
        with self.assertRaises(ValueError):
            module.target_quantity(Decimal(100), Decimal(0), Decimal(10))

    def test_multiple_products_per_side_and_equal_budgets(self):
        products = tuple("ABCDEFGH")
        signal = module.CrossSectionMomentumSignal(products, 1)
        signal.update({item: 100 for item in products})
        result = signal.update({item: 108 - index for index, item in enumerate(products)})
        self.assertEqual(result.long_products, ("A", "B"))
        self.assertEqual(result.short_products, ("G", "H"))
        prices = {item: Decimal(1000) for item in products}
        multipliers = {item: Decimal(10) for item in products}
        targets = module.allocate_group_targets(result, Decimal(100000), prices, multipliers)
        self.assertEqual(targets, {"A": 5, "B": 5, "C": 0, "D": 0,
                                   "E": 0, "F": 0, "G": -5, "H": -5})
        for group in (result.long_products, result.short_products):
            self.assertEqual(sum(abs(targets[item]) * prices[item] * multipliers[item]
                                 for item in group), Decimal(100000))

    def test_rounding_and_insufficient_budget_leave_cash(self):
        products = tuple("ABCDEFGH")
        signal = module.CrossSectionMomentumSignal(products, 1)
        signal.update({item: 100 for item in products})
        result = signal.update({item: 108 - index for index, item in enumerate(products)})
        prices = {item: Decimal(3000) for item in products}
        prices["B"] = Decimal(6000)
        multipliers = {item: Decimal(10) for item in products}
        targets = module.allocate_group_targets(result, Decimal(100000), prices, multipliers)
        self.assertEqual(targets["A"], 1)
        self.assertEqual(targets["B"], 0)
        for group in (result.long_products, result.short_products):
            self.assertLessEqual(sum(abs(targets[item]) * prices[item] * multipliers[item]
                                    for item in group), Decimal(100000))

    def test_fraction_limits_and_nonoverlapping_groups(self):
        for fraction in (Decimal(0), Decimal("0.51"), Decimal("NaN")):
            with self.assertRaises(ValueError):
                module.CrossSectionMomentumSignal(("A", "B"), 1, fraction)
        signal = module.CrossSectionMomentumSignal(tuple("ABCDE"), 1, Decimal("0.5"))
        signal.update({item: 100 for item in "ABCDE"})
        result = signal.update({item: 100 for item in "ABCDE"})
        self.assertEqual(result.long_products, ("A", "B"))
        self.assertEqual(result.short_products, ("D", "E"))
        self.assertFalse(set(result.long_products) & set(result.short_products))


if __name__ == "__main__":
    unittest.main()
