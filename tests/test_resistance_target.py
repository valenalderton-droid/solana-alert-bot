import unittest

from main import find_resistance_target


def candles_with_highs(highs):
    return [
        {"high": high, "low": max(0.01, high * 0.95), "close": high * 0.98}
        for high in highs
    ]


class TestResistanceTarget(unittest.TestCase):
    def test_selects_nearest_swing_resistance_that_meets_minimum_target(self):
        highs = [1.02, 1.03, 1.04, 1.08, 1.04, 1.03, 1.05, 1.12, 1.06, 1.04, 1.03]
        target = find_resistance_target(candles_with_highs(highs), 1.00, 0.07)
        self.assertIsNotNone(target)
        self.assertAlmostEqual(target, 1.08 * 0.995)

    def test_returns_none_when_resistance_cannot_support_required_target(self):
        highs = [1.02, 1.03, 1.04, 1.06, 1.04, 1.03, 1.05, 1.07, 1.06, 1.04, 1.03]
        self.assertIsNone(find_resistance_target(candles_with_highs(highs), 1.00, 0.10))

    def test_returns_none_when_only_resistance_is_too_far(self):
        highs = [1.02, 1.03, 1.04, 1.50, 1.04, 1.03, 1.05, 1.04, 1.06, 1.04, 1.03]
        self.assertIsNone(find_resistance_target(candles_with_highs(highs), 1.00, 0.08, max_target_pct=0.30))

    def test_target_is_below_resistance(self):
        highs = [1.01, 1.02, 1.03, 1.15, 1.03, 1.02, 1.01]
        target = find_resistance_target(candles_with_highs(highs), 1.00, 0.10)
        self.assertAlmostEqual(target, 1.15 * 0.995)


if __name__ == "__main__":
    unittest.main()
