import unittest

from signal_quality import MIN_SIGNAL_QUALITY, score_signal


def candles(n=40):
    rows = []
    price = 10.0
    for i in range(n):
        price *= 1.003
        rows.append({
            "timestamp": 1_700_000_000 + i * 3600,
            "open": price * 0.997,
            "high": price * 1.002,
            "low": price * 0.996,
            "close": price,
            "volume": 1000 + i * 20
        })
    return rows


class SignalQualityTests(unittest.TestCase):
    def test_insufficient_data_fails_closed(self):
        result = score_signal({}, [], 500000, 300000, {})
        self.assertFalse(result["passed"])
        self.assertEqual(result["score"], 0)

    def test_score_is_bounded_and_explains_checks(self):
        rows = candles()
        row = {
            "close": rows[-1]["close"],
            "ema_gap": 0.02,
            "ema_gap34": 0.03,
            "price_vs_ema21": 0.02,
            "price_vs_ema50": 0.03,
            "trend_persistence": 0.9,
            "trend_persistence24": 0.8,
            "ret6": 0.03,
            "rsi": 0.60,
            "volume_ratio": 2.0,
            "close_location": 0.8,
            "upper_wick_pct": 0.1,
            "breakout_distance": 0.02
        }
        result = score_signal(
            row, rows, 500000, 400000,
            {"btc_24h_pct": 1.0, "sol_24h_pct": 2.0},
            stop_pct=0.02, target_pct=0.04
        )
        self.assertGreaterEqual(result["score"], 0)
        self.assertLessEqual(result["score"], 100)
        self.assertIn("checks", result)
        self.assertIn("grade", result)
        self.assertEqual(result["minimum_required"], MIN_SIGNAL_QUALITY)


if __name__ == "__main__":
    unittest.main()
