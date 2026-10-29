import unittest
import numpy as np

from learner import FEATURES, add_features, build_training_rows


def make_flat_candles(count=80):
    candles = []
    for i in range(count):
        close = 100.0 + 0.08 * np.sin(i / 3.0)
        candles.append({
            "timestamp": 1_800_000_000 + i * 3600,
            "open": close,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": 1000.0 + (i % 7) * 10.0,
        })
    return candles


class LearnerTests(unittest.TestCase):
    def test_features_are_finite_when_present(self):
        features = add_features(make_flat_candles())
        self.assertFalse(features.empty)
        numeric = features.select_dtypes(include=[np.number])
        self.assertTrue(np.isfinite(numeric.to_numpy()[~np.isnan(numeric.to_numpy())]).all())

    def test_new_price_action_features_are_available(self):
        features = add_features(make_flat_candles(120))
        for name in ("ret24", "ema_gap34", "volatility_ratio",
                     "volume_trend", "close_location", "breakout_distance"):
            self.assertIn(name, features.columns)
        latest = features.dropna(subset=FEATURES)
        self.assertFalse(latest.empty)
        self.assertTrue(np.isfinite(latest[FEATURES].to_numpy()).all())

    def test_unresolved_timeout_is_not_labelled_as_a_loss(self):
        rows = build_training_rows([make_flat_candles()])
        # The synthetic price range is far too small to hit either the 2%+
        # stop or 4%+ target within four hours. These are timeouts, not losses.
        self.assertTrue(rows.empty)

    def test_training_rows_include_binary_labels_only(self):
        candles = make_flat_candles()
        # Introduce clear future target hits for some samples, while keeping
        # candle OHLC internally valid.
        for i in range(35, len(candles), 8):
            base = candles[i]["close"]
            for j in range(i + 1, min(i + 5, len(candles))):
                candles[j]["high"] = max(candles[j]["high"], base * 1.06)
        rows = build_training_rows([candles])
        self.assertFalse(rows.empty)
        self.assertTrue(set(rows["y"].unique()).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
