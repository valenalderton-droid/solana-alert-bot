import unittest
import numpy as np

from learner import FEATURES, MultiHorizonModel, add_features, build_training_rows, price_action_ok, recency_sample_weights, walk_forward_validation


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
        features = add_features(make_flat_candles(240))
        for name in ("ret24", "ema_gap34", "volatility_ratio",
                     "volume_trend", "close_location", "breakout_distance",
                     "price_vs_ema21", "rsi_short", "upper_wick_pct",
                     "lower_wick_pct", "high_distance24", "low_distance24",
                     "trend_persistence", "ret48", "ema_gap50", "price_vs_ema50",
                     "trend_persistence24", "volatility_regime", "volume_zscore",
                     "momentum_consistency", "drawdown24", "close_location_avg3"):
            self.assertIn(name, features.columns)
        latest = features.dropna(subset=FEATURES)
        self.assertFalse(latest.empty)
        self.assertTrue(np.isfinite(latest[FEATURES].to_numpy()).all())

    def test_price_action_rejects_price_below_ema21(self):
        candles = [
            {"timestamp": 1_800_000_000 + i * 3600,
             "open": 99.0, "high": 101.0, "low": 98.0,
             "close": 100.0, "volume": 2000.0}
            for i in range(30)
        ]
        row = {
            "close": 100.0, "ema_gap": 0.02, "ret6": 0.01,
            "price_vs_ema21": -0.01, "trend_persistence": 0.8,
            "price_vs_ema50": 0.01, "trend_persistence24": 0.8,
            "rsi": 0.5, "close_location": 0.8,
            "upper_wick_pct": 0.1, "volume_ratio": 2.0
        }
        ok, reason = price_action_ok(candles, row)
        self.assertFalse(ok)
        self.assertIn("EMA21", reason)

    def test_price_action_rejects_rejection_wick_without_breakout(self):
        candles = [
            {"timestamp": 1_800_000_000 + i * 3600,
             "open": 99.0, "high": 101.0, "low": 98.0,
             "close": 100.0, "volume": 2000.0}
            for i in range(30)
        ]
        row = {
            "close": 100.0, "ema_gap": 0.02, "ret6": 0.01,
            "price_vs_ema21": 0.01, "trend_persistence": 0.8,
            "price_vs_ema50": 0.01, "trend_persistence24": 0.8,
            "rsi": 0.5, "close_location": 0.3,
            "upper_wick_pct": 0.7, "volume_ratio": 2.0
        }
        ok, reason = price_action_ok(candles, row)
        self.assertFalse(ok)
        self.assertIn("Weak candle", reason)

    def test_recency_weights_favor_newer_rows_but_keep_old_rows(self):
        timestamps = [1_700_000_000, 1_700_000_000 + 21 * 86400,
                      1_700_000_000 + 42 * 86400]
        weights = recency_sample_weights(timestamps)
        self.assertEqual(len(weights), 3)
        self.assertGreater(weights[2], weights[1])
        self.assertGreater(weights[1], weights[0])
        self.assertGreaterEqual(float(weights[0]), 0.20 * float(weights[2]))
        self.assertAlmostEqual(float(np.mean(weights)), 1.0, places=6)

    def test_walk_forward_validation_uses_multiple_future_folds(self):
        rows = []
        base_ts = 1_700_000_000
        for i in range(100):
            label = i % 2
            for repeat in range(4):
                row = {name: float(label) + (i % 5) * 0.01 for name in FEATURES}
                row.update({
                    "timestamp": base_ts + i * 3600,
                    "label_end": base_ts + i * 3600 + 60,
                    "y": label
                })
                rows.append(row)
        report = walk_forward_validation(__import__("pandas").DataFrame(rows), folds=3)
        self.assertGreaterEqual(report["folds"], 1)
        self.assertIn("mean_balanced_accuracy", report)
        self.assertIn("mean_brier_score", report)
        self.assertIn("method", report)

    def test_unresolved_timeout_is_not_labelled_as_a_loss(self):
        rows = build_training_rows([make_flat_candles()])
        # The synthetic price range is far too small to hit either the 2%+
        # stop or 4%+ target within four hours. These are timeouts, not losses.
        self.assertTrue(rows.empty)

    def test_training_rows_support_longer_horizons(self):
        candles = make_flat_candles(360)
        for i in range(200, len(candles), 8):
            base = candles[i]["close"]
            for j in range(i + 1, min(i + 5, len(candles))):
                candles[j]["high"] = max(candles[j]["high"], base * 1.06)
        rows = build_training_rows([candles], horizon=12)
        self.assertFalse(rows.empty)
        self.assertTrue(((rows["label_end"] - rows["timestamp"]) == 12 * 3600).all())

    def test_multi_horizon_model_returns_weighted_and_separate_probabilities(self):
        class FakeModel:
            def __init__(self, probability):
                self.probability = probability

            def predict_proba(self, X):
                return np.array([[1 - self.probability, self.probability]])

        model = MultiHorizonModel([(4, FakeModel(0.8)), (24, FakeModel(0.4))])
        combined = model.predict_proba(np.array([[1.0]]))
        horizon = model.predict_horizon_proba(np.array([[1.0]]))
        self.assertAlmostEqual(float(combined[0][1]), 0.64)
        self.assertEqual(horizon, {"4h": 0.8, "24h": 0.4})

    def test_training_rows_include_binary_labels_only(self):
        candles = make_flat_candles(320)
        # Introduce clear future target hits for some samples, while keeping
        # candle OHLC internally valid.
        for i in range(200, len(candles), 8):
            base = candles[i]["close"]
            for j in range(i + 1, min(i + 5, len(candles))):
                candles[j]["high"] = max(candles[j]["high"], base * 1.06)
        rows = build_training_rows([candles])
        self.assertFalse(rows.empty)
        self.assertTrue(set(rows["y"].unique()).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
