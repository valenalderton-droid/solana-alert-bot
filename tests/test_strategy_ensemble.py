import unittest

from strategy_ensemble import evaluate_strategies, rank_candidate


def make_candles(count=80, trend=True):
    rows = []
    price = 100.0
    for i in range(count):
        if trend:
            price *= 1.003
        else:
            price *= 1.0 + (0.002 if i % 2 else -0.001)
        rows.append({
            "timestamp": i * 3600,
            "open": price * 0.998,
            "high": price * 1.006,
            "low": price * 0.994,
            "close": price,
            "volume": 1000.0 if i < count - 5 else 1800.0,
        })
    return rows


class StrategyEnsembleTests(unittest.TestCase):
    def test_requires_sufficient_candles(self):
        result = evaluate_strategies(make_candles(20))
        self.assertFalse(result["ready"])

    def test_scores_expanded_strategy_families(self):
        result = evaluate_strategies(make_candles())
        self.assertTrue(result["ready"])
        self.assertEqual(
            set(result["strategies"]),
            {"trend_following", "breakout", "momentum", "mean_reversion",
             "volume_flow", "market_structure", "candle_rejection", "risk_control"},
        )
        self.assertGreaterEqual(result["score"], 0)
        self.assertLessEqual(result["score"], 100)
        self.assertIn("risk_flags", result)
        self.assertIn("atr_pct", result["metrics"])
        self.assertIn(result["market_regime"], {
            "high_volatility", "uptrend", "downtrend", "stretched_range", "mixed_range"
        })
        self.assertGreaterEqual(result["consensus_strength"], 0)
        self.assertLessEqual(result["consensus_strength"], 1)
        self.assertGreaterEqual(result["strategy_dispersion"], 0)

    def test_invalid_candle_is_rejected(self):
        candles = make_candles()
        candles[-1]["high"] = candles[-1]["low"] - 1
        self.assertFalse(evaluate_strategies(candles)["ready"])

    def test_rank_penalizes_chasing_and_risk(self):
        ensemble = evaluate_strategies(make_candles())
        normal = rank_candidate(ensemble, 0.7, 80, 150000, 200000, 0.2)
        chased = rank_candidate(ensemble, 0.7, 80, 150000, 200000, 3.0)
        self.assertGreaterEqual(normal, 0)
        self.assertLessEqual(normal, 100)
        self.assertLess(chased, normal)

    def test_rank_penalizes_strategy_disagreement(self):
        base = {
            "score": 75,
            "risk_flags": [],
            "consensus_strength": 0.8,
            "strategy_dispersion": 10
        }
        conflicted = dict(base, consensus_strength=0.2, strategy_dispersion=35)
        ordinary_rank = rank_candidate(base, 0.7, 80, 150000, 200000, 0.2)
        conflicted_rank = rank_candidate(conflicted, 0.7, 80, 150000, 200000, 0.2)
        self.assertLess(conflicted_rank, ordinary_rank)

    def test_rank_candidate_is_bounded(self):
        result = evaluate_strategies(make_candles())
        rank = rank_candidate(result, 0.7, 80)
        self.assertGreaterEqual(rank, 0)
        self.assertLessEqual(rank, 100)


if __name__ == "__main__":
    unittest.main()
