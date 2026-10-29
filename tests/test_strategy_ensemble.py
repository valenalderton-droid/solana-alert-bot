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

    def test_scores_multiple_strategy_families(self):
        result = evaluate_strategies(make_candles())
        self.assertTrue(result["ready"])
        self.assertEqual(
            set(result["strategies"]),
            {"trend_following", "breakout", "momentum", "mean_reversion", "volume_flow"},
        )
        self.assertGreaterEqual(result["score"], 0)
        self.assertLessEqual(result["score"], 100)

    def test_rank_candidate_is_bounded(self):
        result = evaluate_strategies(make_candles())
        rank = rank_candidate(result, 0.7, 80)
        self.assertGreaterEqual(rank, 0)
        self.assertLessEqual(rank, 100)


if __name__ == "__main__":
    unittest.main()
