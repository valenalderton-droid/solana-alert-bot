import unittest
from nine_core_engine import CORE_SPECS, evaluate_nine_cores, score_for_core, evolve_core_population


def candidate(address, name, trend=90, breakout=90, momentum=90, volume=90, structure=90,
              risk=90, quality=90, probability=0.9):
    strategies = {"trend_following": trend, "breakout": breakout, "momentum": momentum,
                  "volume_flow": volume, "market_structure": structure, "risk_control": risk,
                  "mean_reversion": 60, "candle_rejection": 70}
    pred = {"signal_id": address + ":100:BUY", "token_address": address, "name": name,
            "symbol": name[:4].upper(), "timestamp": 100, "entry": 100, "stop": 95,
            "target": 115, "stop_pct": 5, "target_pct": 15, "probability": probability,
            "rank_score": 90, "liquidity_usd": 500000, "volume_24h_usd": 1000000,
            "entry_deviation_pct": 0.5, "strategy_ensemble": {"strategies": strategies,
            "risk_flags": [], "market_regime": "uptrend"},
            "signal_quality": {"score": quality, "passed": True}, "recent_news_context": []}
    return {"prediction": pred, "info": {"price": 100, "liquidity": 500000, "volume24h": 1000000},
            "ensemble": pred["strategy_ensemble"], "quality": pred["signal_quality"],
            "entry": 100, "stop": 95, "target": 115, "stop_pct": .05, "target_pct": .15}


class NineCoreTests(unittest.TestCase):
    def test_nine_distinct_cores_each_have_10000_virtual_usd(self):
        state = {}
        result = evaluate_nine_cores([], state, {})
        self.assertEqual(len(CORE_SPECS), 9)
        self.assertEqual(result["virtual_capital_total_usd"], 90000)
        self.assertEqual(len(result["core_accounts"]), 9)
        self.assertTrue(all(a["initial_cash_usd"] == 10000 for a in result["core_accounts"].values()))
        self.assertEqual(len(result["core_access"]), 9)
        self.assertGreaterEqual(len(result["shared_data_sources"]), 5)
        self.assertTrue(all(item["tools"] for item in result["core_access"].values()))
        self.assertTrue(all(item["uses_shared_scan_data"] for item in result["core_access"].values()))

    def test_strong_candidate_gets_majority_and_opens_separate_virtual_trades(self):
        state = {}
        result = evaluate_nine_cores([candidate("mintA", "Alpha")], state, {})
        self.assertEqual(result["winner"], "mintA")
        self.assertGreaterEqual(result["winner_votes"], 5)
        self.assertEqual(len(state["nine_core_system"]["cores"]), 9)
        self.assertTrue(all(isinstance(c["portfolio"], dict) for c in state["nine_core_system"]["cores"].values()))

    def test_weak_candidate_does_not_force_a_winner(self):
        state = {}
        weak = candidate("mintB", "Beta", trend=20, breakout=20, momentum=20,
                         volume=20, structure=20, risk=20, quality=20, probability=0.2)
        result = evaluate_nine_cores([weak], state, {})
        self.assertIsNone(result["winner"])
        self.assertEqual(result["winner_votes"], 0)

    def test_each_core_has_a_separate_trade_history(self):
        state = {}
        evaluate_nine_cores([candidate("mintA", "Alpha")], state, {})
        cores = state["nine_core_system"]["cores"]
        self.assertEqual(len(cores), 9)
        for core in cores.values():
            self.assertIn("portfolio", core)
            self.assertIn("decisions", core)
        # Mutating one core ledger cannot mutate another ledger.
        cores["trend"]["portfolio"]["cash_usd"] -= 123
        self.assertNotEqual(cores["trend"]["portfolio"]["cash_usd"],
                            cores["breakout"]["portfolio"]["cash_usd"])

    def test_successful_core_spawns_mutated_offspring_for_persistently_weak_core(self):
        from copy import deepcopy
        parent_trades = [{"pnl_usd": 25.0} for _ in range(10)]
        weak_trades = [{"pnl_usd": -15.0} for _ in range(14)]
        system = {"cores": {
            "trend": {"portfolio": {"closed_trades": parent_trades, "cash_usd": 9000},
                      "genome": {"profile_id": "trend", "threshold": 58, "risk_multiplier": 1.0,
                                 "generation": 0, "parent_id": None, "generation_start_closed_count": 0,
                                 "last_evaluated_closed_count": 10, "bad_streak": 0, "status": "founder"}},
            "breakout": {"portfolio": {"closed_trades": weak_trades, "cash_usd": 8500},
                         "genome": {"profile_id": "breakout", "threshold": 58, "risk_multiplier": 1.0,
                                    "generation": 0, "parent_id": None, "generation_start_closed_count": 0,
                                    "last_evaluated_closed_count": 13, "bad_streak": 1, "status": "founder"}}
        }}
        old_ledger = deepcopy(system["cores"]["breakout"]["portfolio"])
        evolution = evolve_core_population(system)
        child = system["cores"]["breakout"]["genome"]
        self.assertEqual(len(evolution["replacements"]), 1)
        self.assertEqual(child["status"], "offspring")
        self.assertEqual(child["parent_id"], "trend")
        self.assertEqual(child["generation"], 1)
        self.assertNotEqual(child["profile_id"], "trend")
        self.assertEqual(system["cores"]["breakout"]["portfolio"], old_ledger)
        self.assertEqual(evolution["active_offspring"], 1)

    def test_evolution_does_not_remove_core_before_minimum_sample(self):
        system = {"cores": {
            "trend": {"portfolio": {"closed_trades": [{"pnl_usd": -10.0} for _ in range(5)]},
                      "genome": {"profile_id": "trend", "threshold": 58, "risk_multiplier": 1.0,
                                 "generation": 0, "parent_id": None, "generation_start_closed_count": 0,
                                 "last_evaluated_closed_count": 4, "bad_streak": 1, "status": "founder"}}
        }}
        evolution = evolve_core_population(system)
        self.assertEqual(evolution["replacements"], [])
        self.assertEqual(system["cores"]["trend"]["genome"]["status"], "founder")


if __name__ == "__main__":
    unittest.main()
