import unittest

from core_civilization import (
    run_civilization_cycle, is_core_eligible, record_civilization_outcome,
)
from nine_core_engine import CORE_SPECS, evaluate_nine_cores
def candidate(address, name):
    strategies = {"trend_following": 90, "breakout": 90, "momentum": 90, "volume_flow": 90,
                  "market_structure": 90, "risk_control": 90, "mean_reversion": 60, "candle_rejection": 70}
    pred = {"signal_id": address + ":100:BUY", "token_address": address, "name": name,
            "symbol": name[:4].upper(), "timestamp": 100, "entry": 100, "stop": 95, "target": 115,
            "stop_pct": 5, "target_pct": 15, "probability": 0.9, "rank_score": 90,
            "liquidity_usd": 500000, "volume_24h_usd": 1000000, "entry_deviation_pct": 0.5,
            "strategy_ensemble": {"strategies": strategies, "risk_flags": [], "market_regime": "uptrend"},
            "signal_quality": {"score": 90, "passed": True}, "recent_news_context": []}
    return {"prediction": pred, "info": {"price": 100, "liquidity": 500000, "volume24h": 1000000},
            "ensemble": pred["strategy_ensemble"], "quality": pred["signal_quality"],
            "entry": 100, "stop": 95, "target": 115, "stop_pct": 0.05, "target_pct": 0.15}


class CoreCivilizationTests(unittest.TestCase):
    def test_first_scan_runs_election_and_publishes_constitution(self):
        system = {"cores": {spec["id"]: {"portfolio": {"initial_cash_usd": 10000,
                  "cash_usd": 10000, "closed_trades": []}, "decisions": [],
                  "genome": {"threshold": 60}} for spec in CORE_SPECS}}
        report = run_civilization_cycle(system, CORE_SPECS, "2026-10-29T12:00:00+00:00")
        self.assertEqual(report["election_number"], 1)
        self.assertIn(report["president"], [s["id"] for s in CORE_SPECS])
        self.assertEqual(len(report["council"]), 3)
        self.assertGreaterEqual(len(report["laws"]), 4)
        self.assertEqual(len(system["civilization"]["constitution"]), 6)

    def test_police_suspends_only_after_repeated_critical_findings(self):
        system = {"cores": {spec["id"]: {"portfolio": {"initial_cash_usd": 10000,
                  "cash_usd": 10000, "closed_trades": []}, "decisions": [],
                  "genome": {"threshold": 60}} for spec in CORE_SPECS}}
        system["cores"]["trend"]["portfolio"]["cash_usd"] = -1
        run_civilization_cycle(system, CORE_SPECS, "t1")
        self.assertTrue(is_core_eligible(system, "trend"))
        run_civilization_cycle(system, CORE_SPECS, "t2")
        self.assertFalse(is_core_eligible(system, "trend"))
        self.assertEqual(system["civilization"]["police"]["strikes"]["trend"], 2)

    def test_court_reinstates_after_suspension_when_audit_is_clean(self):
        system = {"cores": {spec["id"]: {"portfolio": {"initial_cash_usd": 10000,
                  "cash_usd": 10000, "closed_trades": []}, "decisions": [],
                  "genome": {"threshold": 60}} for spec in CORE_SPECS}}
        system["cores"]["trend"]["portfolio"]["cash_usd"] = -1
        run_civilization_cycle(system, CORE_SPECS, "t1")
        run_civilization_cycle(system, CORE_SPECS, "t2")
        system["cores"]["trend"]["portfolio"]["cash_usd"] = 10000
        for i in range(3, 9):
            run_civilization_cycle(system, CORE_SPECS, f"t{i}")
        self.assertTrue(is_core_eligible(system, "trend"))
        self.assertTrue(any(d["core_id"] == "trend" and d["decision"] == "reinstated_after_review"
                            for d in system["civilization"]["court"]["decisions"]))
        self.assertTrue(any(a["core_id"] == "trend" and a["status"] == "accepted"
                            for a in system["civilization"]["court"]["appeals"]))

    def test_governance_is_integrated_into_nine_core_scan(self):
        state = {}
        result = evaluate_nine_cores([candidate("mintA", "Alpha")], state, {}, now="2026-10-29T12:00:00+00:00")
        self.assertIn("civilization", result)
        self.assertEqual(result["civilization"]["election_number"], 1)
        self.assertIsNotNone(result["winner"])
        self.assertIn("academy", state["nine_core_system"]["civilization"])
        self.assertIn("ecology", state["nine_core_system"]["civilization"])

    def test_virtual_treasury_and_academy_process_closed_trades_once(self):
        system = {"cores": {spec["id"]: {"portfolio": {"initial_cash_usd": 10000,
                  "cash_usd": 10000, "closed_trades": []}, "decisions": [],
                  "genome": {"threshold": 60}} for spec in CORE_SPECS}}
        system["cores"]["trend"]["portfolio"]["closed_trades"] = [
            {"pnl_usd": 100.0, "token_address": "mintA", "exit_reason": "target"},
            {"pnl_usd": -20.0, "token_address": "mintB", "exit_reason": "stop"},
        ]
        first = run_civilization_cycle(system, CORE_SPECS, "t1")
        second = run_civilization_cycle(system, CORE_SPECS, "t2")
        self.assertEqual(first["virtual_treasury_usd"], 1.0)
        self.assertEqual(second["virtual_treasury_usd"], 1.0)
        self.assertEqual(second["research_lessons"], 2)


if __name__ == "__main__":
    unittest.main()
