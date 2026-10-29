import unittest

from core_civilization import (
    run_civilization_cycle, is_core_eligible, record_civilization_outcome,
)
from nine_core_engine import CORE_SPECS, evaluate_nine_cores
from tests.test_nine_core_engine import candidate


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

    def test_governance_is_integrated_into_nine_core_scan(self):
        state = {}
        result = evaluate_nine_cores([candidate("mintA", "Alpha")], state, {}, now="test-time")
        self.assertIn("civilization", result)
        self.assertEqual(result["civilization"]["election_number"], 1)
        self.assertIsNotNone(result["winner"])
        self.assertIn("academy", state["nine_core_system"]["civilization"])
        self.assertIn("ecology", state["nine_core_system"]["civilization"])


if __name__ == "__main__":
    unittest.main()
