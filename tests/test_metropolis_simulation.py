import unittest

from metropolis_simulation import (
    DISTRICTS, ensure_metropolis, update_metropolis, metropolis_summary,
)
from core_civilization import run_civilization_cycle
from nine_core_engine import CORE_SPECS


def make_system():
    return {"cores": {
        spec["id"]: {
            "portfolio": {"initial_cash_usd": 10000, "cash_usd": 10000, "closed_trades": []},
            "decisions": [], "genome": {"threshold": 60},
        } for spec in CORE_SPECS
    }}


class MetropolisSimulationTests(unittest.TestCase):
    def test_city_has_multiple_districts_services_and_residents(self):
        civ = {}
        city = ensure_metropolis(civ, CORE_SPECS)
        self.assertGreaterEqual(len(DISTRICTS), 7)
        self.assertGreaterEqual(len(city["services"]), 7)
        self.assertEqual(len(city["residents"]), 9)
        self.assertIn("risk_controls", city["supply_chains"])

    def test_scan_updates_are_persistent_and_collaboration_is_recorded(self):
        system, civ = make_system(), {}
        update_metropolis(system, civ, CORE_SPECS, "t1")
        for i in range(2, 9):
            update_metropolis(system, civ, CORE_SPECS, f"t{i}")
        summary = metropolis_summary(civ)
        self.assertEqual(summary["residents"], 9)
        self.assertGreater(summary["collaborations"], 0)
        self.assertEqual(civ["metropolis"]["scan_count"], 8)

    def test_daily_tick_updates_city_budget_services_and_research(self):
        system, civ = make_system(), {}
        summary = None
        for i in range(1, 97):
            summary = update_metropolis(system, civ, CORE_SPECS, f"t{i}")
        city = civ["metropolis"]
        self.assertEqual(summary["simulated_day"], 1)
        self.assertGreaterEqual(summary["treasury_usd"], 0)
        self.assertGreater(summary["proposals"], 0)
        self.assertGreaterEqual(summary["experiments"], 1)
        self.assertEqual(len(city["history"]), 1)
        self.assertTrue(all(0 <= s["satisfaction"] <= 100 for s in city["services"].values()))

    def test_closed_trades_affect_resident_wellbeing_only_once_per_new_count(self):
        system, civ = make_system(), {}
        update_metropolis(system, civ, CORE_SPECS, "t1")
        system["cores"]["trend"]["portfolio"]["closed_trades"] = [
            {"pnl_usd": 12}, {"pnl_usd": -3},
        ]
        update_metropolis(system, civ, CORE_SPECS, "t2")
        resident = civ["metropolis"]["residents"]["trend"]
        contributions = resident["research_contributions"]
        update_metropolis(system, civ, CORE_SPECS, "t3")
        self.assertEqual(resident["research_contributions"], contributions)
        self.assertEqual(contributions, 2)

    def test_civilization_report_includes_metropolis_metrics(self):
        system = make_system()
        report = run_civilization_cycle(system, CORE_SPECS, "t1")
        self.assertIn("metropolis", report["civilization"])
        self.assertEqual(report["civilization"]["metropolis"]["districts"], 7)


if __name__ == "__main__":
    unittest.main()
