import unittest

from market_university import (
    DEPARTMENTS, ensure_university, update_university,
    education_score_bonus, university_summary,
)


CORE_SPECS = [
    {"id": "trend"}, {"id": "breakout"}, {"id": "momentum"},
    {"id": "volume"}, {"id": "structure"}, {"id": "risk"},
    {"id": "model"}, {"id": "quality"}, {"id": "context"},
]


def make_system():
    return {"cores": {
        spec["id"]: {
            "portfolio": {"initial_cash_usd": 10000, "cash_usd": 10000, "closed_trades": []},
            "decisions": [], "genome": {"threshold": 60},
        } for spec in CORE_SPECS
    }}


class MarketUniversityTests(unittest.TestCase):
    def test_city_has_eight_financial_departments_and_faculty(self):
        civ = {}
        uni = ensure_university(civ, CORE_SPECS)
        self.assertEqual(len(DEPARTMENTS), 8)
        self.assertGreaterEqual(len(uni["faculty"]), 8)
        self.assertEqual(len(uni["students"]), 9)
        self.assertIn("risk_school", uni["departments"])

    def test_students_take_courses_and_earn_credits_over_time(self):
        system, civ = make_system(), {}
        for i in range(1, 6):
            update_university(system, civ, CORE_SPECS, f"t{i}")
        uni = civ["universities"]
        self.assertGreaterEqual(sum(s["credits"] for s in uni["students"].values()), 27)
        self.assertTrue(all(s["completed_courses"] for s in uni["students"].values()))

    def test_resolved_paper_trade_is_recorded_once_and_changes_major_skills(self):
        system, civ = make_system(), {}
        update_university(system, civ, CORE_SPECS, "t1")
        system["cores"]["trend"]["portfolio"]["closed_trades"] = [
            {"pnl_usd": 12.0, "token_address": "mint1"},
            {"pnl_usd": -4.0, "token_address": "mint2"},
        ]
        update_university(system, civ, CORE_SPECS, "t2")
        student = civ["universities"]["students"]["trend"]
        self.assertEqual(student["last_seen_trade_count"], 2)
        self.assertEqual(len([r for r in student["transcript"] if r["type"] == "resolved_paper_case"]), 2)
        self.assertEqual(len(civ["universities"]["research"]["papers"]), 1)
        update_university(system, civ, CORE_SPECS, "t3")
        self.assertEqual(len([r for r in student["transcript"] if r["type"] == "resolved_paper_case"]), 2)

    def test_education_bonus_is_bounded_and_never_large_enough_to_bypass_quorum(self):
        system, civ = make_system(), {}
        update_university(system, civ, CORE_SPECS, "t1")
        student = civ["universities"]["students"]["trend"]
        student["skills"] = {k: 100 for k in student["skills"]}
        candidate = {"prediction": {"strategy_ensemble": {"strategies": {
            "trend_following": 100, "market_structure": 100, "breakout": 100,
        }}}}
        bonus = education_score_bonus(system, "trend", candidate)
        self.assertLessEqual(bonus, 2.0)
        self.assertGreaterEqual(bonus, -2.0)

    def test_summary_reports_courses_research_and_student_count(self):
        system, civ = make_system(), {}
        update_university(system, civ, CORE_SPECS, "t1")
        summary = university_summary(civ)
        self.assertEqual(summary["departments"], 8)
        self.assertEqual(summary["students"], 9)
        self.assertIn("average_gpa", summary)


if __name__ == "__main__":
    unittest.main()
