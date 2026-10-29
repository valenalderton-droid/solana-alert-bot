import unittest
from datetime import datetime, timezone
from research_engine import build_research_plan, _performance_topics, fetch_solana_tvl_snapshot
from unittest.mock import patch


class ResearchEngineTests(unittest.TestCase):
    def test_plan_rotates_queries_and_uses_project_context(self):
        state = {}
        projects = [{"name": "Jupiter", "symbol": "JUP", "approved_non_meme": True}]
        first = build_research_plan(state, [], projects)
        second = build_research_plan(state, [], projects)
        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 2)
        self.assertTrue(all(row[1] for row in first + second))
        self.assertIn("Jupiter", first[0][1])

    def test_recent_losses_prioritize_risk_research(self):
        outcomes = [{"return_pct": -2.0, "result": "loss"} for _ in range(7)]
        self.assertTrue({"security", "liquidity", "tokenomics", "strategy"}.issubset(
            _performance_topics(outcomes)
        ))

    def test_non_approved_projects_are_not_used_for_custom_search(self):
        state = {}
        plan = build_research_plan(
            state, [], [{"name": "Random token", "symbol": "RND", "approved_non_meme": False}]
        )
        self.assertNotIn("Random token", " ".join(row[1] for row in plan))

    def test_tvl_requires_valid_history(self):
        class Response:
            def raise_for_status(self):
                pass
            def json(self):
                return [{"date": 1, "tvl": 100}]
        with patch("research_engine.SESSION.get", return_value=Response()):
            with self.assertRaises(ValueError):
                fetch_solana_tvl_snapshot(datetime(2026, 10, 29, tzinfo=timezone.utc))


if __name__ == "__main__":
    unittest.main()
