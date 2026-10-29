import unittest
from datetime import datetime, timezone
from research_engine import build_research_plan, _performance_topics, fetch_solana_tvl_snapshot, fetch_solana_activity_metrics, assess_research_evidence
from unittest.mock import patch


class ResearchEngineTests(unittest.TestCase):
    def test_research_evidence_marks_mixed_keywords_without_claiming_verified_impact(self):
        now = datetime(2026, 10, 29, tzinfo=timezone.utc)
        result = assess_research_evidence(
            "Solana integration announced after exploit concerns",
            source="Example Publisher", published=now, now=now
        )
        self.assertEqual(result["keyword_tone"], "mixed")
        self.assertIn("exploit", result["risk_keywords"])
        self.assertIn("integration", result["positive_keywords"])
        self.assertGreaterEqual(result["evidence_quality_score"], 70)
        self.assertIn("Keyword triage only", result["assessment_note"])

    def test_research_evidence_is_cautious_with_unclassified_headlines(self):
        result = assess_research_evidence("A vague market update")
        self.assertEqual(result["keyword_tone"], "unclassified")
        self.assertEqual(result["risk_keywords"], [])
        self.assertEqual(result["positive_keywords"], [])

    def test_plan_rotates_queries_and_uses_project_context(self):
        state = {}
        projects = [{"name": "Jupiter", "symbol": "JUP", "approved_non_meme": True}]
        first = build_research_plan(state, [], projects)
        second = build_research_plan(state, [], projects)
        self.assertEqual(len(first), 3)
        self.assertEqual(len(second), 3)
        self.assertTrue(all(row[1] for row in first + second))
        self.assertIn("Jupiter", first[0][1])
        self.assertIn("Jupiter", second[0][1])

    def test_recent_losses_prioritize_risk_research(self):
        outcomes = [{"return_pct": -2.0, "result": "loss"} for _ in range(7)]
        self.assertTrue({"security", "liquidity", "tokenomics", "strategy"}.issubset(
            _performance_topics(outcomes)
        ))

    def test_labeled_losses_prioritize_risk_research_without_numeric_returns(self):
        outcomes = [{"result": "loss"} for _ in range(7)] + [{"result": "win"} for _ in range(3)]
        topics = _performance_topics(outcomes)
        self.assertIn("security", topics)
        self.assertIn("liquidity", topics)

    def test_non_approved_projects_are_not_used_for_custom_search(self):
        state = {}
        plan = build_research_plan(
            state, [], [{"name": "Random token", "symbol": "RND", "approved_non_meme": False}]
        )
        self.assertNotIn("Random token", " ".join(row[1] for row in plan))

    def test_activity_metrics_parse_valid_numeric_changes(self):
        class Response:
            def __init__(self, payload):
                self.payload = payload
            def raise_for_status(self):
                pass
            def json(self):
                return self.payload
        responses = [
            Response({"total24h": 1000000, "change_1d": -27.5}),
            Response({"total24h": 20000, "change_1d": -5.0}),
            Response({"total24h": 500000, "change_1d": -31.0}),
        ]
        with patch("research_engine.SESSION.get", side_effect=responses):
            rows, failures = fetch_solana_activity_metrics(datetime(2026, 10, 29, tzinfo=timezone.utc))
        self.assertEqual(failures, [])
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["change_24h_pct"], -27.5)
        self.assertEqual(rows[2]["metric"], "Solana perpetuals volume")

    def test_tvl_requires_valid_history(self):
        class Response:
            def raise_for_status(self):
                pass
            def json(self):
                return [{"date": 1, "tvl": 100}]
        with patch("research_engine.SESSION.get", return_value=Response()):
            with self.assertRaises(ValueError):
                fetch_solana_tvl_snapshot(datetime(2026, 10, 29, tzinfo=timezone.utc))


    def test_learning_diagnostics_prioritize_next_research_cycle(self):
        import io
        import json
        from unittest.mock import patch
        import research_engine

        original_open = open
        def fake_open(path, *args, **kwargs):
            if str(path).endswith("data/self_improvement.json"):
                return io.StringIO(json.dumps({
                    "learning_diagnostics": {"research_focus": ["liquidity"]}
                }))
            return original_open(path, *args, **kwargs)

        state = {}
        with patch("builtins.open", side_effect=fake_open):
            plan = build_research_plan(state, outcomes=[], configured_projects=[])
        self.assertEqual(plan[0][2], "liquidity")
        self.assertEqual(state["research_query_cursor"], 3)
        self.assertEqual(state["research_priority_signature"], "liquidity")


if __name__ == "__main__":
    unittest.main()
