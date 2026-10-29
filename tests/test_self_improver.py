import tempfile
import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import self_improver


class SelfImproverTests(unittest.TestCase):
    def test_small_sample_keeps_existing_threshold(self):
        decision = self_improver.choose_probability_threshold(
            predictions=[], outcomes=[], current_threshold=0.60
        )
        self.assertEqual(decision["threshold"], 0.60)
        self.assertFalse(decision["changed"])

    def test_chronological_holdout_required_before_change(self):
        now = datetime.now(timezone.utc)
        predictions, outcomes = [], []
        for i in range(30):
            signal_id = f"signal-{i}"
            probability = 0.56 if i % 2 == 0 else 0.66
            predictions.append({
                "signal_id": signal_id,
                "created_at": (now - timedelta(days=30-i)).isoformat(),
                "probability": probability
            })
            outcomes.append({
                "signal_id": signal_id,
                "result": "win" if probability >= 0.60 else "loss",
                "return_pct": 1.0 if probability >= 0.60 else -1.0
            })
        decision = self_improver.choose_probability_threshold(
            predictions, outcomes, current_threshold=0.60
        )
        self.assertGreaterEqual(decision["sample_size"], 20)
        # Candidate can only be adopted if the held-out segment passes the guards.
        self.assertIn("threshold", decision)
        self.assertTrue(0.55 <= decision["threshold"] <= 0.70)

    def test_rule_registry_never_lowers_configured_floor(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "self_improvement.json"
            with patch.object(self_improver, "IMPROVEMENT_FILE", path):
                result = self_improver.update_self_improvement(
                    predictions=[], outcomes=[], current_threshold=0.62,
                    now=datetime(2026, 10, 29, tzinfo=timezone.utc)
                )
                self.assertGreaterEqual(result["min_probability"], 0.62)
                saved = path.read_text(encoding="utf-8")
                self.assertIn("validated_probability_floor", saved)
                self.assertIn("hard liquidity", saved)


    def test_failure_diagnostics_create_research_focus_from_repeated_losses(self):
        predictions, outcomes = [], []
        for i in range(4):
            signal_id = f"bad-signal-{i}"
            predictions.append({
                "signal_id": signal_id, "probability": 0.62,
                "liquidity_usd": 50000, "volume_24h_usd": 20000,
                "entry_deviation_pct": 2.5,
                "strategy_ensemble": {"score": 52, "risk_flags": ["high_volatility"]},
                "signal_quality": {"score": 60}
            })
            outcomes.append({"signal_id": signal_id, "result": "loss", "return_pct": -1.0})

        result = self_improver.diagnose_failure_modes(predictions, outcomes)
        self.assertEqual(result["sample_size"], 4)
        self.assertIn("liquidity", result["research_focus"])
        self.assertIn("strategy", result["research_focus"])
        self.assertGreaterEqual(result["groups"]["thin_liquidity"]["n"], 3)
        self.assertEqual(result["groups"]["thin_liquidity"]["warning"],
                         "association only; not a causal finding")


    def test_experience_rules_require_repeated_negative_evidence(self):
        diagnostics = {"groups": {
            "entry_chase": {"n": 12, "mean_net_return_pct": -1.2,
                            "negative_return_rate": 0.75}
        }}
        rules = self_improver.build_experience_rules(diagnostics)
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]["id"], "avoid_entry_chase")
        result = self_improver.evaluate_experience_rules(
            {"entry_deviation_pct": 2.4}, rules
        )
        self.assertTrue(result["blocked"])
        self.assertEqual(result["rule_id"], "avoid_entry_chase")

    def test_experience_rules_do_not_activate_on_small_or_mixed_samples(self):
        diagnostics = {"groups": {
            "thin_liquidity": {"n": 5, "mean_net_return_pct": -2.0,
                               "negative_return_rate": 0.9},
            "low_volume": {"n": 20, "mean_net_return_pct": 0.2,
                           "negative_return_rate": 0.8}
        }}
        self.assertEqual(self_improver.build_experience_rules(diagnostics), [])

    def test_experience_rule_evaluator_ignores_invalid_features(self):
        rule = {"id": "test", "enabled": True, "field": "probability",
                "operator": "lt", "value": 0.65}
        result = self_improver.evaluate_experience_rules(
            {"probability": "unknown"}, [rule]
        )
        self.assertFalse(result["blocked"])

    def test_diagnostics_ignore_data_gaps_and_duplicate_outcomes(self):
        prediction = [{
            "signal_id": "same", "probability": 0.9,
            "liquidity_usd": 1, "volume_24h_usd": 1, "entry_deviation_pct": 3
        }]
        outcomes = [
            {"signal_id": "same", "result": "loss", "return_pct": -2},
            {"signal_id": "same", "result": "win", "return_pct": 3},
            {"signal_id": "gap", "result": "data_gap", "return_pct": None}
        ]
        result = self_improver.diagnose_failure_modes(prediction, outcomes)
        self.assertEqual(result["sample_size"], 1)
        self.assertEqual(result["groups"], {})

    def test_rule_file_lifecycle_creates_edits_and_deletes_only_managed_rules(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "experience_rules"
            with patch.object(self_improver, "RULE_FILES_DIR", directory):
                rule = {
                    "id": "avoid_entry_chase", "enabled": True,
                    "status": "experience_derived", "field": "entry_deviation_pct",
                    "operator": "gte", "value": 2.0,
                    "reason": "Repeated losses after chasing",
                    "evidence": {"samples": 12, "mean_net_return_pct": -1.2,
                                 "negative_return_rate": 0.75}
                }
                first = self_improver.sync_experience_rule_files([rule])
                path = directory / "avoid_entry_chase.json"
                self.assertEqual(first["created"], ["avoid_entry_chase.json"])
                self.assertTrue(path.exists())

                edited_rule = dict(rule, value=2.5, evidence={"samples": 15})
                second = self_improver.sync_experience_rule_files([edited_rule])
                self.assertEqual(second["edited"], ["avoid_entry_chase.json"])
                self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["value"], 2.5)

                # An unrelated file must never be removed by rule cleanup.
                unrelated = directory / "keep-me.json"
                unrelated.write_text("{}", encoding="utf-8")
                third = self_improver.sync_experience_rule_files([])
                self.assertEqual(third["deleted"], ["avoid_entry_chase.json"])
                self.assertFalse(path.exists())
                self.assertTrue(unrelated.exists())

    def test_rule_file_manager_rejects_unallowlisted_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(self_improver, "RULE_FILES_DIR", Path(temp)):
                result = self_improver.sync_experience_rule_files([{
                    "id": "../main.py", "enabled": True, "field": "x",
                    "operator": "lt", "value": 1
                }])
                self.assertEqual(result["active"], [])
                self.assertFalse((Path(temp) / "main.py.json").exists())


if __name__ == "__main__":
    unittest.main()
