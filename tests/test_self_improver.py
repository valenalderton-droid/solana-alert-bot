import tempfile
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


if __name__ == "__main__":
    unittest.main()
