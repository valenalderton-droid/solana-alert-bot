import unittest
from datetime import datetime, timezone, timedelta
from adaptive_policy import derive_policy


class AdaptivePolicyTests(unittest.TestCase):
    def test_small_sample_does_not_change_base_threshold(self):
        result = derive_policy([], [], base_probability=0.60, base_cooldown_hours=6)
        self.assertEqual(result["min_probability"], 0.60)
        self.assertEqual(result["cooldown_hours"], 6)

    def test_poor_outcomes_make_policy_stricter(self):
        outcomes = [{"result": "loss"} for _ in range(10)] + [{"result": "win"} for _ in range(2)]
        result = derive_policy(outcomes, [], base_probability=0.60, base_cooldown_hours=6)
        self.assertEqual(result["min_probability"], 0.65)
        self.assertEqual(result["cooldown_hours"], 8)

    def test_strong_outcomes_adjust_policy_only_slightly(self):
        outcomes = [{"result": "win"} for _ in range(8)] + [{"result": "loss"} for _ in range(4)]
        result = derive_policy(outcomes, [], base_probability=0.60, base_cooldown_hours=6)
        self.assertEqual(result["min_probability"], 0.58)
        self.assertEqual(result["cooldown_hours"], 5)

    def test_direct_bearish_news_temporarily_raises_threshold(self):
        now = datetime.now(timezone.utc)
        news = [{
            "observed_at": (now - timedelta(hours=1)).isoformat(),
            "sentiment": "BEARISH",
            "relevance": "DIRECT — Solana network / protocol"
        } for _ in range(2)]
        result = derive_policy([], news, base_probability=0.60, now=now)
        self.assertEqual(result["min_probability"], 0.63)
        self.assertEqual(result["direct_bearish_news_24h"], 2)

    def test_low_confidence_rumors_do_not_trigger_risk_off(self):
        now = datetime.now(timezone.utc)
        news = [{
            "observed_at": (now - timedelta(hours=1)).isoformat(),
            "sentiment": "BEARISH",
            "relevance": "DIRECT — Solana network / protocol",
            "confidence": "LOW — unconfirmed wording",
            "feed_category": "Solana security"
        } for _ in range(3)]
        result = derive_policy([], news, base_probability=0.60, now=now)
        self.assertEqual(result["direct_bearish_news_24h"], 0)
        self.assertEqual(result["min_probability"], 0.60)

    def test_global_bearish_news_temporarily_raises_threshold(self):
        now = datetime.now(timezone.utc)
        news = [
            {
                "observed_at": (now - timedelta(hours=2)).isoformat(),
                "sentiment": "BEARISH",
                "feed_category": "Macro / global markets",
                "relevance": "UNCERTAIN"
            },
            {
                "observed_at": (now - timedelta(hours=1)).isoformat(),
                "sentiment": "BEARISH",
                "feed_category": "Geopolitics / global risk",
                "relevance": "UNCERTAIN"
            }
        ]
        result = derive_policy([], news, base_probability=0.60, now=now)
        self.assertEqual(result["global_bearish_news_24h"], 2)
        self.assertEqual(result["min_probability"], 0.64)
        self.assertEqual(result["cooldown_hours"], 7)

    def test_old_headline_is_not_treated_as_fresh_when_discovered_late(self):
        now = datetime.now(timezone.utc)
        news = [{
            "observed_at": now.isoformat(),
            "published_at": (now - timedelta(hours=30)).isoformat(),
            "sentiment": "BEARISH",
            "feed_category": "Geopolitics / global risk",
            "relevance": "UNCERTAIN"
        }]
        result = derive_policy([], news, base_probability=0.60, now=now)
        self.assertEqual(result["global_bearish_news_24h"], 0)
        self.assertEqual(result["min_probability"], 0.60)

    def test_large_tvl_drop_tightens_entries_but_tvl_rise_does_not_loosen(self):
        now = datetime.now(timezone.utc)
        research = [{
            "metric": "Solana DeFi TVL",
            "change_24h_pct": -16.0,
            "observed_at": now.isoformat()
        }]
        result = derive_policy([], [], base_probability=0.60, now=now,
                               research_observations=research)
        self.assertEqual(result["min_probability"], 0.65)
        self.assertEqual(result["cooldown_hours"], 8)
        positive = [{**research[0], "change_24h_pct": 18.0}]
        unchanged = derive_policy([], [], base_probability=0.60, now=now,
                                  research_observations=positive)
        self.assertEqual(unchanged["min_probability"], 0.60)

    def test_non_target_outcomes_do_not_count(self):
        outcomes = [{"result": "timeout"} for _ in range(20)]
        result = derive_policy(outcomes, [], base_probability=0.60)
        self.assertEqual(result["resolved_outcomes_used"], 0)
        self.assertEqual(result["min_probability"], 0.60)


if __name__ == "__main__":
    unittest.main()
