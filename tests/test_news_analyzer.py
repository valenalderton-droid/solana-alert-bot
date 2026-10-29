import unittest
from news_analyzer import analyze_news, format_news_alert


class NewsAnalyzerTests(unittest.TestCase):
    def item(self, title, description="", source="Example Source"):
        return {
            "id": "test-id",
            "title": title,
            "description": description,
            "source": source,
            "published_at": "2026-10-29T10:00:00+00:00",
            "url": "https://example.com/news",
        }

    def test_solananetwork_outage_is_bearish_and_direct(self):
        result = analyze_news(self.item("Solana network outage halts transactions"))
        self.assertEqual(result["sentiment"], "BEARISH")
        self.assertLess(result["impact_score"], 0)
        self.assertIn("DIRECT", result["relevance"])

    def test_ecosystem_project_news_is_not_automatically_direct_sol_news(self):
        result = analyze_news(self.item("Jupiter integrates a new lending feature"))
        self.assertEqual(result["sentiment"], "BULLISH")
        self.assertIn("INDIRECT", result["relevance"])

    def test_unclear_headline_is_not_forced_bullish_or_bearish(self):
        result = analyze_news(self.item("Solana community discusses future roadmap"))
        self.assertEqual(result["sentiment"], "NEUTRAL / UNCLEAR")
        self.assertEqual(result["impact_score"], 0)

    def test_alert_includes_source_link_and_caution(self):
        result = analyze_news(self.item("Solana ETF filing reported"))
        message = format_news_alert(result)
        self.assertIn("https://example.com/news", message)
        self.assertIn("Source:", message)
        self.assertIn("Do not trade on a headline alone", message)

    def test_negated_outage_headline_is_not_classified_as_bearish(self):
        result = analyze_news(self.item("Solana confirms no network outage"))
        self.assertEqual(result["sentiment"], "NEUTRAL / UNCLEAR")
        self.assertEqual(result["impact_score"], 0)


if __name__ == "__main__":
    unittest.main()
