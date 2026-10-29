import unittest

from main import build_comprehensive_analysis


class ComprehensiveAnalysisTests(unittest.TestCase):
    def candles(self, count=200):
        rows = []
        price = 1.0
        for i in range(count):
            previous = price
            price = 1.0 + i * 0.001
            rows.append({
                "timestamp": i * 3600,
                "open": previous,
                "high": max(previous, price) * 1.01,
                "low": min(previous, price) * 0.99,
                "close": price,
                "volume": 1000 + i * 2
            })
        return rows

    def test_report_includes_multiple_timeframes_and_context(self):
        report = build_comprehensive_analysis(
            candles=self.candles(),
            price=1.2,
            ensemble={
                "score": 72.5,
                "bullish_strategies": 5,
                "risk_flags": ["possible_price_chase"],
                "strategies": {"trend_following": 80, "breakout": 65}
            },
            quality={"score": 78, "grade": "B", "passed": True},
            horizon_probabilities={"4h": 0.62, "24h": 0.58},
            global_context={"btc_24h_pct": 1.2, "sol_24h_pct": 2.3, "risk_off": False},
            news_context=[{
                "title": "Protocol integration announced",
                "sentiment": "BULLISH",
                "relevance": "ecosystem",
                "url": "https://example.com/news"
            }],
            liquidity=500000,
            volume24h=800000,
            entry=1.2,
            stop=1.1,
            target=1.35
        )
        self.assertIn("بازده ۲۴ساعت", report)
        self.assertIn("بازده ۷روز", report)
        self.assertIn("حمایت تقریبی", report)
        self.assertIn("مقاومت تقریبی", report)
        self.assertIn("امتیاز استراتژی‌ها", report)
        self.assertIn("اخبار مرتبط اخیر", report)
        self.assertIn("نسبت پاداش/ریسک", report)
        self.assertIn("BTC ۲۴ساعت", report)

    def test_report_handles_missing_data_without_inventing_levels(self):
        report = build_comprehensive_analysis(
            candles=[], price=0, ensemble={}, quality={},
            global_context={}, news_context=[]
        )
        self.assertIn("قیمت مرجع: نامعتبر", report)
        self.assertIn("کندل معتبر کافی نیست", report)
        self.assertIn("تاریخچه", report)
        self.assertIn("به معنی نبود خبر منفی نیست", report)


if __name__ == "__main__":
    unittest.main()
