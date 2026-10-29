import unittest
from paper_portfolio import get_portfolio, open_paper_trade, update_paper_portfolio, portfolio_snapshot, adaptive_risk_fraction


class PaperPortfolioTests(unittest.TestCase):
    def prediction(self, **overrides):
        row = {"signal_id": "mint:100:BUY", "token_address": "mint", "name": "Protocol",
               "symbol": "PRO", "timestamp": 100, "entry": 100.0, "stop": 95.0,
               "target": 110.0, "probability": 0.7, "rank_score": 80}
        row.update(overrides)
        return row

    def test_starts_with_virtual_cash(self):
        state = {}
        p = get_portfolio(state, 10000)
        self.assertEqual(p["cash_usd"], 10000)
        self.assertEqual(p["open_positions"], [])
        self.assertEqual(p["closed_trades"], [])

    def test_risk_and_position_size_are_bounded(self):
        state = {}
        result = open_paper_trade(state, self.prediction())
        self.assertTrue(result["opened"])
        p = state["paper_portfolio"]
        pos = p["open_positions"][0]
        self.assertLessEqual(pos["notional_usd"], 2000)
        self.assertLessEqual(pos["notional_usd"] * 0.05, 100.01)
        self.assertLess(p["cash_usd"], 10000)

    def test_invalid_duplicate_and_capacity_rules(self):
        state = {}
        self.assertFalse(open_paper_trade(state, self.prediction(stop=101))["opened"])
        self.assertTrue(open_paper_trade(state, self.prediction())["opened"])
        self.assertEqual(open_paper_trade(state, self.prediction(signal_id="another"))["reason"], "token_already_open")
        for i in range(1, 5):
            open_paper_trade(state, self.prediction(signal_id=f"id{i}", token_address=f"mint{i}"))
        self.assertEqual(len(state["paper_portfolio"]["open_positions"]), 5)
        self.assertEqual(open_paper_trade(state, self.prediction(signal_id="six", token_address="mint6"))["reason"], "max_open_positions")

    def test_target_exit_books_net_profit_and_fees(self):
        state = {}
        open_paper_trade(state, self.prediction())
        summary = update_paper_portfolio(state, {"mint": {"price": 110, "candles": [
            {"timestamp": 100, "low": 99, "high": 101, "close": 100},
            {"timestamp": 3700, "low": 108, "high": 111, "close": 110}]}})
        p = state["paper_portfolio"]
        self.assertEqual(len(p["open_positions"]), 0)
        self.assertEqual(p["closed_trades"][0]["exit_reason"], "target")
        self.assertGreater(p["closed_trades"][0]["pnl_usd"], 0)
        self.assertGreater(p["fees_paid_usd"], 0)
        self.assertEqual(summary["closed_trades"], 1)

    def test_same_candle_stop_and_target_assumes_stop(self):
        state = {}
        open_paper_trade(state, self.prediction())
        update_paper_portfolio(state, {"mint": {"candles": [
            {"timestamp": 3700, "low": 94, "high": 111, "close": 100}]}})
        trade = state["paper_portfolio"]["closed_trades"][0]
        self.assertEqual(trade["exit_reason"], "stop_and_target_same_candle_assume_stop")
        self.assertLess(trade["pnl_usd"], 0)

    def test_paper_results_reduce_risk_after_persistent_losses(self):
        portfolio = get_portfolio({})
        portfolio["closed_trades"] = [{"pnl_usd": -1.0} for _ in range(20)]
        risk, diagnostics = adaptive_risk_fraction(portfolio)
        self.assertEqual(risk, 0.005)
        self.assertEqual(diagnostics["mode"], "defensive")

    def test_learning_warms_up_before_changing_risk(self):
        portfolio = get_portfolio({})
        portfolio["closed_trades"] = [{"pnl_usd": -1.0} for _ in range(19)]
        risk, diagnostics = adaptive_risk_fraction(portfolio)
        self.assertEqual(risk, 0.01)
        self.assertEqual(diagnostics["mode"], "warmup")

    def test_mark_to_market_tracks_drawdown(self):
        state = {}
        open_paper_trade(state, self.prediction())
        summary = portfolio_snapshot(state, {"mint": {"price": 90}})
        self.assertLess(summary["equity_usd"], summary["initial_cash_usd"])
        self.assertGreater(summary["max_drawdown_pct"], 0)


if __name__ == "__main__":
    unittest.main()
