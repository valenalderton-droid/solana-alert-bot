import unittest
from unittest.mock import patch

import main


class PositionMonitorTests(unittest.TestCase):
    def make_position(self, entry_ts):
        return {
            "signal_id": "test-signal",
            "token_address": "mint-test",
            "name": "Test Protocol",
            "symbol": "TEST",
            "timestamp": entry_ts,
            "entry": 100.0,
            "stop": 90.0,
            "target": 110.0,
            # Simulate a candle already checked earlier in the hour, then updated.
            "last_checked_ts": entry_ts + 3600,
        }

    def test_updated_current_hour_candle_can_trigger_target(self):
        entry_ts = 1_800_000_000
        candle_ts = entry_ts + 3600
        state = {"open_positions": [self.make_position(entry_ts)]}
        market_info = {
            "mint-test": {
                "candles": [{
                    "timestamp": candle_ts,
                    "open": 100.0,
                    "high": 111.0,
                    "low": 99.0,
                    "close": 108.0,
                    "volume": 1000.0,
                }]
            }
        }

        with patch.object(main, "send_telegram", return_value=True) as send, \
             patch.object(main, "append_jsonl") as append:
            main.monitor_open_positions(state, market_info)

        self.assertEqual(state["open_positions"], [])
        send.assert_called_once()
        self.assertIn("\n", send.call_args.args[0])
        self.assertIn("TAKE PROFIT reached", send.call_args.args[0])
        append.assert_called_once()

    def test_fourteen_day_exit_does_not_depend_on_indicator_success(self):
        entry_ts = 1_800_000_000
        latest_ts = entry_ts + 14 * 24 * 3600
        state = {"open_positions": [self.make_position(entry_ts)]}
        state["open_positions"][0]["last_checked_ts"] = latest_ts - 3600
        market_info = {
            "mint-test": {
                "candles": [{
                    "timestamp": latest_ts,
                    "open": 100.0,
                    "high": 105.0,
                    "low": 95.0,
                    "close": 101.0,
                    "volume": 1000.0,
                }]
            }
        }

        with patch.object(main, "latest_features", side_effect=ValueError("not enough data")), \
             patch.object(main, "send_telegram", return_value=True) as send, \
             patch.object(main, "append_jsonl"):
            main.monitor_open_positions(state, market_info)

        self.assertEqual(state["open_positions"], [])
        send.assert_called_once()
        self.assertIn("14-day time limit reached", send.call_args.args[0])


    def test_watchlist_records_shadow_observation_even_when_telegram_fails(self):
        import tempfile
        from pathlib import Path

        candles = []
        start = 1_800_000_000
        for i in range(24):
            candles.append({
                "timestamp": start + i * 3600,
                "open": 99.0,
                "high": 102.0,
                "low": 95.0,
                "close": 100.0,
                "volume": 1000.0,
            })
        project = {
            "name": "Test Protocol", "symbol": "TEST",
            "token_address": "mint-paper-test", "approved_non_meme": True
        }
        info = {
            "price": 100.0, "liquidity": 250000.0,
            "volume24h": 500000.0, "candles": candles
        }
        state = {"paper_open_signals": []}

        with tempfile.TemporaryDirectory() as temp, \\
             patch.object(main, "PAPER_PREDICTIONS_FILE", Path(temp) / "paper_predictions.jsonl"), \\
             patch.object(main, "evaluate_strategies", return_value={"score": 72, "risk_flags": []}), \\
             patch.object(main, "send_telegram", return_value=False), \\
             patch.object(main, "append_jsonl", wraps=main.append_jsonl) as append:
            main.send_watchlist_fallback(
                [project], {"mint-paper-test": info}, state,
                reason="model not ready", configured_projects=[project]
            )
            main.send_watchlist_fallback(
                [project], {"mint-paper-test": info}, state,
                reason="model not ready", configured_projects=[project]
            )

        self.assertEqual(len(state["paper_open_signals"]), 1)
        self.assertEqual(state["paper_open_signals"][0]["source"], "paper_watchlist")
        paper_writes = [
            call for call in append.call_args_list
            if str(call.args[0]).endswith("paper_predictions.jsonl")
        ]
        self.assertEqual(len(paper_writes), 1)
        self.assertIsNone(paper_writes[0].args[1]["model_probability"])

    def test_paper_outcomes_use_separate_ledger_and_do_not_change_live_stats(self):
        import json
        import tempfile
        from pathlib import Path

        entry_ts = 1_800_000_000
        signal = {
            "signal_id": "paper-test", "source": "paper_watchlist",
            "token_address": "mint-test", "name": "Test Protocol",
            "timestamp": entry_ts, "entry": 100.0, "stop": 90.0, "target": 110.0
        }
        candles = {}
        for hour in range(1, 5):
            ts = entry_ts + hour * 3600
            candles[ts] = {
                "timestamp": ts, "open": 100.0, "high": 105.0,
                "low": 95.0, "close": 100.0 + hour, "volume": 1000.0
            }
        state = {
            "paper_open_signals": [signal],
            "stats": {"signals": 7, "resolved": 2, "wins": 1, "losses": 1}
        }

        with tempfile.TemporaryDirectory() as temp:
            outcome_path = Path(temp) / "paper_outcomes.jsonl"
            main.resolve_old_signals(
                state, {"mint-test": candles},
                signal_key="paper_open_signals",
                outcomes_path=outcome_path,
                count_live_stats=False
            )
            rows = [json.loads(line) for line in outcome_path.read_text().splitlines()]

        self.assertEqual(state["paper_open_signals"], [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "paper_watchlist")
        self.assertEqual(rows[0]["result"], "timeout")
        self.assertAlmostEqual(rows[0]["return_pct"], 4.0)
        self.assertEqual(state["stats"], {"signals": 7, "resolved": 2, "wins": 1, "losses": 1})


if __name__ == "__main__":
    unittest.main()
