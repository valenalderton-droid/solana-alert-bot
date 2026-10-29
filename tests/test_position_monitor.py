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


if __name__ == "__main__":
    unittest.main()
