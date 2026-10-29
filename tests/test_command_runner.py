import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import command_runner
import main as scanner


class CommandRunnerTests(unittest.TestCase):
    def test_compact_reply_selects_highest_probability_candidate(self):
        messages = [
            "SOLANA BUY WATCH\nProject: Alpha (ALP)\nEntry reference: $1.2\n"
            "Stop reference: $1.1 (5%)\nTarget reference: $1.4 (4%)\n"
            "Combined model probability: 61.0%\nMint: mint-alpha",
            "SOLANA BUY WATCH\nProject: Beta (BET)\nEntry reference: $2.2\n"
            "Stop reference: $2.0 (5%)\nTarget reference: $2.6 (4%)\n"
            "Combined model probability: 72.0%\nMint: mint-beta",
        ]
        reply = command_runner.concise_reply(messages)
        self.assertIn("Beta (BET)", reply)
        self.assertIn("نقطه ورود: $2.2", reply)
        self.assertIn("نقطه فروش هدف (Take Profit): $2.6", reply)
        self.assertIn("mint-beta", reply)
        self.assertNotIn("Combined model probability", reply)

    def test_no_candidate_returns_short_code(self):
        self.assertEqual(command_runner.concise_reply([]), "NO_SETUP")

    def test_candidate_ranking_uses_horizon_consensus_and_liquidity(self):
        weak_high_score = (
            "SOLANA BUY WATCH\nProject: Alpha (ALP)\nEntry reference: $1.2\n"
            "Stop reference: $1.1 (5%)\nTarget reference: $1.4 (4%)\n"
            "Combined model probability: 75.0%\n"
            'Probability by horizon: {"4h": 0.35, "12h": 0.45, "24h": 0.8}\n'
            "Liquidity: $100,000\n24h volume: $50,000\nMint: mint-alpha"
        )
        consistent = (
            "SOLANA BUY WATCH\nProject: Beta (BET)\nEntry reference: $2.2\n"
            "Stop reference: $2.0 (5%)\nTarget reference: $2.6 (4%)\n"
            "Combined model probability: 72.0%\n"
            'Probability by horizon: {"4h": 0.72, "12h": 0.70, "24h": 0.68}\n'
            "Liquidity: $500,000\n24h volume: $300,000\nMint: mint-beta"
        )
        reply = command_runner.concise_reply([weak_high_score, consistent])
        self.assertIn("Beta (BET)", reply)
        self.assertIn("mint-beta", reply)


    def test_detects_ranked_buy_watch_alert(self):
        message = (
            "SOLANA TOP-RANKED BUY WATCH — NOT AN AUTOMATIC TRADE\n\n"
            "Project: Example (EXM)\n"
            "Entry reference: $1.25\n"
            "Stop reference: $1.18 (5.00%)\n"
            "Target reference: $1.50 (20.00%)\n"
            "Combined model probability: 65.0%\n"
            'Probability by horizon: {"4h": 0.65, "24h": 0.61}\n'
            "Liquidity: $250,000\n"
            "24h volume: $500,000\n"
            "Mint: ExampleMintAddress\n"
        )
        self.assertTrue(command_runner.is_buy_alert(message))
        reply = command_runner.concise_reply([message])
        self.assertIn("Example (EXM)", reply)
        self.assertIn("نقطه ورود: $1.25", reply)
        self.assertNotEqual(reply, "NO_SETUP")

    def test_scan_rotation_covers_universe_without_duplicates(self):
        projects = [{"token_address": f"mint-{i}"} for i in range(10)]
        first, cursor = scanner.select_projects_for_scan(projects, cursor=0, max_projects=5)
        second, next_cursor = scanner.select_projects_for_scan(projects, cursor=cursor, max_projects=5)
        first_addresses = [p["token_address"] for p in first]
        second_addresses = [p["token_address"] for p in second]
        self.assertEqual(len(first_addresses), len(set(first_addresses)))
        self.assertEqual(len(second_addresses), len(set(second_addresses)))
        self.assertEqual(set(first_addresses + second_addresses), {p["token_address"] for p in projects})
        self.assertEqual(next_cursor, 0)

    def test_scan_rotation_always_includes_open_positions(self):
        projects = [{"token_address": f"mint-{i}"} for i in range(8)]
        selected, _ = scanner.select_projects_for_scan(
            projects, cursor=0, max_projects=3,
            always_include_addresses={"mint-7"}
        )
        addresses = [p["token_address"] for p in selected]
        self.assertEqual(len(addresses), 4)
        self.assertIn("mint-7", addresses)

    def test_watchlist_status_is_not_misclassified_as_buy(self):
        message = "SOLANA TOP WATCHLIST — NO VALID BUY SETUP\\nThis is NOT a buy signal."
        self.assertFalse(command_runner.is_buy_alert(message))
        self.assertFalse(command_runner.is_trade_alert(message))

    def test_explicit_sell_exit_alert_is_allowed(self):
        message = (
            "SOLANA SELL / EXIT ALERT — NO AUTOMATIC TRADE\\n"
            "Exit reason: stop reference reached."
        )
        self.assertTrue(command_runner.is_sell_alert(message))
        self.assertTrue(command_runner.is_trade_alert(message))

    def test_watchlist_and_data_status_reports_are_forwarded(self):
        for header in (
            "SOLANA TOP WATCHLIST — NO VALID BUY SETUP",
            "SOLANA WATCHLIST STATUS — MARKET DATA UNAVAILABLE",
            "SOLANA BOT STATUS — NO TOKEN CAN BE RANKED",
            "SOLANA TOP-RANKED PROJECT — NOT A BUY SIGNAL",
            "SOLANA TOP-RANKED PROJECT — DATA UNAVAILABLE",
            "SOLANA SCAN STATUS — NO PROJECT CAN BE RANKED",
        ):
            with patch.object(command_runner, "read_state", return_value={"open_positions": []}), \
                 patch.object(command_runner, "refresh_news_if_due"), \
                 patch.object(command_runner, "run_self_improvement_if_due"), \
                 patch.object(scanner, "main", side_effect=lambda: scanner.send_telegram(header)), \
                 patch.object(scanner, "send_telegram", return_value=True) as sender, \
                 patch.object(command_runner, "telegram_send") as final_sender:
                command_runner.main()
            sender.assert_called_once_with(header)
            final_sender.assert_not_called()

    def test_routine_status_is_not_a_trade_alert(self):
        for message in (
            "SOLANA BOT STATUS — NO TOKEN CAN BE RANKED",
            "فعلاً ستاپ خرید معتبر پیدا نشد.",
            "SOLANA NEWS IMPACT — RESEARCH, NOT A TRADE SIGNAL",
        ):
            self.assertFalse(command_runner.is_trade_alert(message), message)

    def test_automatic_mode_does_not_poll_for_pay_commands(self):
        original_sender = scanner.send_telegram
        with patch.object(command_runner, "read_state", return_value={"open_positions": []}), \
             patch.object(command_runner, "refresh_news_if_due"), \
             patch.object(command_runner, "run_self_improvement_if_due"), \
             patch.object(scanner, "main"), \
             patch.object(scanner, "send_telegram", original_sender):
            # A scheduled run should execute without checking for a Telegram trigger.
            command_runner.main()

    def test_no_setup_report_is_sent_even_during_watchlist_cooldown(self):
        recent = datetime.now(timezone.utc).isoformat()
        with patch.object(command_runner, "read_state", side_effect=[
                 {"open_positions": []},
                 {"open_positions": [], "last_watchlist_ping_utc": recent}
             ]), \
             patch.object(command_runner, "refresh_news_if_due"), \
             patch.object(command_runner, "run_self_improvement_if_due"), \
             patch.object(scanner, "main"), \
             patch.object(scanner, "send_telegram"), \
             patch.object(command_runner, "telegram_send", return_value=True) as send:
            command_runner.main()
        send.assert_called_once()
        self.assertIn("هیچ ستاپ خرید معتبری", send.call_args.args[2])

    def test_recent_cached_candles_are_exit_only_fallback(self):
        now = datetime(2026, 10, 29, 12, 30, tzinfo=timezone.utc)
        current_hour = int(now.timestamp() // 3600) * 3600
        candles = []
        for i in range(40):
            price = 10.0 + i * 0.1
            candles.append({
                "timestamp": current_hour - (40 - i) * 3600,
                "open": price,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "volume": 1000.0,
            })
        store = {"mint-open": {row["timestamp"]: row for row in candles}}
        info = scanner.recent_cached_position_market_info("mint-open", store, now=now)
        self.assertIsNotNone(info)
        self.assertTrue(info["data_is_cached"])
        self.assertIsNone(info["pair"])
        self.assertEqual(info["liquidity"], 0.0)
        self.assertLessEqual(info["candle_age_hours"], 3.0)

    def test_stale_cached_candles_are_rejected_for_exit_fallback(self):
        now = datetime(2026, 10, 29, 12, 30, tzinfo=timezone.utc)
        current_hour = int(now.timestamp() // 3600) * 3600
        candles = []
        for i in range(40):
            price = 10.0 + i * 0.1
            candles.append({
                "timestamp": current_hour - 5 * 3600 - (39 - i) * 3600,
                "open": price,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "volume": 1000.0,
            })
        store = {"mint-open": {row["timestamp"]: row for row in candles}}
        self.assertIsNone(
            scanner.recent_cached_position_market_info("mint-open", store, now=now)
        )

    def test_sender_is_restored_when_scan_raises(self):
        original_sender = scanner.send_telegram
        with patch.object(command_runner, "read_state", return_value={"open_positions": []}), \
             patch.object(command_runner, "refresh_news_if_due"), \
             patch.object(command_runner, "run_self_improvement_if_due"), \
             patch.object(scanner, "main", side_effect=RuntimeError("test failure")), \
             patch.object(command_runner, "telegram_send") as send:
            # The runner should catch a scanner crash, restore the Telegram sender,
            # and attempt to return an explicit failure reply rather than crash silently.
            command_runner.main()
        self.assertIs(scanner.send_telegram, original_sender)
        send.assert_called_once()
        self.assertIn("خطا", send.call_args.args[2])


    def test_telegram_send_retries_transient_server_error(self):
        temporary = Mock()
        temporary.status_code = 500
        temporary.text = "temporary server error"
        temporary.json.return_value = {}
        success = Mock()
        success.status_code = 200
        success.raise_for_status.return_value = None
        success.json.return_value = {"ok": True}
        with patch.object(command_runner.requests, "post", side_effect=[temporary, success]) as post, \
             patch.object(command_runner.time, "sleep"):
            self.assertTrue(command_runner.telegram_send("token", "123", "hello"))
        self.assertEqual(post.call_count, 2)

    def test_telegram_send_reports_missing_credentials_without_network_call(self):
        with patch.object(command_runner.requests, "post") as post:
            with self.assertRaisesRegex(RuntimeError, "TELEGRAM_BOT_TOKEN"):
                command_runner.telegram_send("", "123", "hello")
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
