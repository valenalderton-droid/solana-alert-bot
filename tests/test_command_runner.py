import unittest
from unittest.mock import Mock, patch

import command_runner


class CommandRunnerTests(unittest.TestCase):
    def test_only_exact_pay_command_authorizes_scan(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "ok": True,
            "result": [
                {"update_id": 12, "message": {"chat": {"id": 123}, "text": "hello"}},
                {"update_id": 13, "message": {"chat": {"id": 123}, "text": "//Pay"}},
            ],
        }
        send_response = Mock()
        send_response.raise_for_status.return_value = None
        send_response.json.return_value = {"ok": True}
        state = {}
        with patch.dict("os.environ", {
            "TELEGRAM_BOT_TOKEN": "test-token",
            "TELEGRAM_CHAT_ID": "123",
        }), patch.object(command_runner.requests, "get", return_value=response), \
             patch.object(command_runner.requests, "post", return_value=send_response), \
             patch.object(command_runner, "save_state"):
            self.assertTrue(command_runner.poll_pay(state))
        self.assertEqual(state["telegram_update_offset"], 13)

    def test_rejects_other_command_spellings(self):
        for command in ("/Pay", "//pay", "/pay", "Pay", "//Pay@SomeBot"):
            response = Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {
                "ok": True,
                "result": [
                    {"update_id": 30, "message": {"chat": {"id": 123}, "text": command}},
                ],
            }
            send_response = Mock()
            send_response.raise_for_status.return_value = None
            send_response.json.return_value = {"ok": True}
            with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "test-token", "TELEGRAM_CHAT_ID": "123"}), patch.object(command_runner.requests, "get", return_value=response), patch.object(command_runner.requests, "post", return_value=send_response), patch.object(command_runner, "save_state"):
                self.assertFalse(command_runner.poll_pay({}), command)

    def test_other_chat_cannot_trigger_scan(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "ok": True,
            "result": [
                {"update_id": 20, "message": {"chat": {"id": 999}, "text": "/Pay"}},
            ],
        }
        with patch.dict("os.environ", {
            "TELEGRAM_BOT_TOKEN": "test-token",
            "TELEGRAM_CHAT_ID": "123",
        }), patch.object(command_runner.requests, "get", return_value=response), \
             patch.object(command_runner, "save_state"):
            self.assertFalse(command_runner.poll_pay({}))

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
        self.assertIn("خرید: $2.2", reply)
        self.assertIn("فروش: $2.6", reply)
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
        self.assertIn("خرید: $1.25", reply)
        self.assertNotEqual(reply, "NO_SETUP")

    def test_watchlist_status_is_not_misclassified_as_buy(self):
        message = "SOLANA TOP WATCHLIST — NO VALID BUY SETUP\\nThis is NOT a buy signal."
        self.assertFalse(command_runner.is_buy_alert(message))


if __name__ == "__main__":
    unittest.main()
