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
        state = {}
        with patch.dict("os.environ", {
            "TELEGRAM_BOT_TOKEN": "test-token",
            "TELEGRAM_CHAT_ID": "123",
        }), patch.object(command_runner.requests, "get", return_value=response), \
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
            with patch.dict("os.environ", {
                "TELEGRAM_BOT_TOKEN": "test-token",
                "TELEGRAM_CHAT_ID": "123",
            }), patch.object(command_runner.requests, "get", return_value=response), \\
                 patch.object(command_runner, "save_state"):
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


if __name__ == "__main__":
    unittest.main()
