import unittest
from unittest.mock import patch

import market


def pair(address, price, liquidity, buys=1, sells=1):
    return {
        "chainId": "solana",
        "pairAddress": address,
        "baseToken": {"address": "TOKEN_MINT"},
        "priceUsd": str(price),
        "liquidity": {"usd": liquidity},
        "txns": {"h1": {"buys": buys, "sells": sells}},
    }


class TestCurrentReferencePair(unittest.TestCase):
    @patch("market.get_json")
    def test_chooses_most_liquid_active_consistent_pool(self, get_json):
        get_json.return_value = {
            "pairs": [
                pair("pool-a", 1.00, 300_000),
                pair("pool-b", 1.01, 500_000),
            ]
        }

        result = market.get_best_solana_pair("TOKEN_MINT", 100_000)

        self.assertEqual(result["pairAddress"], "pool-b")
        self.assertEqual(result["_reference_price_usd"], 1.01)

    @patch("market.get_json")
    def test_does_not_use_pool_without_recent_hourly_trades(self, get_json):
        get_json.return_value = {
            "pairs": [pair("dormant-pool", 1.00, 500_000, buys=0, sells=0)]
        }

        self.assertIsNone(market.get_best_solana_pair("TOKEN_MINT", 100_000))

    @patch("market.get_json")
    def test_ignores_price_outlier_when_consistent_pool_exists(self, get_json):
        get_json.return_value = {
            "pairs": [
                pair("outlier-pool", 1.25, 900_000),
                pair("pool-a", 1.00, 250_000),
                pair("pool-b", 1.01, 200_000),
            ]
        }

        result = market.get_best_solana_pair("TOKEN_MINT", 100_000)

        self.assertIn(result["pairAddress"], {"pool-a", "pool-b"})
        self.assertLessEqual(abs(result["_reference_price_usd"] / 1.01 - 1), 0.03)

    @patch("market.get_json")
    def test_returns_none_when_all_pool_prices_disagree(self, get_json):
        get_json.return_value = {
            "pairs": [
                pair("pool-a", 1.00, 300_000),
                pair("pool-b", 1.20, 250_000),
            ]
        }

        self.assertIsNone(market.get_best_solana_pair("TOKEN_MINT", 100_000))


if __name__ == "__main__":
    unittest.main()
