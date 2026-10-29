import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import market


class DiscoveryCacheTests(unittest.TestCase):
    def test_fresh_cache_avoids_repeating_discovery_searches(self):
        project = {
            "name": "Jupiter",
            "symbol": "JUP",
            "token_address": "mint-jup-test",
            "approved_non_meme": True,
        }
        filters = {
            "discovery_version": 3,
            "min_liquidity": 100000.0,
            "min_volume_24h": 50000.0,
            "min_age_days": 30,
            "max_projects": 500,
        }
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "discovery_cache.json"
            cache_path.write_text(json.dumps({
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "filters": filters,
                "projects": [project],
            }), encoding="utf-8")
            with patch.object(market, "DISCOVERY_CACHE_FILE", cache_path), \
                 patch.object(market, "get_json", side_effect=AssertionError("API should not be called")), \
                 patch.dict("os.environ", {"DISCOVERY_CACHE_TTL_MINUTES": "45"}):
                result = market.discover_solana_projects()

        self.assertEqual(result, [project])


if __name__ == "__main__":
    unittest.main()
