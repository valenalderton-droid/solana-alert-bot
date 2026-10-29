
import requests
from datetime import datetime, timezone

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "SolanaProjectTrader/1.0"})


def get_json(url, params=None):
    response = SESSION.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def get_best_solana_pair(token_address, min_liquidity=100000):
    url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
    data = get_json(url)
    pairs = data.get("pairs") or []

    eligible = []
    for pair in pairs:
        if pair.get("chainId") != "solana":
            continue

        base = pair.get("baseToken") or {}
        if base.get("address", "").lower() != token_address.lower():
            continue

        liquidity = float((pair.get("liquidity") or {}).get("usd") or 0)
        price = float(pair.get("priceUsd") or 0)

        if liquidity < min_liquidity or price <= 0:
            continue

        pair["_liquidity_usd"] = liquidity
        eligible.append(pair)

    if not eligible:
        return None

    return max(eligible, key=lambda p: p["_liquidity_usd"])


def get_hourly_candles(pool_address, limit=1000):
    url = (
        "https://api.geckoterminal.com/api/v2/networks/solana/"
        f"pools/{pool_address}/ohlcv/hour"
    )
    params = {
        "aggregate": 1,
        "limit": min(int(limit), 1000),
        "currency": "usd",
        "token": "base"
    }

    data = get_json(url, params=params)
    items = (
        data.get("data", {})
        .get("attributes", {})
        .get("ohlcv_list", [])
    )

    candles = []
    for item in items:
        if len(item) < 6:
            continue
        candles.append({
            "timestamp": int(item[0]),
            "open": float(item[1]),
            "high": float(item[2]),
            "low": float(item[3]),
            "close": float(item[4]),
            "volume": float(item[5])
        })

    candles.sort(key=lambda c: c["timestamp"])

    # Remove duplicate timestamps.
    unique = {c["timestamp"]: c for c in candles}
    return [unique[t] for t in sorted(unique)]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


# Discover established Solana ecosystem tokens from protocol/category searches.
# This is a heuristic filter, not a guarantee that every meme token is excluded.
DISCOVERY_QUERIES = [
    "Jupiter Solana", "Raydium Solana", "Orca Solana", "Kamino Solana",
    "Jito Solana", "Marinade Solana", "Drift Solana", "Pyth Solana",
    "Meteora Solana", "Sanctum Solana", "Solana lending", "Solana DeFi",
    "Solana oracle", "Solana infrastructure", "Solana perpetuals",
    "Solana liquid staking", "Solana RWA", "Solana payments",
    "Solana bridge", "Solana governance"
]

MEME_TERMS = {
    "meme", "memecoin", "inu", "doge", "pepe", "bonk", "wif", "shib",
    "floki", "wojak", "frog", "dog", "cat", "elon", "trump", "moon",
    "rocket", "pump", "kitty", "inu"
}


def discover_solana_projects(min_liquidity=100000, min_volume_24h=50000,
                             min_age_days=30, max_projects=80):
    """Return a deduplicated list of candidate non-meme Solana projects."""
    found = {}
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    for query in DISCOVERY_QUERIES:
        try:
            payload = get_json(
                "https://api.dexscreener.com/latest/dex/search",
                params={"q": query}
            )
            for pair in payload.get("pairs") or []:
                if pair.get("chainId") != "solana":
                    continue

                token = pair.get("baseToken") or {}
                address = str(token.get("address") or "").strip()
                name = str(token.get("name") or "").strip()
                symbol = str(token.get("symbol") or "").strip()
                if not address or not name or not symbol:
                    continue

                text = (name + " " + symbol).lower()
                words = set(text.replace("-", " ").replace("_", " ").split())
                if words & MEME_TERMS or any(term in text for term in
                    ("memecoin", "meme coin", "dogwif", "baby dog", "babyshib")):
                    continue

                liquidity = float((pair.get("liquidity") or {}).get("usd") or 0)
                volume = float((pair.get("volume") or {}).get("h24") or 0)
                price = float(pair.get("priceUsd") or 0)
                created = pair.get("pairCreatedAt")
                if liquidity < min_liquidity or volume < min_volume_24h or price <= 0:
                    continue
                # Require a known, mature pool to reduce brand-new speculative tokens.
                if not created or now_ms - int(created) < min_age_days * 86400000:
                    continue

                existing = found.get(address)
                score = liquidity + volume
                if existing and existing["_score"] >= score:
                    continue
                found[address] = {
                    "name": name,
                    "symbol": symbol,
                    "token_address": address,
                    "approved_non_meme": True,
                    "_score": score
                }
        except Exception as exc:
            print(f"Discovery query failed ({query}): {exc}")

    projects = sorted(found.values(), key=lambda item: item["_score"], reverse=True)
    for project in projects:
        project.pop("_score", None)
    print(f"Automatic discovery found {len(projects)} eligible Solana projects.")
    return projects[:max_projects]
