
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
