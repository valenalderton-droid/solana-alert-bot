import requests
from datetime import datetime, timezone

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "SolanaProjectTrader/1.0"})


def get_json(url, params=None):
    response = SESSION.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def discover_solana_candidates(limit=40):
    """Discover candidate mints from DexScreener public discovery feeds."""
    feeds = [
        "https://api.dexscreener.com/token-profiles/latest/v1",
        "https://api.dexscreener.com/token-boosts/top/v1",
    ]
    found = {}
    for url in feeds:
        try:
            data = get_json(url)
            if not isinstance(data, list):
                continue
            for item in data:
                if item.get("chainId") != "solana":
                    continue
                address = str(item.get("tokenAddress") or "").strip()
                if address:
                    found[address] = {
                        "token_address": address,
                        "name": "",
                        "symbol": "",
                        "category": "discovered",
                        "approved_non_meme": True,
                        "discovered_from": url.rsplit("/", 1)[-1],
                    }
        except requests.RequestException as exc:
            print(f"Discovery feed unavailable ({url}): {exc}")
    return list(found.values())[:max(1, int(limit))]


def looks_like_meme_or_low_quality(pair):
    """Heuristic only; it cannot prove a token is non-meme or safe."""
    base = pair.get("baseToken") or {}
    name = (str(base.get("name") or "") + " " + str(base.get("symbol") or "")).lower()
    suspicious = (
        "meme", "memecoin", "inu", "doge", "pepe", "shib", "wojak",
        "bonk", "frog", "moon", "pump", "jeet", "elon", "trump",
        "wif", "chad", "degen", "ape", "dog", "kitty", "baby "
    )
    if any(word in name for word in suspicious):
        return True

    info = pair.get("info") or {}
    websites = info.get("websites") or []
    socials = info.get("socials") or []
    has_website = any(x.get("url") for x in websites if isinstance(x, dict))
    has_social = any(x.get("url") for x in socials if isinstance(x, dict))
    if not has_website or not has_social:
        return True

    created = pair.get("pairCreatedAt")
    if created:
        age_days = (datetime.now(timezone.utc).timestamp() * 1000 - float(created)) / 86400000
        if age_days < 30:
            return True
    return False


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
        volume24h = float((pair.get("volume") or {}).get("h24") or 0)
        price = float(pair.get("priceUsd") or 0)
        if liquidity < min_liquidity or volume24h < 250000 or price <= 0:
            continue
        if looks_like_meme_or_low_quality(pair):
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
    params = {"aggregate": 1, "limit": min(int(limit), 1000), "currency": "usd", "token": "base"}
    data = get_json(url, params=params)
    items = data.get("data", {}).get("attributes", {}).get("ohlcv_list", [])
    candles = []
    for item in items:
        if len(item) < 6:
            continue
        candles.append({
            "timestamp": int(item[0]), "open": float(item[1]), "high": float(item[2]),
            "low": float(item[3]), "close": float(item[4]), "volume": float(item[5])
        })
    candles.sort(key=lambda c: c["timestamp"])
    unique = {c["timestamp"]: c for c in candles}
    return [unique[t] for t in sorted(unique)]


def utc_now():
    return datetime.now(timezone.utc).isoformat()
