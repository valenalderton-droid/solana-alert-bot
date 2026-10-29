
import requests
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "SolanaProjectTrader/1.1"})
RETRY_POLICY = Retry(
    total=2,
    connect=2,
    read=1,
    status=2,
    backoff_factor=0.6,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET"]),
    respect_retry_after_header=True
)
SESSION.mount("https://", HTTPAdapter(max_retries=RETRY_POLICY, pool_connections=10, pool_maxsize=10))


def get_json(url, params=None):
    response = SESSION.get(url, params=params, timeout=(5, 12))
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected API response format from {url}")
    return payload


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
        try:
            candle = {
                "timestamp": int(item[0]),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5])
            }
        except (TypeError, ValueError, OverflowError):
            continue

        # Reject malformed or non-positive OHLC candles before indicators use them.
        prices = (candle["open"], candle["high"], candle["low"], candle["close"])
        if any(not (value > 0 and value < float("inf")) for value in prices):
            continue
        if candle["high"] < max(candle["open"], candle["close"], candle["low"]):
            continue
        if candle["low"] > min(candle["open"], candle["close"], candle["high"]):
            continue
        if not (candle["volume"] >= 0 and candle["volume"] < float("inf")):
            continue
        candles.append(candle)

    candles.sort(key=lambda c: c["timestamp"])

    # Remove duplicate timestamps.
    unique = {c["timestamp"]: c for c in candles}
    return [unique[t] for t in sorted(unique)]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


# Discover established Solana ecosystem tokens from protocol/category searches.
# This is a heuristic filter, not a guarantee that every meme token is excluded.
# Curated protocol-token universe. Exact-symbol matching reduces unrelated search results.
# This is deliberately conservative; it will miss some legitimate projects.
ECOSYSTEM_SYMBOLS = {
    "JUP", "RAY", "ORCA", "KMNO", "JTO", "PYTH", "DRIFT", "MNDE",
    "HNT", "MOBILE", "RENDER", "SHDW", "NOS", "TNSR", "PRCL", "ZEX",
    "CLOUD", "INF", "JLP", "MPLX", "W", "STEP", "SLND",
    "PORT", "UXD", "C98", "MNGO", "SRM", "FIDA",
    "HONEY", "IO", "ZEUS", "MET", "BLZE",
    "MSOL", "BSOL", "JITOSOL",
    "BNSOL"
}

MEME_TERMS = {
    "meme", "memecoin", "inu", "doge", "pepe", "bonk", "wif", "shib",
    "floki", "wojak", "frog", "elon", "trump", "moon", "rocket", "kitty",
    "dogwif", "baby dog", "babyshib", "pumpfun", "pump.fun"
}


def is_allowed_non_meme_project(project):
    """Fail closed: only known Solana ecosystem utility/protocol tokens pass."""
    if not isinstance(project, dict) or not project.get("approved_non_meme", False):
        return False
    symbol = str(project.get("symbol") or "").strip().upper()
    name = str(project.get("name") or "").strip().lower()
    address = str(project.get("token_address") or "").strip()
    if not symbol or symbol not in ECOSYSTEM_SYMBOLS or not address:
        return False
    text = f"{name} {symbol.lower()}"
    words = set(text.replace("-", " ").replace("_", " ").split())
    if words & MEME_TERMS:
        return False
    if any(term in text for term in ("memecoin", "meme coin", "dogwif", "baby dog", "babyshib", "pump.fun", "pumpfun")):
        return False
    return True


DISCOVERY_CACHE_FILE = Path(__file__).resolve().parent / "data" / "discovery_cache.json"


def discover_solana_projects(min_liquidity=100000, min_volume_24h=50000,
                             min_age_days=30, max_projects=20):
    """Discover candidates with a disk cache to avoid repeated API bursts."""
    found = {}
    filters = {
        "min_liquidity": float(min_liquidity),
        "min_volume_24h": float(min_volume_24h),
        "min_age_days": int(min_age_days),
        "max_projects": int(max_projects),
    }
    cache_ttl_minutes = max(0, int(os.getenv("DISCOVERY_CACHE_TTL_MINUTES", "45")))
    cached_projects = []
    try:
        cache = json.loads(DISCOVERY_CACHE_FILE.read_text(encoding="utf-8"))
        if cache.get("filters") == filters and isinstance(cache.get("projects"), list):
            cached_projects = cache["projects"]
            cached_at = datetime.fromisoformat(
                str(cache.get("saved_at", "")).replace("Z", "+00:00")
            )
            if cached_at.tzinfo is None:
                cached_at = cached_at.replace(tzinfo=timezone.utc)
            age_seconds = (datetime.now(timezone.utc) - cached_at.astimezone(timezone.utc)).total_seconds()
            if cached_projects and cache_ttl_minutes > 0 and 0 <= age_seconds < cache_ttl_minutes * 60:
                print(
                    f"Discovery cache hit: {len(cached_projects)} projects; "
                    f"age={age_seconds / 60:.1f} min."
                )
                return cached_projects[:max_projects]
    except (OSError, ValueError, TypeError, AttributeError):
        cached_projects = []
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    counts = {"queries": 0, "pairs_seen": 0, "wrong_chain": 0,
              "symbol_rejected": 0, "liquidity_volume_age_rejected": 0}

    # Search one ticker at a time, then require exact symbol match.
    for symbol_query in sorted(ECOSYSTEM_SYMBOLS):
        counts["queries"] += 1
        try:
            payload = get_json(
                "https://api.dexscreener.com/latest/dex/search",
                params={"q": symbol_query}
            )
            for pair in payload.get("pairs") or []:
                counts["pairs_seen"] += 1
                if pair.get("chainId") != "solana":
                    counts["wrong_chain"] += 1
                    continue

                token = pair.get("baseToken") or {}
                address = str(token.get("address") or "").strip()
                name = str(token.get("name") or "").strip()
                symbol = str(token.get("symbol") or "").strip().upper()
                if not address or not name or symbol != symbol_query:
                    counts["symbol_rejected"] += 1
                    continue

                text = (name + " " + symbol).lower()
                words = set(text.replace("-", " ").replace("_", " ").split())
                if words & MEME_TERMS or any(term in text for term in
                    ("memecoin", "meme coin", "dogwif", "baby dog", "babyshib")):
                    counts["symbol_rejected"] += 1
                    continue

                liquidity = float((pair.get("liquidity") or {}).get("usd") or 0)
                volume = float((pair.get("volume") or {}).get("h24") or 0)
                price = float(pair.get("priceUsd") or 0)
                created = pair.get("pairCreatedAt")
                if (liquidity < min_liquidity or volume < min_volume_24h or
                        price <= 0 or not created or
                        now_ms - int(created) < min_age_days * 86400000):
                    counts["liquidity_volume_age_rejected"] += 1
                    continue

                previous = found.get(address)
                score = liquidity + volume
                if previous and previous["_score"] >= score:
                    continue
                found[address] = {
                    "name": name,
                    "symbol": symbol,
                    "token_address": address,
                    "approved_non_meme": True,
                    "_score": score
                }
        except Exception as exc:
            print(f"Discovery query failed ({symbol_query}): {exc}")

    projects = sorted(found.values(), key=lambda item: item["_score"], reverse=True)
    for project in projects:
        project.pop("_score", None)
    print("Discovery diagnostics: " + str(counts))
    print(f"Automatic discovery found {len(projects)} eligible Solana projects.")
    if projects:
        try:
            DISCOVERY_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            DISCOVERY_CACHE_FILE.write_text(
                json.dumps({
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "filters": filters,
                    "projects": projects[:max_projects],
                }, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except OSError as exc:
            print(f"Discovery cache write failed: {exc}")
    elif cached_projects:
        # API outages/rate limits should not erase a previously known universe.
        print(f"Discovery returned no projects; using {len(cached_projects)} cached candidates as fallback.")
        return cached_projects[:max_projects]
    return projects[:max_projects]
