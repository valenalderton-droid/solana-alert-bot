
import requests
import json
import os
import math
import statistics
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
    """Choose a liquid, recently active pool with a price consistent with peers.

    DexScreener does not expose a quote timestamp on each pair. Recent hourly
    trades and cross-pool price agreement are therefore used as conservative
    sanity checks; this does not guarantee an executable fill price.
    """
    url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
    data = get_json(url)
    pairs = data.get("pairs") or []

    eligible = []
    for pair in pairs:
        if pair.get("chainId") != "solana":
            continue

        base = pair.get("baseToken") or {}
        if str(base.get("address") or "").lower() != token_address.lower():
            continue

        try:
            liquidity = float((pair.get("liquidity") or {}).get("usd") or 0)
            price = float(pair.get("priceUsd") or 0)
        except (TypeError, ValueError, OverflowError):
            continue
        if not (math.isfinite(liquidity) and math.isfinite(price)):
            continue
        if liquidity < float(min_liquidity) or price <= 0:
            continue

        txns = pair.get("txns") or {}
        h1 = txns.get("h1") or {}
        try:
            recent_trades = int(h1.get("buys") or 0) + int(h1.get("sells") or 0)
        except (TypeError, ValueError, OverflowError):
            recent_trades = 0
        pair["_liquidity_usd"] = liquidity
        pair["_reference_price_usd"] = price
        pair["_recent_trades_1h"] = recent_trades
        eligible.append(pair)

    if not eligible:
        return None

    # When transaction counts are provided, don't treat a dormant pool's old
    # displayed quote as current if no eligible pool traded in the last hour.
    has_h1_data = any(isinstance((p.get("txns") or {}).get("h1"), dict) for p in eligible)
    if has_h1_data:
        active = [p for p in eligible if p["_recent_trades_1h"] > 0]
        if not active:
            return None
        eligible = active

    # A high-liquidity pool can still carry a stale/outlier quote. Compare only
    # eligible, active pools and fail closed if no pool is within 3% of their
    # median price; this also keeps the selected pool's candles aligned to price.
    median_price = statistics.median(p["_reference_price_usd"] for p in eligible)
    consistent = [
        p for p in eligible
        if abs(p["_reference_price_usd"] / median_price - 1.0) <= 0.03
    ]
    if not consistent:
        return None

    return max(consistent, key=lambda p: p["_liquidity_usd"])


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
# Curated symbols are a seed, not the entire universe. Dynamic category searches
# below expand discovery beyond this list; all discovered assets still face liquidity,
# age, metadata, and explicit meme-risk checks before they can be analyzed.
ECOSYSTEM_SYMBOLS = {
    "JUP", "RAY", "ORCA", "KMNO", "JTO", "PYTH", "DRIFT", "MNDE",
    "HNT", "MOBILE", "RENDER", "SHDW", "NOS", "TNSR", "PRCL", "ZEX",
    "CLOUD", "INF", "JLP", "MPLX", "W", "STEP", "SLND", "PORT", "UXD",
    "C98", "FIDA", "IO", "ZEUS", "MET", "BLZE", "MSOL", "BSOL",
    "JITOSOL", "BNSOL", "AUDIO", "ACS", "ATLAS", "POLIS", "GENE", "GMT",
    "GST", "PRISM", "WOO", "CHEX", "MEDIA", "DVPN", "NATIX", "MATRIX",
    "ZBCN", "MELD", "HXD", "SHARK", "TULIP", "SBR", "AURY", "DIO",
    "MPL", "SLND", "HONEY", "PORT", "UXP", "STEP", "JST", "LDO",
    "CROWN", "SHFL", "BORG", "GARI", "KIN"
}

MEME_TERMS = {
    "meme", "memecoin", "inu", "doge", "pepe", "bonk", "wif", "shib",
    "floki", "wojak", "frog", "elon", "trump", "moon", "rocket", "kitty",
    "dogwif", "baby dog", "babyshib", "pumpfun", "pump.fun", "cat coin",
    "dog coin", "frog coin", "memes"
}

# A small seed of high-signal symbols; broad category searches discover the rest.
DISCOVERY_SYMBOL_QUERIES = (
    "JUP", "RAY", "ORCA", "KMNO", "JTO", "PYTH", "DRIFT", "MNDE",
    "HNT", "RENDER", "SHDW", "TNSR", "PRCL", "CLOUD", "JLP", "MPLX",
    "MSOL", "BSOL", "ZBCN", "MOBILE", "W", "FIDA"
)

DISCOVERY_QUERIES = (
    "Solana DeFi protocol", "Solana lending borrowing", "Solana DEX exchange",
    "Solana oracle data", "Solana liquid staking", "Solana restaking",
    "Solana infrastructure developer tools", "Solana decentralized storage",
    "Solana payments stablecoin", "Solana perpetual futures derivatives",
    "Solana yield vaults", "Solana real world assets RWA",
    "Solana privacy security", "Solana wallet infrastructure",
    "Solana governance protocol", "Solana cross chain bridge",
    "Solana compute AI infrastructure", "Solana DePIN network",
    "Solana gaming infrastructure", "Solana tokenized treasury",
)


def _is_meme_like(name, symbol):
    text = f"{name} {symbol}".lower().replace("-", " ").replace("_", " ")
    words = set(text.split())
    return bool(words & MEME_TERMS) or any(
        term in text for term in (
            "memecoin", "meme coin", "dogwif", "baby dog", "babyshib",
            "pump.fun", "pumpfun", "dog token", "cat token"
        )
    )


def _has_project_metadata(pair):
    info = pair.get("info") or {}
    websites = info.get("websites") or []
    socials = info.get("socials") or []
    has_website = any(
        isinstance(item, dict) and str(item.get("url", "")).startswith(("https://", "http://"))
        for item in websites
    )
    has_social = any(
        isinstance(item, dict) and str(item.get("url", "")).startswith(("https://", "http://"))
        for item in socials
    )
    return has_website, has_social


def is_allowed_non_meme_project(project):
    """Allow curated protocol assets or evidence-backed category discoveries.

    This is a risk-reduction heuristic, not proof that a token is legitimate.
    """
    if not isinstance(project, dict) or not project.get("approved_non_meme", False):
        return False
    symbol = str(project.get("symbol") or "").strip().upper()
    name = str(project.get("name") or "").strip()
    address = str(project.get("token_address") or "").strip()
    if not symbol or not address or _is_meme_like(name, symbol):
        return False
    curated = symbol in ECOSYSTEM_SYMBOLS
    discovered = (
        project.get("discovery_source") == "dexscreener_category_search"
        and project.get("discovery_confidence", 0) >= 0.8
        and project.get("has_project_website", False)
    )
    return curated or discovered


DISCOVERY_CACHE_FILE = Path(__file__).resolve().parent / "data" / "discovery_cache.json"


def discover_solana_projects(min_liquidity=100000, min_volume_24h=50000,
                             min_age_days=30, max_projects=500):
    """Build a broad, cached Solana utility-token universe from many category searches.

    The API cannot enumerate literally every Solana project. This gathers a much
    broader candidate set, rotates through it in main.py, and fails closed when
    metadata or market-quality evidence is missing.
    """
    found = {}
    filters = {
        "discovery_version": 2,
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
            cached_at = datetime.fromisoformat(str(cache.get("saved_at", "")).replace("Z", "+00:00"))
            if cached_at.tzinfo is None:
                cached_at = cached_at.replace(tzinfo=timezone.utc)
            age_seconds = (datetime.now(timezone.utc) - cached_at.astimezone(timezone.utc)).total_seconds()
            if cached_projects and cache_ttl_minutes > 0 and 0 <= age_seconds < cache_ttl_minutes * 60:
                print(f"Discovery cache hit: {len(cached_projects)} projects; age={age_seconds / 60:.1f} min.")
                return cached_projects[:max_projects]
    except (OSError, ValueError, TypeError, AttributeError):
        cached_projects = []

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    counts = {
        "queries": 0, "pairs_seen": 0, "wrong_chain": 0,
        "meme_rejected": 0, "metadata_rejected": 0,
        "liquidity_volume_age_rejected": 0, "eligible": 0,
    }

    # Search known protocol symbols plus broad categories. A bounded request count
    # reduces API hammering while finding projects outside the old fixed allowlist.
    queries = [("symbol", symbol) for symbol in DISCOVERY_SYMBOL_QUERIES]
    queries.extend(("category", query) for query in DISCOVERY_QUERIES)
    for query_type, query in queries:
        counts["queries"] += 1
        try:
            payload = get_json(
                "https://api.dexscreener.com/latest/dex/search",
                params={"q": query}
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
                if not address or not name or not symbol:
                    continue
                if query_type == "symbol" and symbol != query:
                    continue
                if _is_meme_like(name, symbol):
                    counts["meme_rejected"] += 1
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

                has_website, has_social = _has_project_metadata(pair)
                curated = symbol in ECOSYSTEM_SYMBOLS
                # New symbols are admitted only when category-discovered and an
                # official website is present. Social-only tokens are too easy to fake.
                if not curated and (query_type != "category" or not has_website):
                    counts["metadata_rejected"] += 1
                    continue

                score = liquidity + volume
                project = {
                    "name": name,
                    "symbol": symbol,
                    "token_address": address,
                    "approved_non_meme": True,
                    "discovery_source": (
                        "curated_symbol_search" if curated
                        else "dexscreener_category_search"
                    ),
                    "discovery_category": query if query_type == "category" else "known_ecosystem_symbol",
                    "discovery_confidence": 1.0 if curated else 0.8,
                    "has_project_website": bool(has_website),
                    "has_project_social": bool(has_social),
                    "liquidity_usd_at_discovery": liquidity,
                    "volume_24h_usd_at_discovery": volume,
                }
                previous = found.get(address)
                if previous and previous["_score"] >= score:
                    continue
                project["_score"] = score
                found[address] = project
        except Exception as exc:
            print(f"Discovery query failed ({query}): {exc}")

    projects = sorted(found.values(), key=lambda item: item["_score"], reverse=True)
    for project in projects:
        project.pop("_score", None)
    counts["eligible"] = len(projects)
    print("Discovery diagnostics: " + str(counts))
    print(f"Automatic discovery found {len(projects)} eligible Solana utility/protocol candidates.")
    if projects:
        try:
            DISCOVERY_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            DISCOVERY_CACHE_FILE.write_text(json.dumps({
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "filters": filters,
                "projects": projects[:max_projects],
                "discovery_diagnostics": counts,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            print(f"Discovery cache write failed: {exc}")
    elif cached_projects:
        print(f"Discovery returned no projects; using {len(cached_projects)} cached candidates as fallback.")
        return cached_projects[:max_projects]
    return projects[:max_projects]
