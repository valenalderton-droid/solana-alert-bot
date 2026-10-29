
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
        h24 = txns.get("h24") or {}
        try:
            recent_trades_1h = int(h1.get("buys") or 0) + int(h1.get("sells") or 0)
            recent_trades_24h = int(h24.get("buys") or 0) + int(h24.get("sells") or 0)
        except (TypeError, ValueError, OverflowError):
            recent_trades_1h = 0
            recent_trades_24h = 0
        pair["_liquidity_usd"] = liquidity
        pair["_reference_price_usd"] = price
        pair["_recent_trades_1h"] = recent_trades_1h
        pair["_recent_trades_24h"] = recent_trades_24h
        eligible.append(pair)

    if not eligible:
        return None

    # Prefer pools with trades in the last hour. If none traded during that
    # hour, accept pools with 24h activity instead of discarding every quote.
    # DexScreener's priceUsd is a current API snapshot, not an executable quote.
    active_1h = [p for p in eligible if p["_recent_trades_1h"] > 0]
    if active_1h:
        eligible = active_1h
    else:
        active_24h = [p for p in eligible if p["_recent_trades_24h"] > 0]
        if not active_24h:
            return None
        eligible = active_24h

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
    """Universe gate: accept Solana project candidates except meme coins.

    Discovery metadata, a curated symbol list, and an official website are not
    prerequisites. Market-data integrity, liquidity, price-action, and signal
    quality checks remain separate downstream safeguards.
    """
    if not isinstance(project, dict):
        return False
    symbol = str(project.get("symbol") or "").strip().upper()
    name = str(project.get("name") or "").strip()
    address = str(project.get("token_address") or "").strip()
    if not symbol or not address:
        return False
    if _is_meme_like(name, symbol):
        return False
    # If an explicit chain is supplied, never let a non-Solana asset through.
    chain = str(project.get("chain") or project.get("chainId") or "").strip().lower()
    if chain and chain not in ("solana", "sol"):
        return False
    return True


DISCOVERY_CACHE_FILE = Path(__file__).resolve().parent / "data" / "discovery_cache.json"


def discover_solana_projects(min_liquidity=100000, min_volume_24h=50000,
                             min_age_days=30, max_projects=500):
    """Discover a broad Solana token universe using parallel query and trending-source search.

    DexScreener search is relevance-based, not an exhaustive token registry.
    Candidate discovery is intentionally broad; explicit meme filtering and
    downstream price/liquidity/signal validation remain separate safeguards.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    found = {}
    filters = {
        "discovery_version": 4,
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

    counts = {
        "queries": 0, "pairs_seen": 0, "wrong_chain": 0, "meme_rejected": 0,
        "invalid_candidates": 0, "eligible": 0, "search_errors": 0,
        "profile_seed_addresses": 0, "boost_seed_addresses": 0,
        "unique_mints": 0, "query_sources": 0
    }
    queries = [("symbol", symbol) for symbol in DISCOVERY_SYMBOL_QUERIES]
    queries.extend(("category", query) for query in DISCOVERY_QUERIES)
    # Synonyms and sub-sector terms improve recall beyond a small set of broad phrases.
    queries.extend(("category", query) for query in (
        "Solana DeFi TVL protocol token",
        "Solana lending protocol token",
        "Solana borrow money market",
        "Solana DEX AMM liquidity",
        "Solana perpetuals trading protocol",
        "Solana options derivatives protocol",
        "Solana liquid staking token LST",
        "Solana staking validator infrastructure",
        "Solana oracle data feeds",
        "Solana bridge interoperability",
        "Solana stablecoin payments",
        "Solana real world asset tokenization",
        "Solana treasury yield token",
        "Solana DePIN wireless mapping compute",
        "Solana AI agent infrastructure token",
        "Solana developer tooling RPC indexing",
        "Solana storage bandwidth infrastructure",
        "Solana privacy cryptography security",
        "Solana gaming metaverse utility token",
        "Solana NFT infrastructure marketplace",
        "Solana governance DAO protocol",
        "Solana restaking security",
        "Solana MEV validator staking",
        "Solana tokenized equities funds",
        "Solana insurance risk management protocol",
        "Solana cross-chain messaging",
        "Solana launchpad infrastructure",
        "Solana portfolio analytics protocol",
        "Solana payments card fintech",
        "Solana data availability compute"
    ))
    counts["queries"] = len(queries)

    def search_query(task):
        query_type, query = task
        try:
            payload = get_json("https://api.dexscreener.com/latest/dex/search", params={"q": query})
            return query_type, query, payload.get("pairs") or [], None
        except Exception as exc:
            return query_type, query, [], str(exc)

    # Execute independent searches concurrently while keeping a bounded request rate.
    max_workers = max(2, min(8, int(os.getenv("DISCOVERY_SEARCH_WORKERS", "6"))))
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(search_query, task) for task in queries]
        for future in as_completed(futures):
            results.append(future.result())

    # Newly promoted/updated token profiles and boosts act only as extra discovery
    # seeds; promotion is not treated as quality, legitimacy, or a BUY signal.
    seed_addresses = {"profile": set(), "boost": set()}
    seed_urls = (
        ("profile", "https://api.dexscreener.com/token-profiles/latest/v1"),
        ("boost", "https://api.dexscreener.com/token-boosts/latest/v1"),
    )
    def fetch_seed(kind, url):
        try:
            response = SESSION.get(url, timeout=(4, 10))
            response.raise_for_status()
            payload = response.json()
            rows = payload if isinstance(payload, list) else payload.get("data", []) if isinstance(payload, dict) else []
            addresses = []
            for row in rows:
                if not isinstance(row, dict) or str(row.get("chainId", "")).lower() != "solana":
                    continue
                address = str(row.get("tokenAddress") or "").strip()
                if address:
                    addresses.append(address)
            return kind, addresses, None
        except Exception as exc:
            return kind, [], str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        seed_results = list(pool.map(lambda args: fetch_seed(*args), seed_urls))
    for kind, addresses, error in seed_results:
        seed_addresses[kind].update(addresses)
        counts[f"{kind}_seed_addresses"] = len(set(addresses))
        if error:
            counts["search_errors"] += 1
            print(f"Discovery seed feed failed ({kind}): {error}")

    # Look up seed mints in small bounded batches to avoid unbounded API traffic.
    seed_list = list(dict.fromkeys(
        list(seed_addresses["profile"])[:25] + list(seed_addresses["boost"])[:25]
    ))
    def lookup_mints(batch):
        try:
            payload = get_json(
                "https://api.dexscreener.com/latest/dex/tokens/" + ",".join(batch)
            )
            return payload.get("pairs") or [], None
        except Exception as exc:
            return [], str(exc)
    seed_results_pairs = []
    batches = [seed_list[i:i + 25] for i in range(0, len(seed_list), 25)]
    if batches:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for pairs, error in pool.map(lookup_mints, batches):
                seed_results_pairs.extend(pairs)
                if error:
                    counts["search_errors"] += 1
                    print(f"Discovery seed token lookup failed: {error}")
        results.append(("seed", "latest_profiles_and_boosts", seed_results_pairs, None))

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    for query_type, query, pairs, error in results:
        counts["query_sources"] += 1
        if error:
            counts["search_errors"] += 1
            print(f"Discovery query failed ({query}): {error}")
            continue
        for pair in pairs:
            counts["pairs_seen"] += 1
            if not isinstance(pair, dict) or pair.get("chainId") != "solana":
                counts["wrong_chain"] += 1
                continue
            token = pair.get("baseToken") or {}
            address = str(token.get("address") or "").strip()
            name = str(token.get("name") or "").strip()
            symbol = str(token.get("symbol") or "").strip().upper()
            if not address or not name or not symbol:
                counts["invalid_candidates"] += 1
                continue
            if query_type == "symbol" and symbol != query:
                continue
            if _is_meme_like(name, symbol):
                counts["meme_rejected"] += 1
                continue
            try:
                liquidity = float((pair.get("liquidity") or {}).get("usd") or 0)
                volume = float((pair.get("volume") or {}).get("h24") or 0)
                price = float(pair.get("priceUsd") or 0)
            except (TypeError, ValueError, OverflowError):
                counts["invalid_candidates"] += 1
                continue
            if not all(math.isfinite(x) for x in (liquidity, volume, price)) or price <= 0 or liquidity < 0 or volume < 0:
                counts["invalid_candidates"] += 1
                continue

            has_website, has_social = _has_project_metadata(pair)
            created = pair.get("pairCreatedAt")
            try:
                pair_age_days = max(0.0, (now_ms - int(created)) / 86400000) if created else None
            except (TypeError, ValueError, OverflowError):
                pair_age_days = None
            # Logarithmic score avoids a single huge pool dominating discovery order.
            score = (
                0.45 * math.log1p(liquidity)
                + 0.45 * math.log1p(volume)
                + (0.7 if has_website else 0.0)
                + (0.25 if has_social else 0.0)
                + (0.15 if pair_age_days is not None and pair_age_days >= 30 else 0.0)
                + (0.1 if query_type == "seed" else 0.0)
            )
            project = {
                "name": name, "symbol": symbol, "token_address": address,
                "chain": "solana", "approved_non_meme": True,
                "discovery_source": (
                    "dexscreener_profile_or_boost" if query_type == "seed"
                    else "curated_symbol_search" if query_type == "symbol"
                    else "dexscreener_search"
                ),
                "discovery_category": query if query_type != "symbol" else "known_ecosystem_symbol",
                "discovery_confidence": 1.0 if query_type == "symbol" else 0.8,
                "has_project_website": bool(has_website),
                "has_project_social": bool(has_social),
                "liquidity_usd_at_discovery": liquidity,
                "volume_24h_usd_at_discovery": volume,
                "pair_age_days_at_discovery": round(pair_age_days, 2) if pair_age_days is not None else None,
            }
            previous = found.get(address)
            if previous and previous["_score"] >= score:
                continue
            project["_score"] = score
            found[address] = project

    projects = sorted(found.values(), key=lambda item: item["_score"], reverse=True)
    for project in projects:
        project.pop("_score", None)
    counts["eligible"] = len(projects)
    counts["unique_mints"] = len(projects)
    print("Discovery diagnostics: " + json.dumps(counts, ensure_ascii=False))
    print(f"Expanded discovery found {len(projects)} unique Solana non-meme candidates.")
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
