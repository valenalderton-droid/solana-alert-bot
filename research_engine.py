"""Bounded autonomous research loop for the Solana alert bot.

Searches public RSS sources using a rotating, performance-aware query plan and
samples DeFiLlama's Solana TVL history. Results are source-linked evidence only:
they never auto-approve tokens, change code, bypass safety filters, or create BUY
signals. Network failures are isolated so research cannot stop market scanning.
"""
import hashlib
import json
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from urllib.parse import quote_plus

import requests

USER_AGENT = "SolanaAlertBot-Research/1.0 (public-source research)"
RSS_URL = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
MAX_ARTICLES_PER_QUERY = 8
MAX_ARTICLE_AGE_DAYS = 7
RESEARCH_INTERVAL_MINUTES = 15
METRICS_INTERVAL_MINUTES = 60

QUERY_PLAN = [
    ("Solana security", "Solana network exploit hack vulnerability outage validator security audit", "security"),
    ("Solana ecosystem", "Solana protocol release upgrade mainnet validator client performance", "protocol"),
    ("Solana ecosystem", "Solana DeFi TVL fees revenue users activity ecosystem data", "adoption"),
    ("Solana ecosystem", "Solana DeFi protocol integration partnership launch adoption", "adoption"),
    ("Solana security", "Solana DeFi exploit audit vulnerability incident", "security"),
    ("Solana ecosystem", "Solana token unlock vesting emissions inflation supply schedule", "tokenomics"),
    ("Solana ecosystem", "Solana DEX liquidity volume market structure liquidity depth", "liquidity"),
    ("Solana regulation", "Solana ETF SEC regulation institutional adoption legal", "regulation"),
    ("Macro / global markets", "Bitcoin crypto Federal Reserve inflation interest rates liquidity markets", "macro"),
    ("Geopolitics / global risk", "geopolitical risk war sanctions oil markets Bitcoin crypto risk-off", "geopolitics"),
    ("Solana ecosystem", "Solana crypto price action false breakout volume volatility trend research", "strategy"),
    ("Solana ecosystem", "Solana crypto trading strategy backtest out-of-sample walk-forward validation", "strategy"),
    ("Solana ecosystem", "Solana protocol governance proposal treasury token utility", "governance"),
    ("Solana ecosystem", "Solana developer activity GitHub releases protocol ecosystem", "developers"),
]

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT, "Accept": "application/rss+xml, application/xml, text/xml"})


def _parse_date(value):
    if not value:
        return None
    try:
        from email.utils import parsedate_to_datetime
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None


def _clean(value):
    import re
    return " ".join(re.sub(r"<[^>]+>", " ", value or "").split())


def _article_id(url, title):
    return hashlib.sha256((str(url) + "|" + str(title)).lower().encode("utf-8")).hexdigest()[:24]


def _performance_topics(outcomes):
    """Use numeric returns when available, otherwise validated win/loss labels."""
    recent = [row for row in (outcomes or [])[-20:] if isinstance(row, dict)]
    numeric = []
    labels = []
    for row in recent:
        if row.get("result") == "data_gap":
            continue
        try:
            value = float(row.get("return_pct"))
            if value == value and abs(value) != float("inf"):
                numeric.append(value)
                continue
        except (TypeError, ValueError, OverflowError):
            pass
        label = str(row.get("result", "")).strip().lower()
        if label in {"loss", "stop", "stop_loss", "stopped", "negative"}:
            labels.append(False)
        elif label in {"win", "target", "take_profit", "positive"}:
            labels.append(True)

    if len(numeric) >= 5:
        loss_ratio = sum(value < 0 for value in numeric) / len(numeric)
    elif len(labels) >= 5:
        loss_ratio = sum(not value for value in labels) / len(labels)
    else:
        return set()

    if loss_ratio >= 0.60:
        return {"security", "liquidity", "tokenomics", "strategy"}
    if loss_ratio <= 0.35:
        return {"adoption", "developers", "protocol", "governance"}
    return set()


def build_research_plan(state, outcomes=None, configured_projects=None):
    """Rotate broad queries, always retain one project query when eligible."""
    priority = _performance_topics(outcomes)
    plan = list(QUERY_PLAN)
    if priority:
        plan.sort(key=lambda row: (row[2] not in priority, row[2]))

    projects = [
        p for p in (configured_projects or [])
        if isinstance(p, dict) and p.get("approved_non_meme")
        and p.get("name") and p.get("symbol")
    ]
    project_query = None
    if projects:
        cursor = int(state.get("project_research_cursor", 0) or 0) % len(projects)
        project = projects[cursor]
        name = str(project["name"]).strip()[:60]
        symbol = str(project["symbol"]).strip()[:16]
        project_query = (
            "Solana ecosystem",
            f'"{name}" "{symbol}" Solana protocol news security audit governance token unlock',
            "project_specific"
        )
        state["project_research_cursor"] = (cursor + 1) % len(projects)

    cursor = int(state.get("research_query_cursor", 0) or 0) % len(plan)
    general = [plan[(cursor + i) % len(plan)] for i in range(min(2 if project_query is None else 1, len(plan)))]
    state["research_query_cursor"] = (cursor + len(general)) % len(plan)
    return ([project_query] if project_query else []) + general


def search_public_rss(query, category, topic, now=None):
    """Search one public news query and return only dated, source-linked records."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=MAX_ARTICLE_AGE_DAYS)
    url = RSS_URL.format(query=quote_plus(query))
    response = SESSION.get(url, timeout=(4, 10))
    response.raise_for_status()
    root = ET.fromstring(response.content)
    found = []
    for item in root.findall(".//item")[:40]:
        title = _clean(item.findtext("title"))
        link = (item.findtext("link") or "").strip()
        published = _parse_date(item.findtext("pubDate"))
        if not title or not link or not published or published < cutoff or published > now + timedelta(minutes=10):
            continue
        source_node = item.find("source")
        source = _clean(source_node.text if source_node is not None else "")
        description = _clean(item.findtext("description"))
        found.append({
            "id": _article_id(link, title),
            "title": title[:350],
            "url": link,
            "published_at": published.isoformat(),
            "source": source or "Google News RSS",
            "description": description[:500],
            "feed_category": category,
            "research_topic": topic,
            "research_query": query[:250],
            "observed_at": now.isoformat(),
        })
    return sorted(found, key=lambda row: row["published_at"], reverse=True)[:MAX_ARTICLES_PER_QUERY]


def fetch_solana_tvl_snapshot(now=None):
    """Read public historical chain TVL and compare the latest point with ~24h ago."""
    now = now or datetime.now(timezone.utc)
    url = "https://api.llama.fi/v2/historicalChainTvl/Solana"
    response = SESSION.get(url, timeout=(4, 10))
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError("DeFiLlama returned insufficient Solana TVL history")
    points = []
    for row in payload:
        try:
            ts, tvl = int(row["date"]), float(row["tvl"])
            if tvl > 0 and tvl < float("inf"):
                points.append((ts, tvl))
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    points.sort()
    if len(points) < 2:
        raise ValueError("No usable Solana TVL observations")
    latest_ts, latest_tvl = points[-1]
    target_ts = latest_ts - 86400
    prior = min(points[:-1], key=lambda point: abs(point[0] - target_ts))
    change = (latest_tvl / prior[1] - 1) * 100 if prior[1] else 0.0
    return {
        "id": f"defillama-solana-tvl-{latest_ts}",
        "kind": "metric",
        "metric": "Solana DeFi TVL",
        "value_usd": round(latest_tvl, 2),
        "change_24h_pct": round(change, 3),
        "observed_at": now.isoformat(),
        "source": "DeFiLlama",
        "url": url,
        "note": "TVL is ecosystem context, not a direct SOL price prediction."
    }


def fetch_solana_activity_metrics(now=None):
    """Fetch protocol-level DEX, fees, and derivatives snapshots from DeFiLlama."""
    now = now or datetime.now(timezone.utc)
    endpoints = [
        ("Solana DEX volume", "https://api.llama.fi/overview/dexs/Solana"),
        ("Solana fees", "https://api.llama.fi/overview/fees/Solana"),
        ("Solana perpetuals volume", "https://api.llama.fi/overview/derivatives/Solana"),
    ]
    snapshots = []
    failures = []
    for label, url in endpoints:
        try:
            response = SESSION.get(url, timeout=(4, 10))
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("unexpected response shape")
            total24h = payload.get("total24h")
            if total24h is None:
                total24h = payload.get("total24hUSD")
            try:
                value = float(total24h) if total24h is not None else None
            except (TypeError, ValueError, OverflowError):
                value = None
            change = None
            for key in ("change_1d", "change1d", "change_24h", "change24h"):
                if payload.get(key) is not None:
                    try:
                        change = float(payload[key])
                        break
                    except (TypeError, ValueError, OverflowError):
                        pass
            if change is not None and not (-100 <= change <= 10000):
                change = None
            if value is not None and (value < 0 or value == float("inf") or value != value):
                value = None
            snapshots.append({
                "id": f"defillama-{label.lower().replace(' ', '-')}-{int(now.timestamp() // 3600)}",
                "kind": "metric",
                "metric": label,
                "value_usd": round(value, 2) if value is not None else None,
                "change_24h_pct": round(change, 3) if change is not None else None,
                "observed_at": now.isoformat(),
                "source": "DeFiLlama",
                "url": url,
                "data_quality": "valid" if value is not None or change is not None else "partial",
                "note": "Ecosystem activity context; not a standalone price prediction."
            })
        except (requests.RequestException, ValueError, TypeError) as exc:
            failures.append({"metric": label, "error": str(exc)[:160]})
            print(f"DeFiLlama metric failed ({label}): {exc}")
    return snapshots, failures


def run_research_cycle(state, outcomes=None, configured_projects=None, now=None):
    """Run a rate-limited research cycle; save evidence and report health counters."""
    now = now or datetime.now(timezone.utc)
    status = state.setdefault("research_status", {})
    last = status.get("last_run_utc")
    if last:
        try:
            parsed = datetime.fromisoformat(str(last).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            age = (now - parsed.astimezone(timezone.utc)).total_seconds()
            if 0 <= age < RESEARCH_INTERVAL_MINUTES * 60:
                return {"skipped": True, "reason": "research cooldown active"}
        except (ValueError, TypeError, OverflowError):
            pass

    evidence = state.setdefault("research_observations", [])
    seen = set(state.setdefault("seen_research_ids", []))
    plan = build_research_plan(state, outcomes, configured_projects)
    new_articles = []
    query_results = []
    failures = 0
    for category, query, topic in plan:
        try:
            items = search_public_rss(query, category, topic, now=now)
            query_results.append({"topic": topic, "results": len(items), "ok": True})
            for item in items:
                if item["id"] in seen:
                    continue
                seen.add(item["id"])
                evidence.append(item)
                new_articles.append(item)
        except (requests.RequestException, ET.ParseError, ValueError, TypeError) as exc:
            failures += 1
            query_results.append({"topic": topic, "ok": False, "error": str(exc)[:180]})
            print(f"Research search failed ({topic}): {exc}")

    metric_due = True
    last_metric = status.get("last_tvl_refresh_utc")
    if last_metric:
        try:
            parsed = datetime.fromisoformat(str(last_metric).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            metric_due = (now - parsed.astimezone(timezone.utc)).total_seconds() >= METRICS_INTERVAL_MINUTES * 60
        except (ValueError, TypeError, OverflowError):
            metric_due = True
    metric = None
    activity_metrics = []
    if metric_due:
        try:
            metric = fetch_solana_tvl_snapshot(now=now)
            activity_metrics, metric_failures = fetch_solana_activity_metrics(now=now)
            failures += len(metric_failures)
            for row in [metric] + activity_metrics:
                if row["id"] not in seen:
                    evidence.append(row)
                    seen.add(row["id"])
            status["last_tvl_refresh_utc"] = now.isoformat()
            status["last_activity_metrics_refresh_utc"] = now.isoformat()
        except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
            failures += 1
            print(f"Solana ecosystem metrics failed: {exc}")

    state["research_observations"] = evidence[-500:]
    state["seen_research_ids"] = list(seen)[-1000:]
    status.update({
        "last_run_utc": now.isoformat(),
        "queries_attempted": len(plan),
        "new_articles": len(new_articles),
        "failures": failures,
        "last_topics": [row[2] for row in plan],
        "query_results": query_results[-4:],
        "total_evidence": len(evidence),
        "latest_tvl_change_24h_pct": metric.get("change_24h_pct") if metric else status.get("latest_tvl_change_24h_pct"),
        "activity_metrics_refreshed": bool(activity_metrics),
        "activity_metrics_count": len(activity_metrics),
    })
    print("Autonomous research cycle:", json.dumps(status, ensure_ascii=False))
    return {"skipped": False, "new_articles": len(new_articles), "failures": failures, "metric": metric, "activity_metrics": len(activity_metrics)}
