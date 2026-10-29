"""Lightweight, source-linked Solana news impact monitor.

This module classifies headlines with transparent keyword rules; it does not
claim to understand full articles or predict price with certainty. Each alert
includes the headline, source/date, a plain-language interpretation, likely
SOL relevance, expected horizon, confidence and the original link.
"""
import hashlib
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta

import requests

FEEDS = [
    ("Solana ecosystem", "https://news.google.com/rss/search?q=Solana+blockchain+ecosystem&hl=en-US&gl=US&ceid=US:en"),
    ("SOL market", "https://news.google.com/rss/search?q=Solana+SOL+crypto+market&hl=en-US&gl=US&ceid=US:en"),
    ("Solana security", "https://news.google.com/rss/search?q=Solana+hack+exploit+outage+security&hl=en-US&gl=US&ceid=US:en"),
    ("Solana regulation", "https://news.google.com/rss/search?q=Solana+ETF+SEC+regulation+institutional&hl=en-US&gl=US&ceid=US:en"),
]
MAX_AGE_HOURS = 36
MAX_ITEMS_PER_RUN = 3
KEYWORDS = {
    "strong_negative": [
        "outage", "network down", "halted", "exploit", "hacked", "hack drains",
        "bridge exploit", "critical vulnerability", "consensus failure",
        "sec lawsuit", "lawsuit against solana", "insolvency"
    ],
    "negative": [
        "breach", "attack", "vulnerability", "downtime", "congestion",
        "failed launch", "delay", "lawsuit", "ban", "investigation",
        "liquidation", "outflows", "token unlock", "sell-off", "selling pressure",
        "inflation", "downgrade"
    ],
    "strong_positive": [
        "etf approved", "etf approval", "approved solana etf", "mainnet upgrade live",
        "major integration", "billion-dollar", "billion dollar", "record inflows"
    ],
    "positive": [
        "etf filing", "etf application", "partnership", "integrates solana",
        "integration", "integrates", "mainnet upgrade", "upgrade goes live", "adoption",
        "institutional", "payments launch", "launches on solana", "record activity",
        "record volume", "fund inflows", "funding round"
    ],
}
SOL_RELEVANCE = {
    "network": ["solana network", "solana blockchain", "solana mainnet", "validator", "consensus", "outage", "solana foundation", "solana upgrade"],
    "market": ["sol", "solana etf", "solana price", "solana token", "solana futures", "solana staking"],
    "ecosystem": ["jupiter", "kamino", "raydium", "meteora", "drift", "jito", "phantom", "pump.fun", "solana defi", "solana ecosystem"],
}
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "SolanaNewsImpactMonitor/1.0 (RSS reader)"})


def _parse_date(value):
    if not value:
        return None
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None


def _clean(value):
    return re.sub(r"\\s+", " ", re.sub(r"<[^>]+>", " ", value or "")).strip()


def _item_id(item):
    basis = (item.get("link", "") + "|" + item.get("title", "")).lower()
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:24]


def fetch_recent_news(now=None):
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=MAX_AGE_HOURS)
    found = {}
    for category, url in FEEDS:
        try:
            response = SESSION.get(url, timeout=(4, 10))
            response.raise_for_status()
            root = ET.fromstring(response.content)
            for item in root.findall(".//item"):
                title = _clean(item.findtext("title"))
                link = (item.findtext("link") or "").strip()
                pub_date = _parse_date(item.findtext("pubDate"))
                description = _clean(item.findtext("description"))
                source_node = item.find("source")
                source = _clean(source_node.text if source_node is not None else "")
                if not title or not link or not pub_date or pub_date < cutoff or pub_date > now + timedelta(minutes=10):
                    continue
                row = {
                    "id": _item_id({"link": link, "title": title}),
                    "title": title[:350],
                    "url": link,
                    "published_at": pub_date.isoformat(),
                    "source": source or category,
                    "description": description[:500],
                    "feed_category": category,
                }
                found[row["id"]] = row
        except (requests.RequestException, ET.ParseError, ValueError) as exc:
            print(f"News feed error ({category}): {exc}")
    return sorted(found.values(), key=lambda x: x["published_at"], reverse=True)


def analyze_news(item):
    text = (item.get("title", "") + " " + item.get("description", "")).lower()
    hits = {}
    for level, terms in KEYWORDS.items():
        hits[level] = [term for term in terms if term in text]

    if hits["strong_negative"]:
        sentiment, score = "BEARISH", -3
        reason = "A serious security, availability, legal or solvency risk may undermine confidence."
    elif hits["negative"]:
        sentiment, score = "BEARISH", -2
        reason = "The headline suggests a risk, delay, selling pressure or a negative catalyst."
    elif hits["strong_positive"]:
        sentiment, score = "BULLISH", 3
        reason = "If confirmed and material, this could improve access, demand or confidence."
    elif hits["positive"]:
        sentiment, score = "BULLISH", 2
        reason = "This may support adoption, liquidity or sentiment, but implementation and scale matter."
    else:
        sentiment, score = "NEUTRAL / UNCLEAR", 0
        reason = "The headline alone does not provide a clear directional catalyst."

    relevance_hits = {
        label: [term for term in terms if term in text]
        for label, terms in SOL_RELEVANCE.items()
    }
    if relevance_hits["network"]:
        relevance = "DIRECT — Solana network / protocol"
        relevance_note = "Potentially relevant to SOL itself; verify official status and actual network impact."
    elif relevance_hits["market"]:
        relevance = "DIRECT — SOL market / regulation"
        relevance_note = "Could affect SOL demand, access or investor expectations if the report is accurate."
    elif relevance_hits["ecosystem"]:
        relevance = "INDIRECT — one Solana ecosystem project"
        relevance_note = "The first-order impact may be limited to that project; broader SOL impact depends on contagion, TVL and liquidity."
    else:
        relevance = "UNCERTAIN — broader crypto or weak Solana link"
        relevance_note = "Do not assume this moves SOL; compare BTC, market-wide sentiment and SOL's own price/volume."

    if any(term in text for term in ["rumor", "rumour", "reportedly", "unconfirmed", "alleged"]):
        confidence = "LOW — unconfirmed wording"
    elif item.get("source") and item.get("url", "").startswith("https://news.google.com/"):
        confidence = "MEDIUM — aggregator headline; open original source"
    else:
        confidence = "MEDIUM — headline classification only"

    if score > 0:
        expected = "hours to days; often priced in quickly"
    elif score < 0:
        expected = "immediate to days; risk can persist if confirmed"
    else:
        expected = "no reliable horizon until the facts and price reaction are checked"

    terms = hits["strong_negative"] + hits["negative"] + hits["strong_positive"] + hits["positive"]
    matched = ", ".join(terms[:5]) if terms else "no decisive keyword"
    concept = (
        f"Meaning: {reason}\n"
        f"Possible SOL impact: {relevance_note}\n"
        f"Direction: {sentiment} (rule score {score:+d}/3)\n"
        f"Relevance: {relevance}\n"
        f"Expected horizon: {expected}\n"
        f"Confidence: {confidence}\n"
        f"Why classified this way: {matched}. This is a headline-based heuristic, not full-article verification."
    )
    return {
        **item,
        "sentiment": sentiment,
        "impact_score": score,
        "relevance": relevance,
        "analysis": concept,
    }


def format_news_alert(item):
    published = item.get("published_at", "unknown time")
    return (
        "SOLANA NEWS IMPACT — RESEARCH, NOT A TRADE SIGNAL\n\n"
        f"Headline: {item.get('title', 'Untitled')}\n"
        f"Source: {item.get('source', 'Unknown')}\n"
        f"Published (UTC): {published}\n\n"
        f"{item.get('analysis', '')}\n\n"
        f"Original link: {item.get('url', '')}\n\n"
        "Do not trade on a headline alone. Confirm the original report and check SOL price, volume, BTC direction and whether the news is already priced in."
    )
