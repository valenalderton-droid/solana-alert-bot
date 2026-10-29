"""Bounded adaptive policy using paper outcomes and recent news.

The policy changes only alert strictness/cooldown. It never trades, mutates the
project allowlist, or treats a tiny sample as reliable. Outcomes must be
simulated/observed results, not assumed live fills.
"""
from datetime import datetime, timezone, timedelta
import math


def _parse_time(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError):
        return None


def _usable_outcomes(outcomes):
    """Use numeric outcomes when available, including timeouts; exclude data gaps."""
    usable = []
    for row in outcomes:
        if not isinstance(row, dict) or row.get("result") == "data_gap":
            continue
        value = row.get("return_pct")
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            value = None
        if value is not None and math.isfinite(value):
            usable.append({"return_pct": value, "result": row.get("result")})
        elif row.get("result") in ("win", "loss"):
            usable.append({"return_pct": None, "result": row["result"]})
    return usable[-30:]


def derive_policy(outcomes, news_observations, base_probability=0.60,
                  base_cooldown_hours=6, now=None, research_observations=None):
    now = now or datetime.now(timezone.utc)
    probability = min(0.75, max(0.55, float(base_probability)))
    cooldown = min(12, max(3, int(base_cooldown_hours)))
    notes = ["bounded adaptation enabled; project/network safety rules remain fixed"]

    resolved = _usable_outcomes(outcomes)
    if len(resolved) >= 12:
        numeric_returns = [row["return_pct"] for row in resolved
                           if row["return_pct"] is not None]
        if len(numeric_returns) >= 12:
            win_rate = sum(value > 0 for value in numeric_returns) / len(numeric_returns)
            avg_return = sum(numeric_returns) / len(numeric_returns)
            notes.append(
                f"last {len(numeric_returns)} numeric outcomes: "
                f"positive-return rate={win_rate:.1%}, mean return={avg_return:.3f}%"
            )
        else:
            win_rate = sum(row["result"] == "win" for row in resolved) / len(resolved)
            avg_return = None
            notes.append(f"last {len(resolved)} target/stop labels: win rate={win_rate:.1%}")

        if win_rate < 0.40 or (avg_return is not None and avg_return < -0.25):
            probability = min(0.75, probability + 0.05)
            cooldown = min(12, cooldown + 2)
            notes.append("weak recent outcomes: stricter threshold and longer cooldown")
        elif win_rate > 0.62 and (avg_return is None or avg_return > 0):
            probability = max(0.55, probability - 0.02)
            cooldown = max(3, cooldown - 1)
            notes.append("stronger recent outcomes: small, bounded relaxation")
        else:
            notes.append("mixed/uncertain outcomes: retain current strictness")
    else:
        notes.append(
            f"only {len(resolved)} usable outcomes; need at least 12 before performance adaptation"
        )

    recent_cutoff = now - timedelta(hours=24)
    recent_direct_bearish = 0
    recent_global_bearish = 0
    recent_global_bullish = 0
    global_categories = ("Macro / global markets", "Geopolitics / global risk")
    for row in news_observations:
        if not isinstance(row, dict):
            continue
        observed = _parse_time(row.get("published_at") or row.get("observed_at"))
        if observed is None or observed < recent_cutoff or observed > now + timedelta(minutes=10):
            continue
        relevance = str(row.get("relevance", ""))
        category = str(row.get("feed_category", ""))
        sentiment = str(row.get("sentiment", ""))
        confidence = str(row.get("confidence", "")).upper()
        # Unconfirmed/rumor headlines remain visible in the news log, but do
        # not independently trigger automated risk-off policy changes.
        if confidence.startswith("LOW"):
            continue
        if sentiment == "BEARISH" and relevance.startswith("DIRECT"):
            recent_direct_bearish += 1
        if category in global_categories:
            if sentiment == "BEARISH":
                recent_global_bearish += 1
            elif sentiment == "BULLISH":
                recent_global_bullish += 1

    if recent_direct_bearish >= 2:
        probability = min(0.75, probability + 0.03)
        cooldown = min(12, cooldown + 1)
        notes.append("multiple direct bearish Solana news observations in 24h: temporary caution")
    else:
        notes.append(f"{recent_direct_bearish} direct bearish Solana news observations in 24h")

    if recent_global_bearish >= 2 and recent_global_bearish > recent_global_bullish:
        probability = min(0.75, probability + 0.04)
        cooldown = min(12, cooldown + 1)
        notes.append("global macro/geopolitical news skew is bearish: temporary caution")
    else:
        notes.append(
            f"global news in 24h: {recent_global_bearish} bearish, "
            f"{recent_global_bullish} bullish"
        )

    # A sharp ecosystem TVL contraction is a caution signal, not a buy/sell
    # prediction. It can tighten entries but can never relax the configured floor.
    tvl_rows = []
    for row in (research_observations or []):
        if not isinstance(row, dict) or row.get("metric") != "Solana DeFi TVL":
            continue
        observed = _parse_time(row.get("observed_at"))
        if observed is None or observed < now - timedelta(hours=48) or observed > now + timedelta(minutes=10):
            continue
        try:
            change = float(row.get("change_24h_pct"))
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(change):
            tvl_rows.append((observed, change))
    if tvl_rows:
        tvl_change = max(tvl_rows, key=lambda item: item[0])[1]
        if tvl_change <= -15:
            probability = min(0.75, probability + 0.05)
            cooldown = min(12, cooldown + 2)
            notes.append(f"DeFiLlama Solana TVL fell {tvl_change:.1f}% over ~24h: strong ecosystem caution")
        elif tvl_change <= -8:
            probability = min(0.75, probability + 0.03)
            cooldown = min(12, cooldown + 1)
            notes.append(f"DeFiLlama Solana TVL fell {tvl_change:.1f}% over ~24h: moderate ecosystem caution")
        else:
            notes.append(f"DeFiLlama Solana TVL 24h change {tvl_change:+.1f}%: no extra TVL risk penalty")
    else:
        notes.append("No recent valid Solana TVL research snapshot")

    # Require a fresh, explicitly numeric DeFiLlama percentage change.
    # These are ecosystem-wide risk guards, not directional price forecasts.
    metric_rows = {}
    for row in (research_observations or []):
        if not isinstance(row, dict) or row.get("kind") != "metric":
            continue
        observed = _parse_time(row.get("observed_at"))
        if observed is None or observed < now - timedelta(hours=30) or observed > now + timedelta(minutes=10):
            continue
        metric_name = str(row.get("metric", ""))
        try:
            change = float(row.get("change_24h_pct"))
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(change) or not -100 <= change <= 10000:
            continue
        current = metric_rows.get(metric_name)
        if current is None or observed > current[0]:
            metric_rows[metric_name] = (observed, change)

    dex = metric_rows.get("Solana DEX volume")
    perps = metric_rows.get("Solana perpetuals volume")
    if dex and dex[1] <= -25:
        probability = min(0.75, probability + 0.03)
        cooldown = min(12, cooldown + 1)
        notes.append(f"DeFiLlama Solana DEX volume change {dex[1]:.1f}%: reduced activity, stricter entries")
    if perps and perps[1] <= -30:
        probability = min(0.75, probability + 0.02)
        cooldown = min(12, cooldown + 1)
        notes.append(f"DeFiLlama Solana perpetuals volume change {perps[1]:.1f}%: derivatives activity is weakening")
    fees = metric_rows.get("Solana fees")
    if fees and fees[1] <= -25:
        probability = min(0.75, probability + 0.02)
        cooldown = min(12, cooldown + 1)
        notes.append(f"DeFiLlama Solana fees change {fees[1]:.1f}%: ecosystem usage may be weakening")
    if not any(name in metric_rows for name in ("Solana DEX volume", "Solana perpetuals volume", "Solana fees")):
        notes.append("No fresh numeric DEX/perpetuals/fees change data; no risk adjustment from these metrics")

    return {
        "min_probability": round(min(0.75, max(0.55, probability)), 3),
        "cooldown_hours": int(min(12, max(3, cooldown))),
        "resolved_outcomes_used": len(resolved),
        "direct_bearish_news_24h": recent_direct_bearish,
        "global_bearish_news_24h": recent_global_bearish,
        "global_bullish_news_24h": recent_global_bullish,
        "notes": notes,
        "guardrails": "non-meme allowlist and Solana-only market checks are fixed"
    }
