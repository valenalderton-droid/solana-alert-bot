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
                  base_cooldown_hours=6, now=None):
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
    for row in news_observations:
        if not isinstance(row, dict):
            continue
        observed = _parse_time(row.get("observed_at") or row.get("published_at"))
        if observed is None or observed < recent_cutoff or observed > now + timedelta(minutes=10):
            continue
        relevance = str(row.get("relevance", ""))
        if row.get("sentiment") == "BEARISH" and relevance.startswith("DIRECT"):
            recent_direct_bearish += 1

    if recent_direct_bearish >= 2:
        probability = min(0.75, probability + 0.03)
        cooldown = min(12, cooldown + 1)
        notes.append("multiple direct bearish Solana news observations in 24h: temporary caution")
    else:
        notes.append(f"{recent_direct_bearish} direct bearish Solana news observations in 24h")

    return {
        "min_probability": round(min(0.75, max(0.55, probability)), 3),
        "cooldown_hours": int(min(12, max(3, cooldown))),
        "resolved_outcomes_used": len(resolved),
        "direct_bearish_news_24h": recent_direct_bearish,
        "notes": notes,
        "guardrails": "non-meme allowlist and Solana-only market checks are fixed"
    }
