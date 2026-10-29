"""Conservative, bounded adaptation from recorded outcomes and news observations.

This module adjusts only signal strictness and cooldown. It never changes the
non-meme allowlist, makes trades, or claims that a small sample is reliable.
"""
from datetime import datetime, timezone, timedelta


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


def derive_policy(outcomes, news_observations, base_probability=0.60,
                  base_cooldown_hours=6, now=None):
    now = now or datetime.now(timezone.utc)
    probability = min(0.75, max(0.55, float(base_probability)))
    cooldown = min(12, max(3, int(base_cooldown_hours)))
    notes = ["base settings retained until enough evidence exists"]

    resolved = [
        row for row in outcomes
        if row.get("result") in ("win", "loss")
    ][-30:]
    if len(resolved) >= 12:
        win_rate = sum(row.get("result") == "win" for row in resolved) / len(resolved)
        if win_rate < 0.40:
            probability = min(0.75, probability + 0.05)
            cooldown = min(12, cooldown + 2)
            notes.append("recent outcomes weak: stricter probability and longer cooldown")
        elif win_rate > 0.62:
            probability = max(0.55, probability - 0.02)
            cooldown = max(3, cooldown - 1)
            notes.append("recent outcomes stronger: slightly more permissive threshold")
        else:
            notes.append("recent outcomes mixed: no performance adjustment")
    else:
        notes.append(f"only {len(resolved)} resolved target/stop outcomes; below 12-example minimum")

    recent_cutoff = now - timedelta(hours=24)
    recent_direct_bearish = 0
    for row in news_observations:
        observed = _parse_time(row.get("observed_at") or row.get("published_at"))
        if observed is None or observed < recent_cutoff or observed > now + timedelta(minutes=10):
            continue
        if row.get("sentiment") == "BEARISH" and str(row.get("relevance", "")).startswith("DIRECT"):
            recent_direct_bearish += 1
    if recent_direct_bearish >= 2:
        probability = min(0.75, probability + 0.03)
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
