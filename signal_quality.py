"""Transparent signal-quality scoring for Solana ecosystem alerts.

This is a rule-based checklist, not a calibrated probability. It supplements,
never replaces, hard market-data, project-approval, model-validation and risk gates.
"""
import math

MIN_SIGNAL_QUALITY = 65


def score_signal(row, candles, liquidity_usd, volume_24h_usd,
                 global_context=None, stop_pct=None, target_pct=None):
    global_context = global_context or {}
    if row is None or len(candles) < 25:
        return {"score": 0, "grade": "REJECT", "reasons": ["Insufficient candles"], "passed": False}

    def number(key, default=None):
        try:
            value = float(row.get(key, default))
            return value if math.isfinite(value) else default
        except (TypeError, ValueError, OverflowError):
            return default

    score = 0
    reasons = []
    checks = []

    def award(condition, points, positive, negative):
        nonlocal score
        if condition:
            score += points
            checks.append({"check": positive, "points": points, "passed": True})
            reasons.append(positive)
        else:
            checks.append({"check": negative, "points": 0, "passed": False})
            reasons.append(negative)

    ema_gap = number("ema_gap", 0)
    ema_gap34 = number("ema_gap34", 0)
    price_vs_ema21 = number("price_vs_ema21", -1)
    price_vs_ema50 = number("price_vs_ema50", -1)
    persistence = number("trend_persistence", 0)
    persistence24 = number("trend_persistence24", 0)
    ret6 = number("ret6", -1)
    rsi = number("rsi", 0)
    volume_ratio = number("volume_ratio", 0)
    close_location = number("close_location", 0)
    upper_wick = number("upper_wick_pct", 1)
    breakout_distance = number("breakout_distance", -1)

    award(ema_gap > 0 and ema_gap34 > 0, 15, "Short and medium EMA trend aligned", "EMA trend disagreement")
    award(price_vs_ema21 > 0 and price_vs_ema50 > 0, 15, "Price above EMA21 and EMA50", "Price below a key trend EMA")
    award(persistence >= 0.60 and persistence24 >= 0.50, 10, "Trend persistence confirmed", "Trend persistence is weak")
    award(ret6 > 0, 8, "Positive six-candle momentum", "Six-candle momentum is not positive")
    award(0.40 <= rsi <= 0.72, 8, "RSI in a constructive, non-extreme zone", "RSI is weak or potentially overextended")
    award(volume_ratio >= 1.5, 12, "Relative volume confirms participation", "Relative volume confirmation is weak")
    award(close_location >= 0.60 and upper_wick <= 0.45, 8, "Strong close with controlled upper wick", "Candle close/wick quality is weak")

    closes = [float(c.get("close", 0)) for c in candles[-21:-1]]
    highs = [float(c.get("high", 0)) for c in candles[-21:-1]]
    current_close = number("close", 0)
    prior_high = max(highs) if highs else 0
    is_breakout = prior_high > 0 and current_close > prior_high
    award(is_breakout, 8, "Closed above the prior 20-candle high", "No confirmed 20-candle breakout")
    liq = float(liquidity_usd or 0)
    vol = float(volume_24h_usd or 0)
    award(liq >= 250000, 8, "Liquidity is robust for this filter", "Liquidity is only near the minimum")
    award(liq > 0 and vol / liq >= 0.5, 4, "24h volume/liquidity ratio is active", "Volume/liquidity activity is low")

    btc_change = global_context.get("btc_24h_pct")
    sol_change = global_context.get("sol_24h_pct")
    try:
        btc_change = float(btc_change)
        sol_change = float(sol_change)
        market_aligned = math.isfinite(btc_change) and math.isfinite(sol_change) and btc_change > -3 and sol_change > -4
    except (TypeError, ValueError, OverflowError):
        market_aligned = False
    award(market_aligned, 4, "BTC/SOL short-term backdrop is not sharply risk-off", "BTC/SOL backdrop is weak or unavailable")

    rr_ok = (
        isinstance(stop_pct, (int, float)) and isinstance(target_pct, (int, float))
        and stop_pct > 0 and target_pct / stop_pct >= 2.0
    )
    award(rr_ok, 5, "Gross reward/risk is at least 2:1", "Reward/risk below 2:1")

    score = min(100, max(0, score))
    passed = score >= MIN_SIGNAL_QUALITY
    grade = "A" if score >= 85 else "B" if score >= 75 else "C" if passed else "REJECT"
    return {
        "score": score,
        "grade": grade,
        "passed": passed,
        "minimum_required": MIN_SIGNAL_QUALITY,
        "checks": checks,
        "reasons": reasons
    }
