"""Lightweight, transparent ensemble of complementary trading styles.

These are rule-based indicators, not a claim that the bot can learn every strategy
from the internet. Scores are descriptive rankings and are not calibrated probabilities.
"""
import math


def _ema(values, period):
    if len(values) < period:
        return None
    alpha = 2.0 / (period + 1.0)
    value = sum(values[:period]) / period
    for item in values[period:]:
        value = alpha * item + (1.0 - alpha) * value
    return value


def _clamp(value, low=0.0, high=100.0):
    return max(low, min(high, float(value)))


def evaluate_strategies(candles):
    """Score trend, breakout, momentum, mean-reversion and volume/flow setups."""
    if not isinstance(candles, list) or len(candles) < 50:
        return {"ready": False, "score": 0, "bullish_strategies": 0,
                "strategies": {}, "reason": "Need at least 50 valid candles"}

    try:
        closes = [float(c["close"]) for c in candles]
        highs = [float(c["high"]) for c in candles]
        lows = [float(c["low"]) for c in candles]
        volumes = [float(c.get("volume", 0) or 0) for c in candles]
        if any(not math.isfinite(x) or x <= 0 for x in closes):
            raise ValueError("invalid close")
        if any(not math.isfinite(x) or x <= 0 for x in highs + lows):
            raise ValueError("invalid high/low")
    except (KeyError, TypeError, ValueError, OverflowError):
        return {"ready": False, "score": 0, "bullish_strategies": 0,
                "strategies": {}, "reason": "Invalid candle data"}

    close = closes[-1]
    ema20 = _ema(closes[-100:], 20)
    ema50 = _ema(closes[-100:], 50)
    ema20_prev = _ema(closes[-101:-1], 20) if len(closes) > 51 else ema20
    returns = [(closes[-1] / closes[-1-n] - 1) * 100 for n in (4, 12, 24)]
    changes = [closes[i] / closes[i-1] - 1 for i in range(1, len(closes))]
    gains = [max(x, 0) for x in changes[-14:]]
    losses = [max(-x, 0) for x in changes[-14:]]
    avg_gain = sum(gains) / max(1, len(gains))
    avg_loss = sum(losses) / max(1, len(losses))
    rsi = 100.0 if avg_loss == 0 else 100 - (100 / (1 + avg_gain / avg_loss))
    prior_high = max(highs[-21:-1])
    prior_low = min(lows[-21:-1])
    recent_volume = sum(volumes[-5:]) / 5
    base_volume = sum(volumes[-25:-5]) / 20
    volume_ratio = recent_volume / base_volume if base_volume > 0 else 0
    mean20 = sum(closes[-20:]) / 20
    std20 = math.sqrt(sum((x - mean20) ** 2 for x in closes[-20:]) / 20)
    zscore = (close - mean20) / std20 if std20 > 0 else 0

    # Each strategy outputs a 0..100 bullish setup score. They intentionally
    # represent different hypotheses and should not be read as win probabilities.
    trend = 50
    if ema20 is not None and ema50 is not None:
        trend += 22 if close > ema20 else -18
        trend += 20 if ema20 > ema50 else -18
        trend += 8 if ema20_prev is not None and ema20 > ema20_prev else -5
    trend = _clamp(trend)

    breakout = 75 if close > prior_high else (45 if close >= prior_high * 0.985 else 20)
    if volume_ratio >= 1.3 and close > prior_high:
        breakout += 15
    breakout = _clamp(breakout)

    momentum = _clamp(50 + returns[0] * 5 + returns[1] * 1.5 + returns[2] * 0.5)
    if returns[0] > 0 and returns[1] > 0:
        momentum = _clamp(momentum + 8)

    # Mean reversion is bullish only after a pullback with momentum stabilizing;
    # it penalizes extreme overbought conditions rather than blindly buying dips.
    mean_reversion = 50
    if zscore < -1.5 and returns[0] > -2:
        mean_reversion = 72
    elif zscore < -0.8 and returns[0] > -1:
        mean_reversion = 62
    elif zscore > 1.8:
        mean_reversion = 25
    elif zscore > 1.0:
        mean_reversion = 38
    if rsi < 25:
        mean_reversion = _clamp(mean_reversion + 5)
    elif rsi > 75:
        mean_reversion = _clamp(mean_reversion - 15)

    flow = 45
    if volume_ratio >= 1.5:
        flow += 20
    elif volume_ratio >= 1.1:
        flow += 10
    if close > closes[-2]:
        flow += 10
    if close > prior_high and volume_ratio >= 1.2:
        flow += 20
    if close < closes[-2] and volume_ratio >= 1.5:
        flow -= 15
    flow = _clamp(flow)

    scores = {
        "trend_following": round(trend, 1),
        "breakout": round(breakout, 1),
        "momentum": round(momentum, 1),
        "mean_reversion": round(mean_reversion, 1),
        "volume_flow": round(flow, 1),
    }
    weights = {
        "trend_following": 0.25,
        "breakout": 0.25,
        "momentum": 0.20,
        "mean_reversion": 0.10,
        "volume_flow": 0.20,
    }
    score = sum(scores[k] * weights[k] for k in scores)
    bullish = sum(value >= 60 for value in scores.values())
    return {
        "ready": True,
        "score": round(score, 1),
        "bullish_strategies": bullish,
        "strategies": scores,
        "metrics": {
            "rsi14": round(rsi, 2),
            "return_4h_pct": round(returns[0], 3),
            "return_12h_pct": round(returns[1], 3),
            "return_24h_pct": round(returns[2], 3),
            "volume_ratio": round(volume_ratio, 3),
            "distance_from_20_candle_mean_z": round(zscore, 3),
        },
        "interpretation": "Heuristic ensemble score; not a probability or guarantee.",
    }


def rank_candidate(ensemble, model_probability, quality_score):
    """Composite ranking for already-qualified candidates, bounded 0..100."""
    ensemble_score = _clamp((ensemble or {}).get("score", 0))
    probability_score = _clamp(float(model_probability or 0) * 100)
    quality = _clamp(quality_score or 0)
    return round(0.45 * ensemble_score + 0.30 * probability_score + 0.25 * quality, 2)
