"""Transparent multi-strategy ensemble for hourly Solana candles.

Scores rank setups; they are NOT calibrated win probabilities and this module does
not autonomously learn from arbitrary web pages. New rules should be backtested
before being trusted with live capital.
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
    """Evaluate trend, breakout, momentum, mean-reversion, flow, structure and risk."""
    if not isinstance(candles, list) or len(candles) < 60:
        return {"ready": False, "score": 0, "bullish_strategies": 0,
                "strategies": {}, "risk_flags": ["insufficient_history"],
                "reason": "Need at least 60 valid hourly candles"}

    try:
        closes = [float(c["close"]) for c in candles]
        opens = [float(c["open"]) for c in candles]
        highs = [float(c["high"]) for c in candles]
        lows = [float(c["low"]) for c in candles]
        volumes = [float(c.get("volume", 0) or 0) for c in candles]
        timestamps = [int(c.get("timestamp", i * 3600)) for i, c in enumerate(candles)]
        if any(not math.isfinite(x) or x <= 0 for x in closes + opens + highs + lows):
            raise ValueError("invalid OHLC")
        if any(not math.isfinite(x) or x < 0 for x in volumes):
            raise ValueError("invalid volume")
        if any(h < max(o, c, l) or l > min(o, c, h)
               for o, h, l, c in zip(opens, highs, lows, closes)):
            raise ValueError("inconsistent OHLC")
    except (KeyError, TypeError, ValueError, OverflowError):
        return {"ready": False, "score": 0, "bullish_strategies": 0,
                "strategies": {}, "risk_flags": ["invalid_candles"],
                "reason": "Invalid candle data"}

    close = closes[-1]
    ema20 = _ema(closes[-100:], 20)
    ema50 = _ema(closes[-100:], 50)
    ema20_prev = _ema(closes[-101:-1], 20) if len(closes) > 51 else ema20
    ret = {n: (close / closes[-1-n] - 1) * 100 for n in (4, 12, 24, 48)}
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
    candle_range = max(highs[-1] - lows[-1], close * 1e-9)
    body = abs(close - opens[-1])
    upper_wick = highs[-1] - max(opens[-1], close)
    lower_wick = min(opens[-1], close) - lows[-1]
    true_ranges = []
    for i in range(1, len(closes)):
        true_ranges.append(max(
            highs[i] - lows[i], abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        ) / closes[i])
    atr_pct = 100 * sum(true_ranges[-14:]) / max(1, len(true_ranges[-14:]))
    prior_support = min(lows[-48:-1])
    prior_resistance = max(highs[-48:-1])
    # Timestamp gaps can make hourly return labels misleading.
    gaps = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
    irregular_candles = sum(1 for gap in gaps[-24:] if gap > 2 * 3600 or gap <= 0)

    trend = 50
    trend += 22 if close > ema20 else -18
    trend += 20 if ema20 > ema50 else -18
    trend += 8 if ema20_prev is not None and ema20 > ema20_prev else -5
    trend = _clamp(trend)

    breakout = 78 if close > prior_high else (48 if close >= prior_high * 0.985 else 20)
    if volume_ratio >= 1.3 and close > prior_high:
        breakout += 12
    breakout = _clamp(breakout)

    momentum = _clamp(50 + ret[4] * 4 + ret[12] * 1.5 + ret[24] * 0.5)
    if ret[4] > 0 and ret[12] > 0:
        momentum = _clamp(momentum + 8)
    if ret[4] > 12:
        momentum = _clamp(momentum - 15)  # avoid rewarding a vertical chase

    mean_reversion = 50
    if zscore < -1.5 and ret[4] > -2:
        mean_reversion = 72
    elif zscore < -0.8 and ret[4] > -1:
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
    flow += 20 if volume_ratio >= 1.5 else (10 if volume_ratio >= 1.1 else 0)
    flow += 10 if close > closes[-2] else 0
    flow += 20 if close > prior_high and volume_ratio >= 1.2 else 0
    flow -= 15 if close < closes[-2] and volume_ratio >= 1.5 else 0
    flow = _clamp(flow)

    # Swing structure: higher lows and close relative to recent support/resistance.
    structure = 50
    low_12 = min(lows[-12:])
    prior_low_12 = min(lows[-24:-12])
    structure += 18 if low_12 > prior_low_12 else -18
    structure += 18 if close > (prior_support + prior_resistance) / 2 else -10
    structure += 14 if close >= prior_resistance * 0.98 else 0
    structure = _clamp(structure)

    # Rejection candle: a lower wick and close in the upper half suggest buyers defended.
    close_location = (close - lows[-1]) / candle_range
    rejection = 50
    if lower_wick > body * 1.2 and close_location >= 0.65:
        rejection += 25
    if upper_wick > body * 1.5 and close_location < 0.5:
        rejection -= 25
    if close > opens[-1]:
        rejection += 8
    rejection = _clamp(rejection)

    # Risk score is high when volatility is extreme, price is stretched, or candles are gappy.
    risk = 15
    risk += 25 if atr_pct > 5 else (12 if atr_pct > 3 else 0)
    risk += 20 if rsi > 82 else (10 if rsi > 75 else 0)
    risk += 20 if ret[4] > 15 else (10 if ret[4] > 10 else 0)
    risk += 15 if irregular_candles >= 2 else 0
    risk = _clamp(risk)
    risk_score = 100 - risk

    scores = {
        "trend_following": round(trend, 1),
        "breakout": round(breakout, 1),
        "momentum": round(momentum, 1),
        "mean_reversion": round(mean_reversion, 1),
        "volume_flow": round(flow, 1),
        "market_structure": round(structure, 1),
        "candle_rejection": round(rejection, 1),
        "risk_control": round(risk_score, 1),
    }
    weights = {
        "trend_following": 0.20, "breakout": 0.18, "momentum": 0.16,
        "mean_reversion": 0.08, "volume_flow": 0.14,
        "market_structure": 0.12, "candle_rejection": 0.05,
        "risk_control": 0.07,
    }
    score = sum(scores[k] * weights[k] for k in scores)
    bullish = sum(scores[k] >= 60 for k in (
        "trend_following", "breakout", "momentum", "mean_reversion",
        "volume_flow", "market_structure", "candle_rejection"
    ))
    risk_flags = []
    if atr_pct > 5:
        risk_flags.append("extreme_volatility")
    if rsi > 82:
        risk_flags.append("overbought_rsi")
    if ret[4] > 15:
        risk_flags.append("possible_price_chase")
    if irregular_candles >= 2:
        risk_flags.append("irregular_hourly_candles")
    if close < ema50:
        risk_flags.append("below_ema50")
    if volume_ratio < 0.5:
        risk_flags.append("weak_recent_volume")

    return {
        "ready": True, "score": round(score, 1),
        "bullish_strategies": bullish, "strategies": scores,
        "risk_flags": risk_flags,
        "metrics": {
            "rsi14": round(rsi, 2),
            "return_4h_pct": round(ret[4], 3),
            "return_12h_pct": round(ret[12], 3),
            "return_24h_pct": round(ret[24], 3),
            "return_48h_pct": round(ret[48], 3),
            "volume_ratio": round(volume_ratio, 3),
            "atr_pct": round(atr_pct, 3),
            "distance_from_20_candle_mean_z": round(zscore, 3),
            "close_location_in_candle": round(close_location, 3),
            "irregular_hourly_candles_24h": irregular_candles,
        },
        "interpretation": "Heuristic ranking only; not a win probability or guarantee.",
    }


def rank_candidate(ensemble, model_probability, quality_score, liquidity_usd=0,
                   volume_24h_usd=0, entry_deviation_pct=0):
    """Rank qualified candidates with a modest liquidity/chase adjustment."""
    ensemble_score = _clamp((ensemble or {}).get("score", 0))
    probability_score = _clamp(float(model_probability or 0) * 100)
    quality = _clamp(quality_score or 0)
    base = 0.45 * ensemble_score + 0.30 * probability_score + 0.25 * quality

    liquidity = max(0.0, float(liquidity_usd or 0))
    volume = max(0.0, float(volume_24h_usd or 0))
    deviation = max(0.0, float(entry_deviation_pct or 0))
    # Scores do not replace the hard liquidity/quality gates; these are ranking adjustments.
    market_bonus = min(5.0, math.log10(1 + liquidity / 10000) * 1.5)
    market_bonus += min(3.0, math.log10(1 + volume / 10000) * 0.8)
    chase_penalty = min(10.0, deviation * 2.0)
    risk_flags = (ensemble or {}).get("risk_flags", [])
    risk_penalty = min(12.0, len(risk_flags) * 1.5)
    return round(_clamp(base + market_bonus - chase_penalty - risk_penalty), 2)
