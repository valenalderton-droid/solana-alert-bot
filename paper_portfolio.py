"""Virtual USD portfolio; deterministic simulation only, no real orders."""
from datetime import datetime, timezone
import math

DEFAULT_STARTING_CASH = 10_000.0
RISK_PER_TRADE_PCT = 0.01
MAX_POSITION_PCT = 0.20
MAX_OPEN_POSITIONS = 5
ONE_WAY_COST_PCT = 0.0015
MAX_HOLD_SECONDS = 14 * 24 * 3600


def _positive(value):
    try:
        x = float(value)
        return x if math.isfinite(x) and x > 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def _now():
    return datetime.now(timezone.utc).isoformat()


def get_portfolio(state, starting_cash=DEFAULT_STARTING_CASH):
    p = state.get("paper_portfolio")
    if not isinstance(p, dict):
        cash = _positive(starting_cash) or DEFAULT_STARTING_CASH
        p = {"version": 1, "currency": "USD", "initial_cash_usd": cash,
             "cash_usd": cash, "open_positions": [], "closed_trades": [],
             "fees_paid_usd": 0.0, "peak_equity_usd": cash,
             "max_drawdown_pct": 0.0, "created_at": _now(), "updated_at": _now()}
        state["paper_portfolio"] = p
    p.setdefault("open_positions", [])
    p.setdefault("closed_trades", [])
    p.setdefault("fees_paid_usd", 0.0)
    p.setdefault("peak_equity_usd", p.get("initial_cash_usd", DEFAULT_STARTING_CASH))
    p.setdefault("max_drawdown_pct", 0.0)
    return p


def _equity(p, prices=None):
    total = float(p.get("cash_usd", 0))
    prices = prices or {}
    for pos in p["open_positions"]:
        price = _positive(prices.get(pos.get("token_address"))) or _positive(pos.get("last_price")) or float(pos["entry"])
        total += float(pos["quantity"]) * price
    return total


def portfolio_snapshot(state, market_info=None):
    p = get_portfolio(state)
    prices = {}
    for address, info in (market_info or {}).items():
        price = _positive((info or {}).get("price"))
        candles = (info or {}).get("candles") or []
        if price is None and candles:
            price = _positive(candles[-1].get("close"))
        if price:
            prices[address] = price
    equity = _equity(p, prices)
    peak = max(float(p.get("peak_equity_usd", equity)), equity)
    p["peak_equity_usd"] = peak
    drawdown = (peak - equity) / peak * 100 if peak > 0 else 0
    p["max_drawdown_pct"] = max(float(p.get("max_drawdown_pct", 0)), drawdown)
    p["updated_at"] = _now()
    realized = sum(float(t.get("pnl_usd", 0)) for t in p["closed_trades"])
    return {"initial_cash_usd": round(float(p["initial_cash_usd"]), 2),
            "cash_usd": round(float(p["cash_usd"]), 2), "equity_usd": round(equity, 2),
            "realized_pnl_usd": round(realized, 2),
            "unrealized_pnl_usd": round(equity - float(p["initial_cash_usd"]) - realized, 2),
            "return_pct": round((equity / float(p["initial_cash_usd"]) - 1) * 100, 3),
            "open_positions": len(p["open_positions"]), "closed_trades": len(p["closed_trades"]),
            "fees_paid_usd": round(float(p["fees_paid_usd"]), 2),
            "max_drawdown_pct": round(float(p["max_drawdown_pct"]), 3)}


def open_paper_trade(state, prediction, now=None):
    p = get_portfolio(state)
    if not isinstance(prediction, dict):
        return {"opened": False, "reason": "invalid_prediction"}
    address = str(prediction.get("token_address") or "").strip()
    signal_id = str(prediction.get("signal_id") or "").strip()
    entry, stop, target = (_positive(prediction.get(k)) for k in ("entry", "stop", "target"))
    if not address or not signal_id or not entry or not stop or not target:
        return {"opened": False, "reason": "missing_or_invalid_trade_fields"}
    if not stop < entry < target:
        return {"opened": False, "reason": "invalid_stop_target_geometry"}
    if any(x.get("token_address") == address for x in p["open_positions"]):
        return {"opened": False, "reason": "token_already_open"}
    if any(x.get("signal_id") == signal_id for x in p["closed_trades"]):
        return {"opened": False, "reason": "signal_already_traded"}
    if len(p["open_positions"]) >= MAX_OPEN_POSITIONS:
        return {"opened": False, "reason": "max_open_positions"}
    equity, cash = _equity(p), float(p["cash_usd"])
    stop_fraction = (entry - stop) / entry
    if not math.isfinite(stop_fraction) or stop_fraction <= 0:
        return {"opened": False, "reason": "invalid_risk"}
    notional = min(equity * RISK_PER_TRADE_PCT / stop_fraction,
                   equity * MAX_POSITION_PCT, cash / (1 + ONE_WAY_COST_PCT))
    if not math.isfinite(notional) or notional < 10:
        return {"opened": False, "reason": "insufficient_virtual_cash"}
    quantity = notional / entry
    fee = notional * ONE_WAY_COST_PCT
    p["cash_usd"] = round(cash - notional - fee, 8)
    p["fees_paid_usd"] = float(p["fees_paid_usd"]) + fee
    p["open_positions"].append({
        "signal_id": signal_id, "token_address": address,
        "name": str(prediction.get("name") or "Unknown"), "symbol": str(prediction.get("symbol") or ""),
        "timestamp": int(prediction.get("timestamp") or 0), "opened_at": now or _now(),
        "entry": entry, "stop": stop, "target": target, "quantity": quantity,
        "notional_usd": notional, "entry_fee_usd": fee, "last_price": entry,
        "probability": prediction.get("probability"), "rank_score": prediction.get("rank_score")})
    p["updated_at"] = now or _now()
    return {"opened": True, "signal_id": signal_id, "notional_usd": round(notional, 2),
            "quantity": quantity, "entry_fee_usd": round(fee, 4)}


def update_paper_portfolio(state, market_info, now=None):
    p = get_portfolio(state)
    now_dt = now or datetime.now(timezone.utc)
    if isinstance(now_dt, str):
        try:
            now_dt = datetime.fromisoformat(now_dt.replace("Z", "+00:00"))
        except ValueError:
            now_dt = datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)
    now_text = now_dt.astimezone(timezone.utc).isoformat()
    still_open = []
    for pos in p["open_positions"]:
        address = pos.get("token_address")
        info = (market_info or {}).get(address) or {}
        candles = sorted((c for c in (info.get("candles") or []) if isinstance(c, dict)),
                         key=lambda c: int(c.get("timestamp", 0) or 0))
        entry_ts, entry = int(pos.get("timestamp", 0) or 0), float(pos["entry"])
        exit_price = exit_reason = exit_ts = None
        for candle in candles:
            try:
                ts, low, high = int(candle["timestamp"]), float(candle["low"]), float(candle["high"])
                if ts <= entry_ts or not all(math.isfinite(v) for v in (low, high)) or low <= 0 or high < low:
                    continue
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            stop_hit, target_hit = low <= float(pos["stop"]), high >= float(pos["target"])
            if stop_hit:
                exit_price, exit_reason, exit_ts = float(pos["stop"]), ("stop_and_target_same_candle_assume_stop" if target_hit else "stop"), ts
                break
            if target_hit:
                exit_price, exit_reason, exit_ts = float(pos["target"]), "target", ts
                break
        latest = _positive(info.get("price"))
        if latest is None and candles:
            latest = _positive(candles[-1].get("close"))
        if latest:
            pos["last_price"] = latest
        if exit_price is None and entry_ts > 0 and candles and latest:
            latest_ts = int(candles[-1].get("timestamp", 0) or 0)
            if latest_ts - entry_ts >= MAX_HOLD_SECONDS:
                exit_price, exit_reason, exit_ts = latest, "14_day_time_exit", latest_ts
        if exit_price is None:
            still_open.append(pos)
            continue
        qty, entry_fee = float(pos["quantity"]), float(pos.get("entry_fee_usd", 0))
        proceeds = qty * exit_price
        exit_fee = proceeds * ONE_WAY_COST_PCT
        gross = (exit_price - entry) * qty
        net = gross - entry_fee - exit_fee
        p["cash_usd"] = round(float(p["cash_usd"]) + proceeds - exit_fee, 8)
        p["fees_paid_usd"] = float(p["fees_paid_usd"]) + exit_fee
        p["closed_trades"].append({
            "signal_id": pos.get("signal_id"), "token_address": address, "name": pos.get("name"),
            "symbol": pos.get("symbol"), "entry": entry, "exit": exit_price, "quantity": qty,
            "notional_usd": float(pos.get("notional_usd", 0)), "gross_pnl_usd": round(gross, 8),
            "pnl_usd": round(net, 8),
            "return_pct_net": round(net / max(float(pos.get("notional_usd", 0)), 1e-12) * 100, 6),
            "entry_fee_usd": entry_fee, "exit_fee_usd": exit_fee, "exit_reason": exit_reason,
            "exit_timestamp": exit_ts, "opened_at": pos.get("opened_at"), "closed_at": now_text,
            "probability": pos.get("probability"), "rank_score": pos.get("rank_score")})
    p["open_positions"] = still_open
    p["updated_at"] = now_text
    return portfolio_snapshot(state, market_info)
