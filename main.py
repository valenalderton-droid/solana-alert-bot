
import os
import json
import math
from pathlib import Path
from datetime import datetime, timezone

import requests

from market import get_best_solana_pair, get_hourly_candles, discover_solana_candidates, utc_now
from learner import train_model, latest_features, price_action_ok

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PROJECTS_FILE = DATA / "projects.json"
STATE_FILE = DATA / "state.json"
CANDLES_FILE = DATA / "candles.jsonl"
PREDICTIONS_FILE = DATA / "predictions.jsonl"
OUTCOMES_FILE = DATA / "outcomes.jsonl"

MIN_LIQUIDITY = float(os.getenv("MIN_LIQUIDITY_USD", "100000"))
MIN_PROBABILITY = float(os.getenv("MIN_PROBABILITY", "0.60"))
MAX_STOP_PCT = float(os.getenv("MAX_STOP_PCT", "0.05"))
TARGET_PCT = float(os.getenv("TARGET_PCT", "0.04"))
MAX_CANDLE_HISTORY = 1500


def read_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json_atomic(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    temp.replace(path)


def read_jsonl(path):
    result = []
    if not path.exists():
        return result

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return result


def write_jsonl_atomic(path, rows):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def send_telegram(message):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("Telegram secrets missing; message was not sent.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        response = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": message,
                "disable_web_page_preview": True
            },
            timeout=20
        )
        response.raise_for_status()
        return bool(response.json().get("ok"))
    except requests.RequestException as exc:
        print(f"Telegram error: {exc}")
        return False


def load_candle_store():
    # Address -> timestamp -> candle
    store = {}
    for row in read_jsonl(CANDLES_FILE):
        address = row.get("token_address")
        candle = row.get("candle")
        if address and candle and "timestamp" in candle:
            store.setdefault(address, {})[int(candle["timestamp"])] = candle
    return store


def save_candle_store(store):
    rows = []
    for address, candles in store.items():
        ordered = sorted(candles.values(), key=lambda c: c["timestamp"])
        ordered = ordered[-MAX_CANDLE_HISTORY:]
        store[address] = {int(c["timestamp"]): c for c in ordered}
        for candle in ordered:
            rows.append({
                "token_address": address,
                "candle": candle
            })
    write_jsonl_atomic(CANDLES_FILE, rows)


def get_closed_candles(raw_candles):
    # Do not use the currently forming hourly candle.
    current_hour = int(datetime.now(timezone.utc).timestamp() // 3600) * 3600
    return [
        c for c in raw_candles
        if int(c["timestamp"]) < current_hour
    ]


def resolve_old_signals(state, store):
    pending = state.get("open_signals", [])
    still_pending = []
    known_outcomes = {
        row.get("signal_id") for row in read_jsonl(OUTCOMES_FILE)
    }

    for signal in pending:
        signal_id = signal["signal_id"]
        if signal_id in known_outcomes:
            continue

        address = signal["token_address"]
        candles = sorted(
            store.get(address, {}).values(),
            key=lambda c: c["timestamp"]
        )
        exit_time = int(signal["timestamp"]) + 4 * 3600

        eligible = [
            c for c in candles
            if int(c["timestamp"]) >= exit_time
        ]
        if not eligible:
            still_pending.append(signal)
            continue

        exit_candle = eligible[0]
        exit_price = float(exit_candle["close"])
        entry_price = float(signal["entry"])
        if entry_price <= 0:
            continue

        ret = exit_price / entry_price - 1
        outcome = {
            "signal_id": signal_id,
            "token_address": address,
            "name": signal["name"],
            "entry": entry_price,
            "exit": exit_price,
            "return_pct": round(ret * 100, 4),
            "result": "win" if ret > 0 else "loss",
            "exit_timestamp": int(exit_candle["timestamp"]),
            "resolved_at": utc_now(),
            "method": "close_after_4h_not_stop_target_simulation"
        }
        append_jsonl(OUTCOMES_FILE, outcome)

        state["stats"]["resolved"] += 1
        if ret > 0:
            state["stats"]["wins"] += 1
        else:
            state["stats"]["losses"] += 1

    state["open_signals"] = still_pending


def main():
    DATA.mkdir(parents=True, exist_ok=True)

    config = read_json(PROJECTS_FILE, {"projects": []})
    # Auto-discover candidate tokens; configured projects are also considered.
    candidates = list(config.get("projects", []))
    seen_addresses = {str(p.get("token_address", "")).strip() for p in candidates}
    for candidate in discover_solana_candidates(limit=int(os.getenv("MAX_DISCOVERY_CANDIDATES", "25"))):
        if candidate["token_address"] not in seen_addresses:
            candidates.append(candidate)
            seen_addresses.add(candidate["token_address"])
    print(f"Candidate tokens: {len(candidates)} (manual + discovered)")
    state = read_json(STATE_FILE, {
        "last_alerts": {},
        "open_signals": [],
        "stats": {
            "signals": 0,
            "resolved": 0,
            "wins": 0,
            "losses": 0
        },
        "last_run_utc": None
    })

    state.setdefault("last_alerts", {})
    state.setdefault("open_signals", [])
    state.setdefault("stats", {
        "signals": 0, "resolved": 0, "wins": 0, "losses": 0
    })

    store = load_candle_store()
    active_projects = []
    market_info = {}

    for project in candidates:
        # Manually configured entries must explicitly opt in; discovery entries are filtered in market.py.
        if project.get("discovered_from") is None and not project.get("approved_non_meme", False):
            continue

        address = str(project.get("token_address", "")).strip()
        if not address or address.startswith("REPLACE_"):
            continue

        try:
            pair = get_best_solana_pair(address, MIN_LIQUIDITY)
            if not pair:
                print(f"No eligible liquid Solana pair: {project.get('name')}")
                continue

            base_token = pair.get("baseToken") or {}
            project["name"] = project.get("name") or base_token.get("name") or "Unknown"
            project["symbol"] = project.get("symbol") or base_token.get("symbol") or ""
            pool_address = pair.get("pairAddress")
            raw = get_hourly_candles(pool_address)
            candles = get_closed_candles(raw)
            if len(candles) < 35:
                print(f"Not enough candles: {project.get('name')}")
                continue

            address_store = store.setdefault(address, {})
            for candle in candles:
                address_store[int(candle["timestamp"])] = candle

            merged = sorted(
                address_store.values(),
                key=lambda c: c["timestamp"]
            )[-MAX_CANDLE_HISTORY:]
            store[address] = {
                int(c["timestamp"]): c for c in merged
            }

            active_projects.append(project)
            market_info[address] = {
                "pair": pair,
                "candles": merged,
                "price": float(pair.get("priceUsd") or candles[-1]["close"]),
                "liquidity": float(
                    (pair.get("liquidity") or {}).get("usd") or 0
                ),
                "volume24h": float(
                    (pair.get("volume") or {}).get("h24") or 0
                )
            }

        except Exception as exc:
            print(f"Market data error for {project.get('name')}: {exc}")

    save_candle_store(store)

    # Resolve older observations before adding new ones.
    resolve_old_signals(state, store)

    candle_sets = [
        sorted(store[p["token_address"]].values(),
               key=lambda c: c["timestamp"])
        for p in active_projects
        if p["token_address"] in store
    ]

    model, metrics = train_model(candle_sets)
    print("Model evaluation:", json.dumps(metrics))

    if model is None:
        print("Model is not ready; no BUY alerts will be generated.")
        # A throttled heartbeat distinguishes a healthy scan from a silent failure.
        now = datetime.now(timezone.utc).timestamp()
        last_heartbeat = float(state.get("last_heartbeat_epoch", 0) or 0)
        if now - last_heartbeat >= 6 * 3600:
            message = (
                "SOLANA BOT STATUS\\n"
                f"Scan completed: {utc_now()}\\n"
                f"Candidates discovered: {len(candidates)}\\n"
                f"Eligible projects: {len(active_projects)}\\n"
                f"Model not ready: {metrics.get('reason', 'unknown reason')}\\n"
                "No trade signal was generated. This is a status message, not a BUY alert."
            )
            if send_telegram(message):
                state["last_heartbeat_epoch"] = now
        state["last_run_utc"] = utc_now()
        write_json_atomic(STATE_FILE, state)
        return

    for project in active_projects:
        address = project["token_address"]
        info = market_info.get(address)
        if not info:
            continue

        candles = info["candles"]
        X, row = latest_features(candles)
        if X is None or row is None:
            continue

        # Use the last completed candle as the signal timestamp.
        candle_ts = int(row["timestamp"])
        signal_id = f"{address}:{candle_ts}:BUY_WATCH"

        if state["last_alerts"].get(address) == candle_ts:
            continue

        pa_ok, pa_reason = price_action_ok(candles, row)
        if not pa_ok:
            continue

        probability = float(model.predict_proba(X)[0][1])
        if probability < MIN_PROBABILITY:
            continue

        entry = float(row["close"])
        atr_pct = float(row["atr_pct"])
        if not math.isfinite(atr_pct) or atr_pct <= 0:
            continue

        stop_pct = min(MAX_STOP_PCT, max(0.02, 1.5 * atr_pct))
        target_pct = max(TARGET_PCT, 3 * atr_pct)

        # Require a stop distance no greater than the configured cap.
        if stop_pct > MAX_STOP_PCT:
            continue

        stop = entry * (1 - stop_pct)
        target = entry * (1 + target_pct)

        prediction = {
            "signal_id": signal_id,
            "timestamp": candle_ts,
            "created_at": utc_now(),
            "token_address": address,
            "name": project.get("name", "Unknown"),
            "symbol": project.get("symbol", ""),
            "entry": entry,
            "stop": stop,
            "target": target,
            "stop_pct": round(stop_pct * 100, 3),
            "target_pct": round(target_pct * 100, 3),
            "probability": round(probability, 4),
            "liquidity_usd": info["liquidity"],
            "volume_24h_usd": info["volume24h"],
            "price_action": pa_reason,
            "model_metrics": metrics
        }

        append_jsonl(PREDICTIONS_FILE, prediction)
        state["open_signals"].append({
            "signal_id": signal_id,
            "token_address": address,
            "name": prediction["name"],
            "timestamp": candle_ts,
            "entry": entry
        })
        state["last_alerts"][address] = candle_ts
        state["stats"]["signals"] += 1

        message = (
            "SOLANA BUY WATCH — NOT AN AUTOMATIC TRADE\n\n"
            f"Project: {prediction['name']} ({prediction['symbol']})\n"
            f"Entry reference: ${entry:.8g}\n"
            f"Stop reference: ${stop:.8g} ({stop_pct * 100:.2f}%)\n"
            f"Target reference: ${target:.8g} ({target_pct * 100:.2f}%)\n"
            f"Model probability: {probability:.1%}\n"
            f"Liquidity: ${info['liquidity']:,.0f}\n"
            f"24h volume: ${info['volume24h']:,.0f}\n"
            f"Evaluation: {json.dumps(metrics)}\n"
            f"Mint: {address}\n"
            f"Time (UTC): {utc_now()}\n\n"
            "This is a research alert, not financial advice. "
            "Slippage, fees and execution can change results."
        )

        sent = send_telegram(message)
        print(f"Signal {signal_id}; telegram_sent={sent}")

    state["last_run_utc"] = utc_now()
    write_json_atomic(STATE_FILE, state)


if __name__ == "__main__":
    main()
