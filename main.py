
import os
import json
import math
from pathlib import Path
from datetime import datetime, timezone

import requests

from market import (
    get_best_solana_pair, get_hourly_candles, utc_now,
    discover_solana_projects, is_allowed_non_meme_project
)
from adaptive_policy import derive_policy
from learner import train_model, latest_features, price_action_ok
from news_analyzer import fetch_recent_news, analyze_news, format_news_alert

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
        signal_time = int(signal["timestamp"])
        future = [
            candle for candle in candles
            if signal_time < int(candle["timestamp"]) <= signal_time + 4 * 3600
        ]
        # Require four consecutive hourly candles. A missing candle must not
        # make the evaluator silently treat a shorter/irregular window as 4h.
        expected_times = [signal_time + hour * 3600 for hour in range(1, 5)]
        future_by_time = {int(candle["timestamp"]): candle for candle in future}
        complete_horizon = all(ts in future_by_time for ts in expected_times)
        if not complete_horizon:
            latest_ts = int(candles[-1]["timestamp"]) if candles else signal_time
            # Keep waiting while data may still arrive; after 8h, close the
            # paper observation as a data gap rather than leave it pending forever.
            if latest_ts < signal_time + 8 * 3600:
                still_pending.append(signal)
                continue
            append_jsonl(OUTCOMES_FILE, {
                "signal_id": signal_id,
                "token_address": address,
                "name": signal.get("name", "Unknown"),
                "entry": float(signal.get("entry", 0) or 0),
                "result": "data_gap",
                "return_pct": None,
                "resolved_at": utc_now(),
                "method": "missing_hourly_candles"
            })
            state["stats"]["resolved"] += 1
            continue
        future = [future_by_time[ts] for ts in expected_times]

        entry_price = float(signal["entry"])
        stop_price = float(signal.get("stop", entry_price * 0.98))
        target_price = float(signal.get("target", entry_price * 1.04))
        if entry_price <= 0:
            continue

        exit_price = None
        exit_candle = None
        exit_method = "close_after_4h"
        for candle in future[:4]:
            hit_stop = float(candle["low"]) <= stop_price
            hit_target = float(candle["high"]) >= target_price
            # Conservative assumption when OHLC data cannot reveal which level hit first.
            if hit_stop and hit_target:
                exit_price = stop_price
                exit_candle = candle
                exit_method = "both_hit_same_candle_assume_stop"
                break
            if hit_stop:
                exit_price = stop_price
                exit_candle = candle
                exit_method = "stop_hit"
                break
            if hit_target:
                exit_price = target_price
                exit_candle = candle
                exit_method = "target_hit"
                break

        if exit_price is None:
            exit_candle = future[3]
            exit_price = float(exit_candle["close"])

        ret = exit_price / entry_price - 1
        outcome = {
            "signal_id": signal_id,
            "token_address": address,
            "name": signal["name"],
            "entry": entry_price,
            "stop": stop_price,
            "target": target_price,
            "exit": exit_price,
            "return_pct": round(ret * 100, 4),
            # A win means the defined target was hit before the stop. If neither
            # level was hit within four candles, keep it as a timeout rather
            # than relabelling a small positive close as a target-hit win.
            "result": (
                "win" if exit_method == "target_hit"
                else "loss" if exit_method in ("stop_hit", "both_hit_same_candle_assume_stop")
                else "timeout"
            ),
            "exit_timestamp": int(exit_candle["timestamp"]),
            "resolved_at": utc_now(),
            "method": exit_method
        }
        append_jsonl(OUTCOMES_FILE, outcome)

        state["stats"]["resolved"] += 1
        if outcome["result"] == "win":
            state["stats"]["wins"] += 1
        elif outcome["result"] == "loss":
            state["stats"]["losses"] += 1

    state["open_signals"] = still_pending


def main():
    DATA.mkdir(parents=True, exist_ok=True)

    config = read_json(PROJECTS_FILE, {"projects": []})

    # Combine any manually configured projects with automatic discovery.
    # Manual entries take precedence for the same mint address.
    configured_projects = config.get("projects", [])
    discovered_projects = discover_solana_projects(
        min_liquidity=MIN_LIQUIDITY,
        min_volume_24h=float(os.getenv("MIN_VOLUME_24H_USD", "50000")),
        min_age_days=int(os.getenv("MIN_POOL_AGE_DAYS", "30")),
        max_projects=int(os.getenv("MAX_DISCOVERED_PROJECTS", "20"))
    )
    combined = {}
    for project in discovered_projects:
        address = str(project.get("token_address", "")).strip()
        if address:
            combined[address] = project
    for project in configured_projects:
        address = str(project.get("token_address", "")).strip()
        if address:
            combined[address] = project
    # Hard safety boundary: configured projects cannot bypass the non-meme allowlist.
    candidates = list(combined.values())
    config["projects"] = [
        project for project in candidates
        if is_allowed_non_meme_project(project)
    ]
    rejected = len(candidates) - len(config["projects"])
    print(
        f"Project list: {len(configured_projects)} configured + "
        f"{len(discovered_projects)} discovered; "
        f"{len(config['projects'])} approved Solana utility/protocol candidates; "
        f"{rejected} rejected by non-meme allowlist."
    )

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

    # News is supplemental context: it never directly creates a buy/sell order.
    # Store IDs only after Telegram confirms delivery, so failed sends can retry.
    seen_news = state.setdefault("seen_news_ids", [])
    seen_news_set = set(seen_news)
    news_sent = 0
    for item in fetch_recent_news():
        if item["id"] in seen_news_set:
            continue
        analyzed = analyze_news(item)
        observations = state.setdefault("news_observations", [])
        observations.append({
            "observed_at": utc_now(),
            "published_at": analyzed.get("published_at"),
            "sentiment": analyzed.get("sentiment", "NEUTRAL / UNCLEAR"),
            "impact_score": int(analyzed.get("impact_score", 0)),
            "relevance": analyzed.get("relevance", "UNCERTAIN"),
            "title": analyzed.get("title", "")[:250],
            "url": analyzed.get("url", "")
        })
        state["news_observations"] = observations[-500:]
        print(
            "News analysis:",
            json.dumps({
                "title": analyzed["title"],
                "sentiment": analyzed["sentiment"],
                "impact_score": analyzed["impact_score"],
                "relevance": analyzed["relevance"]
            }, ensure_ascii=False)
        )
        if send_telegram(format_news_alert(analyzed)):
            seen_news.append(item["id"])
            seen_news_set.add(item["id"])
            news_sent += 1
        if news_sent >= int(os.getenv("MAX_NEWS_ALERTS_PER_RUN", "3")):
            break
    # Bound the state file while retaining enough history for deduplication.
    state["seen_news_ids"] = seen_news[-500:]

    # Adaptive policy learns cautiously from resolved paper outcomes and recent news.
    policy = derive_policy(
        outcomes=read_jsonl(OUTCOMES_FILE),
        news_observations=state.get("news_observations", []),
        base_probability=MIN_PROBABILITY,
        base_cooldown_hours=max(1, int(os.getenv("ALERT_COOLDOWN_HOURS", "6")))
    )
    print("Adaptive policy:", json.dumps(policy, ensure_ascii=False))

    store = load_candle_store()
    active_projects = []
    market_info = {}
    scan_diagnostics = {
        "projects_seen": 0,
        "not_approved_or_missing_address": 0,
        "no_eligible_pair": 0,
        "insufficient_candles": 0,
        "stale_market_data": 0,
        "market_errors": 0,
        "usable_projects": 0,
        "price_action_rejected": 0,
        "invalid_features": 0,
        "below_probability_threshold": 0,
        "invalid_atr": 0,
        "alerts_created": 0
    }

    for project in config.get("projects", []):
        scan_diagnostics["projects_seen"] += 1
        if not is_allowed_non_meme_project(project):
            scan_diagnostics["not_approved_or_missing_address"] += 1
            continue

        address = str(project.get("token_address", "")).strip()
        if not address or address.startswith("REPLACE_"):
            scan_diagnostics["not_approved_or_missing_address"] += 1
            continue

        try:
            pair = get_best_solana_pair(address, MIN_LIQUIDITY)
            if not pair:
                scan_diagnostics["no_eligible_pair"] += 1
                print(f"No eligible liquid Solana pair: {project.get('name')}")
                continue

            pool_address = pair.get("pairAddress")
            raw = get_hourly_candles(pool_address)
            candles = get_closed_candles(raw)
            if len(candles) < 35:
                scan_diagnostics["insufficient_candles"] += 1
                print(f"Not enough candles: {project.get('name')}")
                continue

            latest_candle_ts = int(candles[-1]["timestamp"])
            current_hour_ts = int(datetime.now(timezone.utc).timestamp() // 3600) * 3600
            candle_age_hours = (current_hour_ts - latest_candle_ts) / 3600
            if candle_age_hours > 3:
                scan_diagnostics["stale_market_data"] += 1
                print(
                    f"Stale hourly data for {project.get('name')}: "
                    f"latest candle is {candle_age_hours:.1f} hours old"
                )
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
            scan_diagnostics["usable_projects"] += 1
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
            scan_diagnostics["market_errors"] += 1
            print(f"Market data error for {project.get('name')}: {exc}")

    save_candle_store(store)

    # Resolve older observations before adding new ones.
    resolve_old_signals(state, store)

    # Send a compact performance digest at most once every 24 hours.
    now = datetime.now(timezone.utc)
    last_report = state.get("last_stats_report_utc")
    report_due = not last_report
    if last_report:
        try:
            previous_report = datetime.fromisoformat(last_report.replace("Z", "+00:00"))
            report_due = (now - previous_report).total_seconds() >= 24 * 3600
        except (ValueError, TypeError):
            report_due = True

    if report_due:
        outcomes = read_jsonl(OUTCOMES_FILE)
        predictions = read_jsonl(PREDICTIONS_FILE)
        window_start = now.timestamp() - 24 * 3600

        def timestamp_in_window(value):
            if not value:
                return False
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.timestamp() >= window_start
            except (ValueError, TypeError, OverflowError):
                return False

        recent_predictions = [
            item for item in predictions
            if timestamp_in_window(item.get("created_at"))
        ]
        recent_outcomes = [
            item for item in outcomes
            if timestamp_in_window(item.get("resolved_at"))
        ]
        wins = sum(1 for item in recent_outcomes if item.get("result") == "win")
        losses = sum(1 for item in recent_outcomes if item.get("result") == "loss")
        timeouts = sum(1 for item in recent_outcomes if item.get("result") == "timeout")
        data_gaps = sum(1 for item in recent_outcomes if item.get("result") == "data_gap")
        resolved = wins + losses
        win_rate = (wins / resolved * 100) if resolved else 0.0
        returns = [
            float(item["return_pct"]) for item in recent_outcomes
            if isinstance(item.get("return_pct"), (int, float))
        ]
        avg_return = (sum(returns) / len(returns)) if returns else 0.0
        digest = (
            "SOLANA BOT — LAST 24H PERFORMANCE\n\n"
            f"BUY-watch alerts recorded in last 24h: {len(recent_predictions)}\n"
            f"Target/stop outcomes in last 24h: {resolved} (wins: {wins}, losses: {losses}); timeouts: {timeouts}; data gaps: {data_gaps}\n"
            f"Target-hit rate among target/stop outcomes: {win_rate:.1f}%\n"
            f"Average return of alerts resolved in last 24h: {avg_return:.2f}%\n"
            f"Projects with usable market data this run: {len(active_projects)}\n"
            f"Time (UTC): {utc_now()}\n\n"
            "Win rate uses target/stop outcomes only; average return includes numeric resolved outcomes. Data gaps are excluded from average return. "
            "This is not a guarantee of future performance."
        )
        if send_telegram(digest):
            state["last_stats_report_utc"] = utc_now()

    candle_sets = [
        sorted(store[p["token_address"]].values(),
               key=lambda c: c["timestamp"])
        for p in active_projects
        if p["token_address"] in store
    ]

    model, metrics = train_model(candle_sets)
    print("Model evaluation:", json.dumps(metrics))
    print("Scan diagnostics before signals:", json.dumps(scan_diagnostics))

    if model is None:
        print("Model is not ready; no BUY alerts will be generated.")

        # Send a status message on first run and then at most every 6 hours,
        # so the owner can distinguish "no signal" from a broken Telegram setup.
        now = datetime.now(timezone.utc)
        last_status = state.get("last_status_message_utc")
        send_status = not last_status
        if last_status:
            try:
                previous = datetime.fromisoformat(last_status.replace("Z", "+00:00"))
                send_status = (now - previous).total_seconds() >= 6 * 3600
            except (ValueError, TypeError):
                send_status = True

        if send_status:
            status_message = (
                "SOLANA BOT STATUS — RUNNING\n\n"
                "The bot ran successfully, but the model is not ready yet, "
                "so no BUY alert was generated.\n"
                f"Model details: {json.dumps(metrics)}\n"
                f"Projects with usable market data: {len(active_projects)}\n"
                f"Time (UTC): {utc_now()}\n\n"
                "This is a status message, not a trading signal."
            )
            sent = send_telegram(status_message)
            print(f"Status heartbeat sent={sent}")
            if sent:
                state["last_status_message_utc"] = utc_now()

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
            scan_diagnostics["invalid_features"] += 1
            continue

        # Use the last completed candle as the signal timestamp.
        candle_ts = int(row["timestamp"])
        signal_id = f"{address}:{candle_ts}:BUY_WATCH"

        last_alert_ts = int(state["last_alerts"].get(address, 0) or 0)
        cooldown_hours = policy["cooldown_hours"]
        if last_alert_ts and candle_ts - last_alert_ts < cooldown_hours * 3600:
            continue

        pa_ok, pa_reason = price_action_ok(candles, row)
        if not pa_ok:
            scan_diagnostics["price_action_rejected"] += 1
            print(f"Price-action rejection {project.get('name')}: {pa_reason}")
            continue

        probability = float(model.predict_proba(X)[0][1])
        if probability < policy["min_probability"]:
            scan_diagnostics["below_probability_threshold"] += 1
            print(
                f"Probability rejection {project.get('name')}: "
                f"{probability:.3f} < adaptive {policy['min_probability']:.3f}"
            )
            continue

        entry = float(row["close"])
        atr_pct = float(row["atr_pct"])
        if not math.isfinite(atr_pct) or atr_pct <= 0:
            scan_diagnostics["invalid_atr"] += 1
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

        # Only record a new alert after Telegram confirms delivery. If delivery
        # fails, the same completed candle can be retried on a later run.
        if sent:
            scan_diagnostics["alerts_created"] += 1
            append_jsonl(PREDICTIONS_FILE, prediction)
            state["open_signals"].append({
                "signal_id": signal_id,
                "token_address": address,
                "name": prediction["name"],
                "timestamp": candle_ts,
                "entry": entry,
                "stop": stop,
                "target": target
            })
            state["last_alerts"][address] = candle_ts
            state["stats"]["signals"] += 1
        else:
            print(f"Telegram delivery failed; will retry signal next run: {signal_id}")

    print("Final scan diagnostics:", json.dumps(scan_diagnostics))
    if scan_diagnostics["alerts_created"] == 0:
        print(
            "No BUY-watch alert this run. This can be normal; inspect the "
            "diagnostic counters above to identify the strictest filter."
        )

    state["last_run_utc"] = utc_now()
    write_json_atomic(STATE_FILE, state)


if __name__ == "__main__":
    main()
