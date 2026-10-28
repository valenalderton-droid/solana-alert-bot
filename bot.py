
import json
import os
import re
import time
from datetime import datetime, timezone

import requests

API = "https://api.geckoterminal.com/api/v2"
NETWORK = "solana"
STATE_FILE = "state.json"

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

MIN_LIQUIDITY = 100_000
MIN_VOLUME_24H = 500_000
MAX_POOLS = 12

# فیلتر اولیه؛ تضمین نمی‌کند همه میم‌کوین‌ها حذف شوند.
MEME_WORDS = re.compile(
    r"meme|memecoin|doge|pepe|wojak|inu|shib|bonk|"
    r"\bwif\b|frog|pump.fun|pumpfun|trump|chad|degen",
    re.I,
)

HEADERS = {
    "accept": "application/json;version=20230203",
    "user-agent": "SolanaPriceActionBot/1.0",
}

session = requests.Session()
session.headers.update(HEADERS)


def api(path, params=None):
    response = session.get(
        API + path, params=params, timeout=25
    )
    response.raise_for_status()
    return response.json()


def num(value):
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def telegram(message):
    if not BOT_TOKEN or not CHAT_ID:
        raise RuntimeError("Telegram secrets are missing")

    response = requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        json={
            "chat_id": CHAT_ID,
            "text": message,
            "disable_web_page_preview": True,
        },
        timeout=25,
    )
    response.raise_for_status()

    if not response.json().get("ok"):
        raise RuntimeError("Telegram did not accept the message")


def discover_pools():
    # کشف از چند فهرست عمومی؛ این روش شامل تمام توکن‌های
    # موجود در اکوسیستم سولانا نمی‌شود.
    endpoints = [
        "/networks/solana/trending_pools",
        "/networks/solana/pools",
        "/networks/solana/new_pools",
    ]

    pools = {}

    for endpoint in endpoints:
        params = {"page": 1, "include": "base_token"}
        if endpoint.endswith("/pools"):
            params["sort"] = "h24_volume_usd_desc"

        try:
            data = api(endpoint, params).get("data", [])

            for item in data:
                attrs = item.get("attributes") or {}
                address = attrs.get("address")

                if address:
                    pools[address] = attrs

        except Exception as exc:
            print(f"Discovery error: {endpoint}: {exc}")

        time.sleep(1.5)

    return list(pools.values())


def volume_24h(pool):
    volumes = pool.get("volume_usd") or {}
    if isinstance(volumes, dict):
        return num(volumes.get("h24"))
    return 0.0


def eligible(pool):
    liquidity = num(pool.get("reserve_in_usd"))
    volume = volume_24h(pool)
    name = str(pool.get("name") or "")

    if liquidity < MIN_LIQUIDITY:
        return False

    if volume < MIN_VOLUME_24H:
        return False

    if MEME_WORDS.search(name):
        return False

    return True


def get_candles(address):
    result = api(
        f"/networks/{NETWORK}/pools/{address}/ohlcv/hour",
        {
            "aggregate": 1,
            "limit": 100,
            "currency": "usd",
            "token": "base",
        },
    )

    rows = (
        result.get("data", {})
        .get("attributes", {})
        .get("ohlcv_list", [])
    )

    candles = []

    for row in rows:
        if not isinstance(row, list) or len(row) < 6:
            continue

        try:
            candles.append({
                "time": int(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            })
        except (ValueError, TypeError):
            continue

    candles.sort(key=lambda candle: candle["time"])

    if len(candles) < 60:
        return []

    return candles


def ema(values, period):
    if len(values) < period:
        return 0.0

    alpha = 2 / (period + 1)
    result = sum(values[:period]) / period

    for value in values[period:]:
        result = alpha * value + (1 - alpha) * result

    return result


def analyze(candles):
    # آخرین کندل را حذف می‌کنیم تا کندل ناقص مبنای سیگنال نباشد.
    candles = candles[:-1]

    if len(candles) < 55:
        return None

    closes = [c["close"] for c in candles]
    volumes = [c["volume"] for c in candles]

    last = candles[-1]
    previous = candles[-21:-1]

    if len(previous) < 20:
        return None

    resistance = max(c["high"] for c in previous)
    support = min(c["low"] for c in candles[-11:-1])

    ema20 = ema(closes[-80:], 20)
    ema50 = ema(closes[-100:], 50)

    avg_volume = sum(volumes[-21:-1]) / 20
    volume_ratio = (
        last["volume"] / avg_volume if avg_volume > 0 else 0
    )

    change_6h = (
        (last["close"] / closes[-7] - 1) * 100
        if closes[-7] > 0 else 0
    )

    change_24h = (
        (last["close"] / closes[-25] - 1) * 100
        if closes[-25] > 0 else 0
    )

    buy = (
        last["close"] > resistance
        and last["close"] > ema20 > ema50
        and volume_ratio >= 1.3
        and change_6h > 0
    )

    sell = (
        last["close"] < support
        or (last["close"] < ema20 and change_6h < 0)
    )

    return {
        "time": last["time"],
        "price": last["close"],
        "support": support,
        "resistance": resistance,
        "volume_ratio": volume_ratio,
        "change_6h": change_6h,
        "change_24h": change_24h,
        "buy": buy,
        "sell": sell,
    }


def load_state():
    try:
        with open(STATE_FILE, encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"pools": {}}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, indent=2)


def format_alert(pool, signal, result):
    name = pool.get("name") or "Unknown token"
    candle_time = datetime.fromtimestamp(
        result["time"], timezone.utc
    ).strftime("%Y-%m-%d %H:%M UTC")

    if signal == "BUY":
        title = "🟢 هشدار خرید احتمالی"
        reason = "شکست مقاومت با روند صعودی و افزایش حجم"
    else:
        title = "🔴 هشدار فروش/خروج"
        reason = "ضعف روند یا شکست حمایت کوتاه‌مدت"

    address = pool.get("address", "")
    chart = f"https://www.geckoterminal.com/solana/pools/{address}"

    return (
        f"{title}\n"
        f"استخر: {name}\n"
        f"قیمت: ${result['price']:.8g}\n"
        f"حمایت مرجع: ${result['support']:.8g}\n"
        f"مقاومت مرجع: ${result['resistance']:.8g}\n"
        f"تغییر ۶ ساعت: {result['change_6h']:+.2f}%\n"
        f"تغییر ۲۴ ساعت: {result['change_24h']:+.2f}%\n"
        f"حجم نسبت به میانگین: {result['volume_ratio']:.2f}x\n"
        f"دلیل: {reason}\n"
        f"زمان کندل: {candle_time}\n"
        f"نمودار: {chart}\n"
        "هشدار احتمالی است و تضمین سود نیست."
    )


def main():
    state = load_state()
    state.setdefault("pools", {})

    discovered = discover_pools()
    candidates = [p for p in discovered if eligible(p)]

    candidates.sort(
        key=lambda p: (
            num(p.get("reserve_in_usd"))
            * max(volume_24h(p), 1)
        ),
        reverse=True,
    )

    candidates = candidates[:MAX_POOLS]
    print(
        f"Discovered {len(discovered)} pools; "
        f"analyzing {len(candidates)}."
    )

    for pool in candidates:
        address = pool.get("address")
        if not address:
            continue

        try:
            candles = get_candles(address)
            result = analyze(candles)

            if not result:
                continue

            entry = state["pools"].setdefault(
                address,
                {"position": "flat", "last_candle": 0},
            )

            if result["time"] <= int(entry["last_candle"]):
                continue

            signal = None

            if result["buy"] and entry["position"] != "long":
                signal = "BUY"
            elif result["sell"] and entry["position"] == "long":
                signal = "SELL"

            if signal:
                telegram(format_alert(pool, signal, result))

                entry["position"] = (
                    "long" if signal == "BUY" else "flat"
                )
                entry["last_candle"] = result["time"]

                # Save after a successful Telegram send.
                save_state(state)
                print(f"Sent {signal}: {pool.get('name')}")

            time.sleep(1.5)

        except Exception as exc:
            # خطاها فقط در لاگ GitHub ثبت می‌شوند.
            print(
                f"Skipped {address}: "
                f"{type(exc).__name__}: {exc}"
            )
            time.sleep(1.5)

    save_state(state)
    print("Finished. No routine Telegram reports sent.")


if __name__ == "__main__":
    main()

