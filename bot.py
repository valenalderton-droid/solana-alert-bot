import os
import json
import time
import traceback
from datetime import datetime, timezone

import requests
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

TOKENS_FILE = "tokens.json"
STATE_FILE = "state.json"

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

BASE_URL = "https://api.geckoterminal.com/api/v2"
HEADERS = {"accept": "application/json"}
ALERT_THRESHOLD = 0.65


def get_json(url, params=None):
    response = requests.get(
        url,
        headers=HEADERS,
        params=params,
        timeout=25
    )
    response.raise_for_status()
    return response.json()


def send_telegram(text):
    if not BOT_TOKEN or not CHAT_ID:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing"
        )

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    response = requests.post(
        url,
        json={
            "chat_id": CHAT_ID,
            "text": text,
            "disable_web_page_preview": True
        },
        timeout=25
    )
    response.raise_for_status()
    result = response.json()

    if not result.get("ok"):
        raise RuntimeError(
            "Telegram rejected the message: " +
            str(result.get("description", "unknown error"))
        )


def load_tokens():
    with open(TOKENS_FILE, "r", encoding="utf-8") as file:
        tokens = json.load(file)

    return [
        token for token in tokens
        if token.get("enabled", False)
        and token.get("category") == "project"
        and token.get("address")
    ]


def find_pool(token_address):
    url = f"{BASE_URL}/networks/solana/tokens/{token_address}/pools"
    data = get_json(url, {"page": 1}).get("data", [])

    candidates = []

    for item in data:
        attributes = item.get("attributes", {})
        try:
            liquidity = float(attributes.get("reserve_in_usd") or 0)
        except (TypeError, ValueError):
            liquidity = 0

        if liquidity >= 50000:
            candidates.append((liquidity, item))

    if not candidates:
        raise RuntimeError("No pool found with at least $50,000 liquidity")

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return candidates[0][1]


def get_ohlcv(pool_address):
    url = (
        f"{BASE_URL}/networks/solana/pools/"
        f"{pool_address}/ohlcv/hour"
    )
    result = get_json(
        url,
        {
            "aggregate": 1,
            "limit": 200,
            "currency": "usd",
            "token": "base"
        }
    )

    rows = result.get("data", {}).get("attributes", {}).get(
        "ohlcv_list", []
    )

    rows = sorted(rows, key=lambda row: row[0])

    if len(rows) < 40:
        raise RuntimeError(
            f"Not enough hourly candles: {len(rows)}"
        )

    # GeckoTerminal row: [timestamp, open, high, low, close, volume]
    candles = []
    for row in rows:
        if len(row) < 6:
            continue
        candles.append([
            int(row[0]),
            float(row[1]),
            float(row[2]),
            float(row[3]),
            float(row[4]),
            float(row[5])
        ])

    if len(candles) < 40:
        raise RuntimeError("Not enough valid candles")

    return candles


def pct_change(current, previous):
    if not previous:
        return 0.0
    return (current / previous - 1.0) * 100.0


def rsi_value(closes, period=14):
    values = np.asarray(closes, dtype=float)
    changes = np.diff(values)

    if len(changes) < period:
        return None

    gains = np.maximum(changes, 0)
    losses = np.maximum(-changes, 0)

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def make_features(candles, index):
    closes = np.array(
        [candle[4] for candle in candles[:index + 1]],
        dtype=float
    )
    volumes = np.array(
        [candle[5] for candle in candles[:index + 1]],
        dtype=float
    )

    if len(closes) < 25:
        return None

    current = closes[-1]
    r1 = pct_change(current, closes[-2])
    r3 = pct_change(current, closes[-4])
    r6 = pct_change(current, closes[-7])
    r24 = pct_change(current, closes[-25])

    ma6 = np.mean(closes[-6:])
    ma24 = np.mean(closes[-24:])

    volume_mean = np.mean(volumes[-24:])
    volume_ratio = volumes[-1] / volume_mean if volume_mean else 0

    candle = candles[index]
    candle_range = (
        (candle[2] - candle[3]) / current * 100
        if current else 0
    )

    rsi = rsi_value(closes)
    if rsi is None:
        rsi = 50.0

    return [
        r1,
        r3,
        r6,
        r24,
        (current / ma6 - 1) * 100 if ma6 else 0,
        (current / ma24 - 1) * 100 if ma24 else 0,
        rsi,
        volume_ratio,
        candle_range
    ]


def model_probability(candles):
    # Features use completed historical candles only.
    # Target: close six hours later is at least 1% higher.
    features = []
    labels = []

    last_train_index = len(candles) - 7

    for i in range(25, last_train_index):
        feature = make_features(candles, i)
        if feature is None:
            continue

        future_close = candles[i + 6][4]
        current_close = candles[i][4]
        label = int(future_close >= current_close * 1.01)

        features.append(feature)
        labels.append(label)

    if len(features) < 30 or len(set(labels)) < 2:
        return None, len(features)

    split = int(len(features) * 0.8)
    if split < 20 or split >= len(features):
        return None, len(features)

    x_train = np.asarray(features[:split], dtype=float)
    y_train = np.asarray(labels[:split], dtype=int)

    model = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    model.fit(x_train, y_train)

    latest_feature = make_features(candles, len(candles) - 1)
    if latest_feature is None:
        return None, len(features)

    probability = float(
        model.predict_proba([latest_feature])[0][
            list(model.named_steps["classifier"].classes_).index(1)
        ]
    )

    # This is an exploratory estimate, not a calibrated forecast.
    return probability, len(features)


def analyze_token(token):
    symbol = token.get("symbol", "UNKNOWN")
    address = token["address"]

    pool = find_pool(address)
    pool_attributes = pool.get("attributes", {})
    pool_address = pool_attributes.get("address")

    if not pool_address:
        raise RuntimeError("Pool address missing from API response")

    candles = get_ohlcv(pool_address)
    closes = [candle[4] for candle in candles]

    current_price = closes[-1]
    change_1h = pct_change(closes[-1], closes[-2])
    change_6h = pct_change(closes[-1], closes[-7])
    change_24h = pct_change(closes[-1], closes[-25])

    probability, sample_count = model_probability(candles)

    if probability is None:
        signal = "اطلاعات کافی برای مدل وجود ندارد"
    elif (
        probability >= ALERT_THRESHOLD
        and change_1h > 0
        and change_6h > 0
    ):
        signal = "شرایط هشدار اولیه برقرار است"
    else:
        signal = "فعلاً هشدار خرید صادر نشده"

    return {
        "symbol": symbol,
        "price": current_price,
        "change_1h": change_1h,
        "change_6h": change_6h,
        "change_24h": change_24h,
        "probability": probability,
        "sample_count": sample_count,
        "signal": signal,
        "liquidity": pool_attributes.get("reserve_in_usd", "نامشخص"),
        "pool_address": pool_address,
        "candle_time": candles[-1][0]
    }


def format_report(result):
    probability = result["probability"]

    if probability is None:
        probability_text = "قابل محاسبه نیست"
    else:
        probability_text = f"{probability * 100:.1f}%"

    candle_time = datetime.fromtimestamp(
        result["candle_time"],
        tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M UTC")

    return (
        f"📊 گزارش توکن {result['symbol']}\n\n"
        f"💵 قیمت آخرین کندل ساعتی: ${result['price']:.8g}\n"
        f"🕐 تغییر ۱ ساعته: {result['change_1h']:+.2f}%\n"
        f"🕕 تغییر ۶ ساعته: {result['change_6h']:+.2f}%\n"
        f"📅 تغییر ۲۴ ساعته: {result['change_24h']:+.2f}%\n"
        f"🤖 احتمال تخمینی رشد حداقل ۱٪ طی ۶ ساعت: {probability_text}\n"
        f"🧪 تعداد نمونه‌های مدل: {result['sample_count']}\n"
        f"💧 نقدینگی استخر: ${float(result['liquidity'] or 0):,.0f}\n"
        f"📍 وضعیت: {result['signal']}\n"
        f"🕒 زمان آخرین کندل: {candle_time}\n\n"
        "⚠️ این گزارش توصیه مالی نیست؛ احتمال مدل تضمین سود نیست."
    )


def main():
    tokens = load_tokens()

    if not tokens:
        raise RuntimeError(
            "No enabled project tokens found in tokens.json"
        )

    # Send a start message so a run never stays silent.
    send_telegram(
        "🔎 ربات تحقیق سولانا شروع به کار کرد.\n"
        f"تعداد توکن‌های فعال: {len(tokens)}"
    )

    errors = []

    for token in tokens:
        symbol = token.get("symbol", "UNKNOWN")

        try:
            result = analyze_token(token)
            send_telegram(format_report(result))
        except Exception as error:
            error_text = (
                f"⚠️ تحلیل {symbol} ناموفق بود.\n"
                f"نوع خطا: {type(error).__name__}\n"
                f"جزئیات: {str(error)[:700]}"
            )
            errors.append(symbol)
            try:
                send_telegram(error_text)
            except Exception:
                print("Could not send Telegram error report for", symbol)
            print("Analysis error for", symbol, ":", repr(error))

        time.sleep(2)

    if errors:
        print("Tokens with errors:", ", ".join(errors))
    else:
        print("All enabled tokens processed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("FATAL ERROR:", repr(error))
        traceback.print_exc()
        raise
