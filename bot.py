
import json
import os
import time
from pathlib import Path

import requests
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

BASE = Path(__file__).parent
TOKENS_FILE = BASE / "tokens.json"
STATE_FILE = BASE / "state.json"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

HEADERS = {"User-Agent": "SolanaResearchBot/1.0"}
API = "https://api.geckoterminal.com/api/v2"
HORIZON = 6
MIN_SAMPLES = 40
ALERT_THRESHOLD = 0.65


def get_json(url, params=None):
    response = requests.get(
        url, params=params, headers=HEADERS, timeout=25
    )
    response.raise_for_status()
    return response.json()


def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram secrets are missing.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    response = requests.post(
        url,
        json={"chat_id": TELEGRAM_CHAT_ID, "text": message},
        timeout=20
    )
    response.raise_for_status()

    if not response.json().get("ok"):
        raise RuntimeError("Telegram rejected the message")


def load_state():
    if not STATE_FILE.exists():
        return {"alerts": {}}

    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"alerts": {}}


def save_state(state):
    STATE_FILE.write_text(
        json.dumps(state, indent=2),
        encoding="utf-8"
    )


def discover_pool(token_address):
    url = f"{API}/networks/solana/tokens/{token_address}/pools"
    result = get_json(url, {"page": 1})
    pools = result.get("data", [])

    candidates = []
    for pool in pools:
        attrs = pool.get("attributes", {})
        liquidity = float(attrs.get("reserve_in_usd") or 0)
        address = attrs.get("address")

        if address and liquidity >= 50000:
            candidates.append((liquidity, address))

    if not candidates:
        return None

    # Choose the pool with the greatest reported USD liquidity.
    return max(candidates)[1]


def fetch_candles(pool_address):
    url = (
        f"{API}/networks/solana/pools/"
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

    rows = (
        result.get("data", {})
        .get("attributes", {})
        .get("ohlcv_list", [])
    )

    # GeckoTerminal returns newest candles first.
    rows = sorted(rows, key=lambda row: row[0])

    candles = []
    for row in rows:
        if len(row) < 6:
            continue

        try:
            ts, open_, high, low, close, volume = row[:6]
            values = list(map(float, [open_, high, low, close, volume]))
            if min(values[:4]) <= 0:
                continue

            candles.append({
                "ts": int(ts),
                "open": values[0],
                "high": values[1],
                "low": values[2],
                "close": values[3],
                "volume": max(values[4], 0)
            })
        except (TypeError, ValueError):
            continue

    return candles


def make_features(candles, index):
    # Features use only candles at or before this index.
    current = candles[index]
    close = current["close"]

    def change(period):
        if index < period:
            return 0.0
        old = candles[index - period]["close"]
        return (close / old - 1) * 100 if old else 0.0

    def mean_close(period):
        start = max(0, index - period + 1)
        values = [c["close"] for c in candles[start:index + 1]]
        return sum(values) / len(values)

    ma6 = mean_close(6)
    ma24 = mean_close(24)

    # RSI using a simple rolling average of gains/losses.
    changes = []
    for j in range(max(1, index - 13), index + 1):
        changes.append(candles[j]["close"] - candles[j - 1]["close"])

    gains = [max(x, 0) for x in changes]
    losses = [max(-x, 0) for x in changes]
    avg_gain = sum(gains) / max(len(gains), 1)
    avg_loss = sum(losses) / max(len(losses), 1)

    if avg_loss == 0:
        rsi = 100.0 if avg_gain > 0 else 50.0
    else:
        rsi = 100 - 100 / (1 + avg_gain / avg_loss)

    # Volume ratio compared with the preceding 24 candles.
    start = max(0, index - 23)
    volumes = [c["volume"] for c in candles[start:index + 1]]
    avg_volume = sum(volumes) / max(len(volumes), 1)
    volume_ratio = current["volume"] / max(avg_volume, 1e-9)

    return [
        change(1),
        change(3),
        change(6),
        change(24),
        (close / ma6 - 1) * 100 if ma6 else 0,
        (close / ma24 - 1) * 100 if ma24 else 0,
        rsi,
        min(volume_ratio, 100),
        (current["high"] / max(current["low"], 1e-12) - 1) * 100
    ]


def train_and_predict(candles):
    # Use only candles with a complete 6-hour future outcome.
    last_train_index = len(candles) - HORIZON - 1
    if last_train_index < 50:
        return None

    X, y = [], []

    for i in range(24, last_train_index + 1):
        future = candles[i + HORIZON]["close"]
        present = candles[i]["close"]

        if present <= 0:
            continue

        # Label: at least 1% price rise over the next 6 hours.
        label = int(future / present >= 1.01)
        X.append(make_features(candles, i))
        y.append(label)

    if len(X) < MIN_SAMPLES or len(set(y)) < 2:
        return None

    # Chronological holdout: keep the newest 20% out of training.
    split = int(len(X) * 0.8)
    if split < 30:
        return None

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced")
    )

    model.fit(X[:split], y[:split])

    # Basic out-of-sample accuracy, for monitoring only.
    test_predictions = model.predict(X[split:])
    accuracy = sum(
        int(a == b) for a, b in zip(test_predictions, y[split:])
    ) / max(len(test_predictions), 1)

    # Refit on all labelled historical samples for the current estimate.
    model.fit(X, y)
    probability = float(
        model.predict_proba([make_features(candles, len(candles) - 1)])[0][
            list(model.named_steps["logisticregression"].classes_).index(1)
        ]
    )

    return {
        "probability": probability,
        "accuracy": accuracy,
        "samples": len(X)
    }


def process_token(token, state):
    if not token.get("enabled", True):
        return

    symbol = token["symbol"].upper()
    address = token["address"]

    pool = discover_pool(address)
    if not pool:
        print(f"{symbol}: no pool with enough reported liquidity")
        return

    candles = fetch_candles(pool)
    if len(candles) < 80:
        print(f"{symbol}: not enough hourly candles")
        return

    result = train_and_predict(candles)
    if result is None:
        print(f"{symbol}: not enough usable training data")
        return

    latest = candles[-1]
    features = make_features(candles, len(candles) - 1)
    change_1h = features[0]
    change_6h = features[2]
    price = latest["close"]

    print(
        f"{symbol}: price={price:.8g}, "
        f"probability={result['probability']:.1%}, "
        f"holdout_accuracy={result['accuracy']:.1%}, "
        f"samples={result['samples']}"
    )

    # Conservative research filter, not an automatic trade.
    if not (
        result["probability"] >= ALERT_THRESHOLD
        and change_1h > 0
        and change_6h > 0
    ):
        return

    now = int(time.time())
    last_alert = state["alerts"].get(symbol, 0)

    if now - last_alert < 6 * 60 * 60:
        return

    message = (
        f"Solana project research alert: {symbol}\n"
        f"Estimated probability of 1%+ rise in 6h: "
        f"{result['probability']:.1%}\n"
        f"Price: ${price:.8g}\n"
        f"1h change: {change_1h:.2f}%\n"
        f"6h change: {change_6h:.2f}%\n"
        f"Holdout accuracy: {result['accuracy']:.1%}\n"
        f"Labelled samples: {result['samples']}\n"
        "Research alert only; not a guarantee or an order."
    )

    send_telegram(message)
    state["alerts"][symbol] = now


def main():
    tokens = json.loads(TOKENS_FILE.read_text(encoding="utf-8"))
    state = load_state()

    for token in tokens:
        try:
            process_token(token, state)
        except Exception as exc:
            # Do not print tokens, credentials, or full request headers.
            print(f"Error for {token.get('symbol', 'unknown')}: {exc}")

    save_state(state)


if __name__ == "__main__":
    main()
