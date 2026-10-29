
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, brier_score_loss

FEATURES = [
    "ret1", "ret3", "ret6", "ret12",
    "ema_gap", "rsi", "atr_pct",
    "volume_ratio", "body_pct", "range_pct"
]


def add_features(candles):
    df = pd.DataFrame(candles).copy()
    if len(df) < 35:
        return pd.DataFrame()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    close = df["close"]
    for n in [1, 3, 6, 12]:
        df[f"ret{n}"] = close.pct_change(n)

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    df["ema_gap"] = (ema9 / ema21) - 1

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = (100 - 100 / (1 + rs)) / 100

    previous_close = close.shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs()
    ], axis=1).max(axis=1)

    df["atr_pct"] = tr.rolling(14).mean() / close
    vol_avg = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / vol_avg.replace(0, np.nan)
    df["body_pct"] = (df["close"] - df["open"]) / close
    df["range_pct"] = (df["high"] - df["low"]) / close

    return df.replace([np.inf, -np.inf], np.nan)


def build_training_rows(candle_sets):
    rows = []

    for candles in candle_sets:
        df = add_features(candles)
        if df.empty:
            continue

        # The last four candles are needed to label earlier rows.
        for i in range(25, len(df) - 4):
            entry = float(df.iloc[i]["close"])
            if entry <= 0:
                continue

            target = entry * 1.03
            stop = entry * 0.98
            label = 0

            for j in range(i + 1, i + 5):
                candle = df.iloc[j]
                hit_target = float(candle["high"]) >= target
                hit_stop = float(candle["low"]) <= stop

                # Ambiguous same-candle hits count as unsuccessful.
                if hit_target and hit_stop:
                    label = 0
                    break
                if hit_stop:
                    label = 0
                    break
                if hit_target:
                    label = 1
                    break

            values = df.iloc[i][FEATURES]
            if values.isna().any():
                continue

            rows.append({
                **{name: float(values[name]) for name in FEATURES},
                "timestamp": int(df.iloc[i]["timestamp"]),
                "label_end": int(df.iloc[i + 4]["timestamp"]),
                "y": label
            })

    return pd.DataFrame(rows)


def train_model(candle_sets):
    data = build_training_rows(candle_sets)
    if len(data) < 250:
        return None, {
            "ready": False,
            "reason": f"Not enough labelled examples: {len(data)} (need 250)"
        }

    data = data.sort_values("timestamp").reset_index(drop=True)
    cutoff_index = int(len(data) * 0.8)
    cutoff_time = int(data.iloc[cutoff_index]["timestamp"])

    # Purge training labels that extend into the test period.
    train = data[
        (data["timestamp"] < cutoff_time) &
        (data["label_end"] < cutoff_time)
    ]
    test = data[data["timestamp"] >= cutoff_time]

    if (
        len(train) < 100 or len(test) < 30 or
        train["y"].nunique() < 2 or test["y"].nunique() < 2
    ):
        return None, {
            "ready": False,
            "reason": "Insufficient class diversity or chronological test data"
        }

    model = Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ])

    model.fit(train[FEATURES], train["y"])
    probabilities = model.predict_proba(test[FEATURES])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "ready": True,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "accuracy": round(float(accuracy_score(test["y"], predictions)), 4),
        "precision": round(float(
            precision_score(test["y"], predictions, zero_division=0)
        ), 4),
        "brier_score": round(float(
            brier_score_loss(test["y"], probabilities)
        ), 4)
    }

    # Refit on all labelled rows for the current prediction.
    final_model = Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ])
    final_model.fit(data[FEATURES], data["y"])

    return final_model, metrics


def latest_features(candles):
    df = add_features(candles)
    if df.empty:
        return None, None

    # Ignore rows with incomplete features.
    valid = df.dropna(subset=FEATURES)
    if valid.empty:
        return None, None

    row = valid.iloc[-1]
    return row[FEATURES].to_frame().T, row


def price_action_ok(candles, row):
    df = pd.DataFrame(candles)
    if len(df) < 25 or row is None:
        return False, "Not enough candles"

    close = float(row["close"])
    if close <= 0:
        return False, "Invalid close"

    prior_high = float(df.iloc[-21:-1]["high"].max())
    uptrend = float(row["ema_gap"]) > 0 and float(row["ret6"]) > 0
    breakout = close > prior_high
    rsi = float(row["rsi"])
    volume_ok = float(row["volume_ratio"]) >= 1.5

    if not uptrend:
        return False, "Trend is not bullish"
    if not (0.35 <= rsi <= 0.78):
        return False, "RSI outside the selected range"
    if not (volume_ok or breakout):
        return False, "No volume confirmation or breakout"

    return True, "Uptrend and confirmation present"
