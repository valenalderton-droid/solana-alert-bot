
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, brier_score_loss

FEATURES = [
    "ret1", "ret3", "ret6", "ret12", "ret24",
    "ema_gap", "ema_gap34", "rsi", "atr_pct",
    "volatility_ratio", "volume_ratio", "volume_trend",
    "body_pct", "range_pct", "close_location", "breakout_distance",
    "price_vs_ema21", "rsi_short", "upper_wick_pct", "lower_wick_pct",
    "high_distance24", "low_distance24", "trend_persistence"
]


def add_features(candles):
    df = pd.DataFrame(candles).copy()
    if len(df) < 35:
        return pd.DataFrame()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    close = df["close"]
    for n in [1, 3, 6, 12, 24]:
        df[f"ret{n}"] = close.pct_change(n)

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema34 = close.ewm(span=34, adjust=False).mean()
    df["ema_gap"] = (ema9 / ema21) - 1
    df["ema_gap34"] = (ema9 / ema34) - 1
    df["price_vs_ema21"] = (close / ema21) - 1
    df["trend_persistence"] = (close > ema21).rolling(12).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = (100 - 100 / (1 + rs)) / 100
    short_gain = delta.clip(lower=0).rolling(6).mean()
    short_loss = (-delta.clip(upper=0)).rolling(6).mean()
    short_rs = short_gain / short_loss.replace(0, np.nan)
    df["rsi_short"] = (100 - 100 / (1 + short_rs)) / 100

    previous_close = close.shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs()
    ], axis=1).max(axis=1)

    df["atr_pct"] = tr.rolling(14).mean() / close
    atr_baseline = df["atr_pct"].rolling(48, min_periods=24).median()
    df["volatility_ratio"] = df["atr_pct"] / atr_baseline.replace(0, np.nan)

    vol_avg = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / vol_avg.replace(0, np.nan)
    prior_vol_avg = df["volume"].shift(5).rolling(20).mean()
    df["volume_trend"] = vol_avg / prior_vol_avg.replace(0, np.nan) - 1

    df["body_pct"] = (df["close"] - df["open"]) / close
    df["range_pct"] = (df["high"] - df["low"]) / close
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["close_location"] = (df["close"] - df["low"]) / candle_range
    df["upper_wick_pct"] = (df["high"] - df[["open", "close"]].max(axis=1)) / candle_range
    df["lower_wick_pct"] = (df[["open", "close"]].min(axis=1) - df["low"]) / candle_range
    prior_high = df["high"].shift(1).rolling(20).max()
    prior_high24 = df["high"].shift(1).rolling(24).max()
    prior_low24 = df["low"].shift(1).rolling(24).min()
    df["breakout_distance"] = close / prior_high - 1
    df["high_distance24"] = close / prior_high24 - 1
    df["low_distance24"] = close / prior_low24 - 1

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

            # Keep training labels aligned with the live signal's ATR-based
            # risk rules. Otherwise the model learns a different trade setup
            # from the one the bot actually sends.
            atr_pct = float(df.iloc[i]["atr_pct"])
            if not np.isfinite(atr_pct) or atr_pct <= 0:
                continue
            stop_pct = min(0.05, max(0.02, 1.5 * atr_pct))
            target_pct = max(0.04, 3 * atr_pct)
            target = entry * (1 + target_pct)
            stop = entry * (1 - stop_pct)
            label = None
            label_end = int(df.iloc[i + 4]["timestamp"])

            # Match the live evaluator: only use the next four hours. Missing
            # hourly bars or a timeout must not be silently labelled as a loss.
            signal_ts = int(df.iloc[i]["timestamp"])
            future = df.iloc[i + 1:i + 5]
            expected_times = [signal_ts + hour * 3600 for hour in range(1, 5)]
            future_times = [int(value) for value in future["timestamp"].tolist()]
            if len(future) != 4 or future_times != expected_times:
                continue

            for j in range(i + 1, i + 5):
                candle = df.iloc[j]
                hit_target = float(candle["high"]) >= target
                hit_stop = float(candle["low"]) <= stop

                # If both levels are touched in one candle, assume the stop
                # came first, matching the conservative live paper evaluator.
                if hit_target and hit_stop:
                    label = 0
                    break
                if hit_stop:
                    label = 0
                    break
                if hit_target:
                    label = 1
                    break

            # If neither level was reached within the horizon, it is a timeout,
            # not a negative example. Exclude it from this binary target-hit model.
            if label is None:
                continue

            values = df.iloc[i][FEATURES]
            if values.isna().any():
                continue

            rows.append({
                **{name: float(values[name]) for name in FEATURES},
                "timestamp": signal_ts,
                "label_end": label_end,
                "y": label
            })

    return pd.DataFrame(rows)


def train_model(candle_sets):
    data = build_training_rows(candle_sets)
    if data.empty:
        return None, {
            "ready": False,
            "labelled_examples": 0,
            "positive_labels": 0,
            "negative_labels": 0,
            "reason": "No usable labelled examples; check candle history and feature calculation"
        }

    positive_labels = int((data["y"] == 1).sum())
    negative_labels = int((data["y"] == 0).sum())
    if len(data) < 250:
        return None, {
            "ready": False,
            "labelled_examples": int(len(data)),
            "positive_labels": positive_labels,
            "negative_labels": negative_labels,
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
            "labelled_examples": int(len(data)),
            "positive_labels": positive_labels,
            "negative_labels": negative_labels,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_classes": int(train["y"].nunique()),
            "test_classes": int(test["y"].nunique()),
            "reason": "Insufficient class diversity or chronological test data"
        }

    model = Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(
            class_weight=None,
            C=0.5,
            max_iter=1500,
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
        "balanced_accuracy": round(float(balanced_accuracy_score(test["y"], predictions)), 4),
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
            class_weight=None,
            C=0.5,
            max_iter=1500,
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
    above_trend = float(row.get("price_vs_ema21", -1)) > 0
    trend_persistence = float(row.get("trend_persistence", 0)) >= 0.55
    breakout = close > prior_high
    rsi = float(row["rsi"])
    close_location = float(row.get("close_location", 0.5))
    upper_wick = float(row.get("upper_wick_pct", 1.0))
    volume_ok = float(row["volume_ratio"]) >= 1.5

    if not uptrend:
        return False, "Trend is not bullish"
    if not above_trend or not trend_persistence:
        return False, "Price is not holding above a persistent EMA21 trend"
    if not (0.35 <= rsi <= 0.78):
        return False, "RSI outside the selected range"
    if not breakout and (close_location < 0.50 or upper_wick > 0.65):
        return False, "Weak candle close or excessive upper wick"
    if not (volume_ok or breakout):
        return False, "No volume confirmation or breakout"

    return True, "Trend, candle quality and volume/breakout confirmation present"
