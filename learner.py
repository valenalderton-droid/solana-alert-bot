
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
    "high_distance24", "low_distance24", "trend_persistence",
    "ret48", "ema_gap50", "price_vs_ema50", "trend_persistence24",
    "volatility_regime", "volume_zscore", "momentum_consistency",
    "drawdown24", "close_location_avg3"
]


def add_features(candles):
    df = pd.DataFrame(candles).copy()
    if len(df) < 35:
        return pd.DataFrame()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    close = df["close"]
    for n in [1, 3, 6, 12, 24, 48]:
        df[f"ret{n}"] = close.pct_change(n)

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema34 = close.ewm(span=34, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    df["ema_gap"] = (ema9 / ema21) - 1
    df["ema_gap34"] = (ema9 / ema34) - 1
    df["price_vs_ema21"] = (close / ema21) - 1
    df["ema_gap50"] = (ema21 / ema50) - 1
    df["price_vs_ema50"] = (close / ema50) - 1
    df["trend_persistence"] = (close > ema21).rolling(12).mean()
    df["trend_persistence24"] = (close > ema21).rolling(24).mean()

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
    df["volatility_regime"] = df["atr_pct"] / df["atr_pct"].rolling(168, min_periods=48).median().replace(0, np.nan)

    vol_avg = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / vol_avg.replace(0, np.nan)
    prior_vol_avg = df["volume"].shift(5).rolling(20).mean()
    df["volume_trend"] = vol_avg / prior_vol_avg.replace(0, np.nan) - 1
    volume_std = df["volume"].rolling(48, min_periods=24).std().replace(0, np.nan)
    df["volume_zscore"] = (df["volume"] - df["volume"].rolling(48, min_periods=24).mean()) / volume_std

    df["body_pct"] = (df["close"] - df["open"]) / close
    df["range_pct"] = (df["high"] - df["low"]) / close
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["close_location"] = (df["close"] - df["low"]) / candle_range
    df["close_location_avg3"] = df["close_location"].rolling(3).mean()
    df["momentum_consistency"] = (df["ret1"] > 0).rolling(12).mean()
    df["upper_wick_pct"] = (df["high"] - df[["open", "close"]].max(axis=1)) / candle_range
    df["lower_wick_pct"] = (df[["open", "close"]].min(axis=1) - df["low"]) / candle_range
    prior_high = df["high"].shift(1).rolling(20).max()
    prior_high24 = df["high"].shift(1).rolling(24).max()
    prior_low24 = df["low"].shift(1).rolling(24).min()
    df["breakout_distance"] = close / prior_high - 1
    df["high_distance24"] = close / prior_high24 - 1
    df["low_distance24"] = close / prior_low24 - 1
    df["drawdown24"] = close / prior_high24 - 1

    return df.replace([np.inf, -np.inf], np.nan)


def build_training_rows(candle_sets, horizon=4):
    rows = []

    for candles in candle_sets:
        df = add_features(candles)
        if df.empty:
            continue

        horizon = max(1, int(horizon))
        # A separate label horizon supports short through multi-day setups.
        for i in range(25, len(df) - horizon):
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
            # Keep training labels aligned with the live alert's minimum 2R target.
            target_pct = max(0.04, 3 * atr_pct, 2.0 * stop_pct)
            target = entry * (1 + target_pct)
            stop = entry * (1 - stop_pct)
            label = None
            label_end = int(df.iloc[i + horizon]["timestamp"])

            # Require a complete hourly horizon. Missing bars or a timeout
            # must not be silently labelled as a loss.
            signal_ts = int(df.iloc[i]["timestamp"])
            future = df.iloc[i + 1:i + horizon + 1]
            expected_times = [signal_ts + hour * 3600 for hour in range(1, horizon + 1)]
            future_times = [int(value) for value in future["timestamp"].tolist()]
            if len(future) != horizon or future_times != expected_times:
                continue

            for j in range(i + 1, i + horizon + 1):
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



def recency_sample_weights(timestamps, half_life_days=21):
    """Give newer observations more influence without discarding older regimes."""
    values = pd.to_numeric(pd.Series(timestamps), errors="coerce").to_numpy(dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("timestamps must be non-empty finite epoch seconds")
    half_life_seconds = max(1.0, float(half_life_days) * 86400.0)
    age_seconds = np.maximum(0.0, float(np.max(values)) - values)
    weights = np.power(2.0, -age_seconds / half_life_seconds)
    # Keep a floor so older market regimes still contribute to training.
    weights = np.maximum(weights, 0.20)
    return weights / float(np.mean(weights))


def walk_forward_validation(data, folds=3):
    """Evaluate on later, unseen time blocks; purge labels crossing each boundary."""
    if data.empty or "timestamp" not in data or "label_end" not in data:
        return {"folds": 0, "reason": "No timestamped rows for walk-forward validation"}

    unique_times = np.array(sorted(pd.to_numeric(data["timestamp"], errors="coerce").dropna().unique()))
    if len(unique_times) < 6:
        return {"folds": 0, "reason": "Too few distinct timestamps for walk-forward validation"}

    # Reserve the earliest 50% for initial training; validate on later time blocks.
    split_at = max(2, int(len(unique_times) * 0.50))
    validation_times = unique_times[split_at:]
    blocks = [block for block in np.array_split(validation_times, min(int(folds), len(validation_times))) if len(block)]
    results = []

    for block in blocks:
        start_time = int(block[0])
        end_time = int(block[-1])
        train = data[
            (data["timestamp"] < start_time) &
            (data["label_end"] < start_time)
        ]
        valid = data[
            (data["timestamp"] >= start_time) &
            (data["timestamp"] <= end_time)
        ]
        if len(train) < 100 or len(valid) < 20 or train["y"].nunique() < 2 or valid["y"].nunique() < 2:
            continue

        fold_model = Pipeline([
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(
                class_weight=None, C=0.5, max_iter=1500, random_state=42
            ))
        ])
        weights = recency_sample_weights(train["timestamp"])
        fold_model.fit(train[FEATURES], train["y"], logistic__sample_weight=weights)
        probs = fold_model.predict_proba(valid[FEATURES])[:, 1]
        preds = (probs >= 0.5).astype(int)
        results.append({
            "balanced_accuracy": float(balanced_accuracy_score(valid["y"], preds)),
            "brier_score": float(brier_score_loss(valid["y"], probs)),
            "precision": float(precision_score(valid["y"], preds, zero_division=0)),
            "rows": int(len(valid)),
            "start_timestamp": start_time,
            "end_timestamp": end_time
        })

    if not results:
        return {
            "folds": 0,
            "reason": "Not enough class-diverse data for valid chronological folds"
        }

    return {
        "folds": len(results),
        "mean_balanced_accuracy": round(float(np.mean([r["balanced_accuracy"] for r in results])), 4),
        "mean_brier_score": round(float(np.mean([r["brier_score"] for r in results])), 4),
        "mean_precision": round(float(np.mean([r["precision"] for r in results])), 4),
        "fold_details": [
            {**r,
             "balanced_accuracy": round(r["balanced_accuracy"], 4),
             "brier_score": round(r["brier_score"], 4),
             "precision": round(r["precision"], 4)}
            for r in results
        ],
        "method": "expanding chronological windows; labels crossing validation boundary purged"
    }

def _train_single_horizon(candle_sets, horizon=4):
    data = build_training_rows(candle_sets, horizon=horizon)
    if data.empty:
        return None, {
            "ready": False,
            "labelled_examples": 0,
            "positive_labels": 0,
            "negative_labels": 0,
            "reason": f"No usable labelled examples for {horizon}h horizon; check candle history and feature calculation",
            "horizon_hours": int(horizon)
        }

    positive_labels = int((data["y"] == 1).sum())
    negative_labels = int((data["y"] == 0).sum())
    if len(data) < 250:
        return None, {
            "ready": False,
            "labelled_examples": int(len(data)),
            "positive_labels": positive_labels,
            "negative_labels": negative_labels,
            "reason": f"Not enough labelled examples for {horizon}h horizon: {len(data)} (need 250)",
            "horizon_hours": int(horizon)
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
            "reason": "Insufficient class diversity or chronological test data",
            "horizon_hours": int(horizon)
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

    train_weights = recency_sample_weights(train["timestamp"])
    model.fit(
        train[FEATURES], train["y"],
        logistic__sample_weight=train_weights
    )
    probabilities = model.predict_proba(test[FEATURES])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "ready": True,
        "horizon_hours": int(horizon),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "accuracy": round(float(accuracy_score(test["y"], predictions)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(test["y"], predictions)), 4),
        "precision": round(float(
            precision_score(test["y"], predictions, zero_division=0)
        ), 4),
        "brier_score": round(float(
            brier_score_loss(test["y"], probabilities)
        ), 4),
        "evaluation_method": "chronological holdout; training labels purged across test boundary",
        "train_positive_rate": round(float(train["y"].mean()), 4),
        "test_positive_rate": round(float(test["y"].mean()), 4),
        "predicted_positive_rate": round(float(np.mean(predictions)), 4),
        "recency_weighting": "21-day half-life; minimum relative weight 0.20",
        "walk_forward": walk_forward_validation(data)
    }

    # Refuse to deploy a horizon when both the chronological holdout and
    # walk-forward evidence look no better than chance. This avoids treating
    # a technically trainable model as a useful trading model.
    wf = metrics.get("walk_forward", {})
    wf_folds = int(wf.get("folds", 0) or 0)
    wf_balanced_accuracy = wf.get("mean_balanced_accuracy")
    wf_brier = wf.get("mean_brier_score")
    holdout_direction_poor = metrics["balanced_accuracy"] < 0.50
    holdout_brier_poor = metrics["brier_score"] >= 0.25
    walk_forward_direction_poor = (
        wf_folds == 0 or wf_balanced_accuracy is None or wf_balanced_accuracy < 0.50
    )
    walk_forward_brier_poor = (
        wf_folds == 0 or wf_brier is None or wf_brier >= 0.25
    )
    if (
        holdout_direction_poor and walk_forward_direction_poor and
        holdout_brier_poor and walk_forward_brier_poor
    ):
        metrics["ready"] = False
        metrics["reason"] = (
            "Validation gate rejected this horizon: chronological holdout and "
            "walk-forward results did not demonstrate a reliable edge"
        )
        return None, metrics

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
    final_weights = recency_sample_weights(data["timestamp"])
    final_model.fit(
        data[FEATURES], data["y"],
        logistic__sample_weight=final_weights
    )

    return final_model, metrics


class MultiHorizonModel:
    """Weighted probability ensemble across available, independently validated horizons."""
    def __init__(self, models):
        self.models = models
        self.base_weights = {4: 0.30, 12: 0.22, 24: 0.18, 72: 0.15, 168: 0.10, 336: 0.05}
        total = sum(self.base_weights.get(h, 0.05) for h, _ in models)
        self.weights = {
            h: self.base_weights.get(h, 0.05) / total
            for h, _ in models
        }

    def predict_horizon_proba(self, X):
        return {
            f"{h}h": float(model.predict_proba(X)[0][1])
            for h, model in self.models
        }

    def predict_proba(self, X):
        probabilities = [
            self.weights[h] * model.predict_proba(X)[:, 1]
            for h, model in self.models
        ]
        positive = np.sum(np.vstack(probabilities), axis=0)
        return np.column_stack([1.0 - positive, positive])


def train_model(candle_sets):
    """Train separate 4h, 12h, 24h, 72h, 168h and 336h outcome models."""
    horizons = (4, 12, 24, 72, 168, 336)
    trained = []
    horizon_metrics = {}
    for horizon in horizons:
        model, metrics = _train_single_horizon(candle_sets, horizon=horizon)
        horizon_metrics[f"{horizon}h"] = metrics
        if model is not None and metrics.get("ready"):
            trained.append((horizon, model))

    if not trained:
        best = max(
            horizon_metrics.values(),
            key=lambda item: int(item.get("labelled_examples", 0))
        )
        return None, {
            **best,
            "ready": False,
            "horizons": horizon_metrics,
            "reason": "No horizon had enough labelled examples and valid chronological evaluation"
        }

    active = [horizon for horizon, _ in trained]
    return MultiHorizonModel(trained), {
        "ready": True,
        "active_horizons": [f"{h}h" for h in active],
        "horizons": horizon_metrics,
        "ensemble": "weighted average of independently validated horizon probabilities",
        "weights": {
            f"{h}h": round(weight, 4)
            for h, weight in MultiHorizonModel(trained).weights.items()
        }
    }


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
    above_medium_trend = float(row.get("price_vs_ema50", -1)) > 0
    trend_persistence = float(row.get("trend_persistence", 0)) >= 0.55
    medium_persistence = float(row.get("trend_persistence24", 0)) >= 0.50
    breakout = close > prior_high
    rsi = float(row["rsi"])
    close_location = float(row.get("close_location", 0.5))
    upper_wick = float(row.get("upper_wick_pct", 1.0))
    volume_ok = float(row["volume_ratio"]) >= 1.5

    if not uptrend:
        return False, "Trend is not bullish"
    if not above_trend or not trend_persistence or not above_medium_trend or not medium_persistence:
        return False, "Price is not holding above persistent EMA21 and EMA50 trends"
    if not (0.35 <= rsi <= 0.78):
        return False, "RSI outside the selected range"
    if not breakout and (close_location < 0.50 or upper_wick > 0.65):
        return False, "Weak candle close or excessive upper wick"
    if not (volume_ok or breakout):
        return False, "No volume confirmation or breakout"

    return True, "Trend, candle quality and volume/breakout confirmation present"
