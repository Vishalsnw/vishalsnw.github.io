#!/usr/bin/env python3
"""
CoinDCX ML V4 controlled experiment.
READ-ONLY: public market-data GET requests only.
Does not import bot_v3, open the production SQLite DB, read API secrets,
write config/model files, call private endpoints, or place orders.

Run from /root/coindcx-bot with the existing venv:
  /root/coindcx-bot/venv/bin/python ml_v4_experiment.py
"""
import time
import traceback
import requests
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_sample_weight

PUBLIC_URL = "https://public.coindcx.com"
CANDIDATE_ASSETS = [
    "BTC", "ETH", "SOL", "XRP", "BNB", "DOGE",
    "ADA", "LINK", "LTC", "BCH", "AVAX", "DOT",
]
FEATURES = [
    "ret1", "ret3", "ret6", "rsi", "ema_gap", "ema_slope",
    "bb_pos", "atr_pct", "vol_ratio", "upper_wick",
    "lower_wick", "body_ratio", "macd_norm",
]
LABEL_NAMES = ["SELL", "HOLD", "BUY"]
LABEL_IDS = [0, 1, 2]
BAND_FACTORS = [0.65, 0.85, 1.05]
RANDOM_STATE = 42
TIMEOUT = 15


def active_pairs(session):
    url = "https://api.coindcx.com/exchange/v1/derivatives/futures/data/active_instruments"
    response = session.get(
        url, params={"margin_currency_short_name[]": "INR"}, timeout=TIMEOUT
    )
    response.raise_for_status()
    raw = response.json()
    active = set()
    if isinstance(raw, list):
        for item in raw:
            pair = item.get("pair") if isinstance(item, dict) else item
            if pair:
                active.add(str(pair).upper())
    return [f"B-{asset}_USDT" for asset in CANDIDATE_ASSETS
            if f"B-{asset}_USDT" in active]


def fetch_candles(session, pair, limit=700):
    now = int(time.time())
    start = now - (limit * 5 * 60 + 3600)
    url = PUBLIC_URL + "/market_data/candlesticks"
    response = session.get(
        url,
        params={
            "pair": pair, "from": start, "to": now,
            "resolution": "5", "pcode": "f",
        },
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    raw = response.json()
    if isinstance(raw, dict):
        if raw.get("s") not in (None, "ok"):
            return pd.DataFrame()
        raw = raw.get("data", raw.get("candlesticks", []))
    frame = pd.DataFrame(raw)
    if frame.empty:
        return frame
    if "time" in frame.columns and "timestamp" not in frame.columns:
        frame = frame.rename(columns={"time": "timestamp"})
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    if not all(column in frame.columns for column in required):
        raise ValueError(f"{pair}: unexpected candle columns {list(frame.columns)}")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = (
        frame.dropna(subset=required)
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )
    if not frame.empty and frame.timestamp.iloc[-1] > 1e12:
        frame["timestamp"] = frame["timestamp"] / 1000.0
    return frame.tail(limit).reset_index(drop=True)


def make_features(src):
    if src is None or len(src) < 60:
        return pd.DataFrame()
    d = src.copy()
    c, h, low, o, v = d.close, d.high, d.low, d.open, d.volume
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-9))
    e9 = c.ewm(span=9, adjust=False).mean()
    e21 = c.ewm(span=21, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean()
    d["ema_gap"] = (e9 - e21) / (c + 1e-9)
    d["ema_slope"] = (e21 - e50) / (c + 1e-9)
    d["ret1"] = c.pct_change(1)
    d["ret3"] = c.pct_change(3)
    d["ret6"] = c.pct_change(6)
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    d["bb_pos"] = (c - (mid - 2 * sd)) / (4 * sd + 1e-9)
    tr = pd.concat(
        [h - low, (h - c.shift()).abs(), (low - c.shift()).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    d["atr"] = atr
    d["atr_pct"] = atr / (c + 1e-9)
    d["vol_ratio"] = v / (v.rolling(20).mean() + 1e-9)
    rng = (h - low).replace(0, np.nan)
    d["upper_wick"] = (h - np.maximum(o, c)) / rng
    d["lower_wick"] = (np.minimum(o, c) - low) / rng
    d["body_ratio"] = (c - o).abs() / rng
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["macd_norm"] = macd / (c + 1e-9)
    return d.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def label_variant(features, factor):
    # Same 3-candle forward horizon and ATR-based label rule as V3.
    # Only the ATR band multiplier changes in this controlled label experiment.
    future = features.close.shift(-3) / features.close - 1
    band = (features.atr_pct * 0.85 * factor).clip(lower=0.0018, upper=0.012)
    labels = pd.Series(1, index=features.index, dtype=int)
    labels[future > band] = 2
    labels[future < -band] = 0
    valid = future.notna()
    x = features.loc[valid, FEATURES].copy()
    y = labels.loc[valid].copy()
    return x, y


def split_pair(x, y):
    # Chronological 70/15/15 split; purge 3 bars at both boundaries.
    n = len(x)
    a = int(n * 0.70)
    b = int(n * 0.85)
    if a <= 100 or b - a < 25 or n - b < 25:
        return None
    return (
        x.iloc[:a - 3], y.iloc[:a - 3],
        x.iloc[a:b - 3], y.iloc[a:b - 3],
        x.iloc[b:], y.iloc[b:],
    )


def build_dataset(pair_frames, factor):
    parts = {"train_x": [], "train_y": [], "val_x": [], "val_y": [],
             "test_x": [], "test_y": []}
    pair_counts = {}
    for pair, features in pair_frames.items():
        x, y = label_variant(features, factor)
        split = split_pair(x, y)
        if split is None:
            print(f"  {pair}: skipped; not enough rows for 70/15/15 split")
            continue
        xtr, ytr, xv, yv, xt, yt = split
        parts["train_x"].append(xtr)
        parts["train_y"].append(ytr)
        parts["val_x"].append(xv)
        parts["val_y"].append(yv)
        parts["test_x"].append(xt)
        parts["test_y"].append(yt)
        pair_counts[pair] = {
            "all": len(y), "train": len(ytr), "validation": len(yv),
            "test": len(yt), "labels": y.value_counts().reindex(LABEL_IDS, fill_value=0).to_dict()
        }
    if not parts["train_x"]:
        return None
    return {
        key: pd.concat(value) for key, value in parts.items()
    } | {"pair_counts": pair_counts}


def fit_and_score(data, balanced):
    xtr, ytr = data["train_x"], data["train_y"]
    xv, yv = data["val_x"], data["val_y"]
    if ytr.nunique() < 3 or yv.nunique() < 2:
        return None
    model = HistGradientBoostingClassifier(
        max_iter=120, learning_rate=0.05, max_leaf_nodes=15,
        l2_regularization=1.5, random_state=RANDOM_STATE,
    )
    weights = compute_sample_weight(class_weight="balanced", y=ytr) if balanced else None
    model.fit(xtr, ytr, sample_weight=weights)
    pred = model.predict(xv)
    return {
        "model": model,
        "accuracy": accuracy_score(yv, pred),
        "balanced_accuracy": balanced_accuracy_score(yv, pred),
        "predictions": pred,
        "actual": yv,
    }


def print_distribution(title, y):
    counts = y.value_counts().reindex(LABEL_IDS, fill_value=0)
    total = int(counts.sum())
    print(title)
    for label, name in zip(LABEL_IDS, LABEL_NAMES):
        count = int(counts[label])
        pct = 100 * count / total if total else 0
        print(f"  {name}: {count} ({pct:.1f}%)")


def report(title, result):
    pred, actual = result["predictions"], result["actual"]
    print(f"\n--- {title} ---")
    print(f"Accuracy:          {result['accuracy']:.4f}")
    print(f"Balanced accuracy: {result['balanced_accuracy']:.4f}")
    print("Confusion matrix (rows actual, columns predicted; SELL/HOLD/BUY):")
    print(confusion_matrix(actual, pred, labels=LABEL_IDS))
    print(classification_report(
        actual, pred, labels=LABEL_IDS, target_names=LABEL_NAMES,
        zero_division=0,
    ))
    print_distribution("Predicted distribution:", pd.Series(pred))


def main():
    print("=" * 68)
    print("COINDCX ML V4 CONTROLLED EXPERIMENT")
    print("READ-ONLY PUBLIC DATA | NO LIVE ORDERS | NO PRODUCTION FILE/DB CHANGES")
    print("=" * 68)
    session = requests.Session()
    session.headers.update({"User-Agent": "CoinDCX-ML-V4-Research/1.0"})
    pairs = active_pairs(session)
    if not pairs:
        raise RuntimeError("No active allowlisted INR futures pairs returned.")
    print(f"Active pairs ({len(pairs)}): {', '.join(pairs)}")

    pair_frames = {}
    for pair in pairs:
        print(f"\nFetching and preparing {pair}...")
        try:
            raw = fetch_candles(session, pair)
            features = make_features(raw)
            if len(features) < 180:
                print(f"  Skipped: only {len(features)} usable feature rows")
                continue
            pair_frames[pair] = features
            print(f"  Usable feature rows: {len(features)}")
        except Exception as exc:
            print(f"  ERROR: {exc}")

    if not pair_frames:
        raise RuntimeError("No pairs had enough usable candle data.")

    validation_results = []
    datasets = {}
    print("\n" + "=" * 68)
    print("EXPERIMENTS: 3 LABEL BANDS x 2 TRAINING METHODS")
    print("Validation is chronological; test split remains untouched until selection.")
    print("=" * 68)

    for factor in BAND_FACTORS:
        data = build_dataset(pair_frames, factor)
        if data is None:
            print(f"\nBand factor {factor:.2f}: insufficient combined data")
            continue
        datasets[factor] = data
        print(f"\nLABEL BAND FACTOR {factor:.2f}")
        print_distribution("Training labels:", data["train_y"])
        print_distribution("Validation labels:", data["val_y"])
        print_distribution("Held-out test labels (not used for selection):", data["test_y"])

        for balanced in (False, True):
            name = "CLASS_WEIGHTED" if balanced else "BASELINE"
            try:
                result = fit_and_score(data, balanced)
                if result is None:
                    print(f"{name}: skipped; missing classes in train/validation")
                    continue
                key = (factor, balanced)
                validation_results.append((result["balanced_accuracy"], key, result))
                report(f"VALIDATION band={factor:.2f} method={name}", result)
            except Exception as exc:
                print(f"{name} failed for band {factor:.2f}: {exc}")

    if not validation_results:
        raise RuntimeError("No experiment produced a valid validation result.")

    validation_results.sort(key=lambda item: item[0], reverse=True)
    best_score, best_key, best_result = validation_results[0]
    factor, balanced = best_key
    data = datasets[factor]
    method = "CLASS_WEIGHTED" if balanced else "BASELINE"

    print("\n" + "=" * 68)
    print("VALIDATION SELECTION")
    print(f"Selected candidate: band={factor:.2f}, method={method}")
    print(f"Validation balanced accuracy: {best_score:.4f}")
    print("This is an experiment winner, NOT production approval.")

    # Refit the selected recipe on train + validation, then evaluate once on
    # the final chronological holdout. No threshold is changed and no live model
    # or production configuration is touched.
    x_fit = pd.concat([data["train_x"], data["val_x"]])
    y_fit = pd.concat([data["train_y"], data["val_y"]])
    final_model = HistGradientBoostingClassifier(
        max_iter=120, learning_rate=0.05, max_leaf_nodes=15,
        l2_regularization=1.5, random_state=RANDOM_STATE,
    )
    final_weights = compute_sample_weight(class_weight="balanced", y=y_fit) if balanced else None
    final_model.fit(x_fit, y_fit, sample_weight=final_weights)
    test_pred = final_model.predict(data["test_x"])
    test_result = {
        "accuracy": accuracy_score(data["test_y"], test_pred),
        "balanced_accuracy": balanced_accuracy_score(data["test_y"], test_pred),
        "predictions": test_pred,
        "actual": data["test_y"],
    }
    report(f"FINAL HELD-OUT TEST band={factor:.2f} method={method}", test_result)

    print("\nPER-PAIR LABEL COUNTS FOR SELECTED BAND")
    for pair, counts in data["pair_counts"].items():
        c = counts["labels"]
        total = sum(c.values())
        parts = [
            f"{name}={c[label]} ({100*c[label]/total:.1f}%)"
            for label, name in zip(LABEL_IDS, LABEL_NAMES)
        ]
        print(f"{pair}: " + " | ".join(parts))

    print("\n" + "=" * 68)
    print("EXPERIMENT COMPLETED")
    print("No production bot, config, SQLite DB, or live position was changed.")
    print("Do not deploy based only on this result; repeat on later unseen periods.")
    print("=" * 68)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nEXPERIMENT ERROR:")
        traceback.print_exc()
        raise
